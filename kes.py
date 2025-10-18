print("Importing...", flush=True)
import comfy.cli_args
comfy.cli_args.args.disable_xformers = True
comfy.cli_args.args.lowvram = True
# comfy.cli_args.args.fp16_unet=False
# comfy.cli_args.args.fp16_vae=False
# comfy.cli_args.args.fp16_text_enc=False
import comfy.model_management as mm
import comfy.sample
from fractions import Fraction
from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoComponents
from PIL import Image, ImageOps
import numpy as np
import comfy.model_sampling
import comfy.sd
import comfy.utils
import torch
import node_helpers
import gc
import torch
import nodes
import comfy_extras.nodes_model_advanced as nodes_adv
import comfy_extras.nodes_wan as nodes_wan
import importlib

torch.set_grad_enabled(False)

DEFAULT_NEG = """
low quality, blurry, flicker, low contrast, overexposed, underexposed, noise,  
distorted anatomy, stiff motion, unrealistic skin, painting, cartoon, CGI,  
3D render, extra limbs, watermark, text, logo, color banding, compression  
artifacts, unnatural camera shake
"""

class BaseModel:
    @torch.inference_mode()
    def __init__(self):
        unet_path = "/workspace/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp8_e4m3fn.safetensors"
        clip_path = "/workspace/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors"
        vae_path = "/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors"

        print("UNET...", flush=True)
        self.unet = comfy.sd.load_diffusion_model(unet_path, model_options={})

        # class ModelSamplingAdvanced(
        #     comfy.model_sampling.ModelSamplingDiscreteFlow,
        #     comfy.model_sampling.CONST):
        #     pass

        # self.unet.add_object_patch("model_sampling", ModelSamplingAdvanced(self.unet.model.model_config))

        print("Text Encoder...", flush=True)
        self.clip = comfy.sd.load_clip([clip_path], clip_type=comfy.sd.CLIPType.WAN)

        print("VAE...", flush=True)
        self.vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))


class Cache:
    @torch.inference_mode()
    def __init__(self, base_models):
        self.base_models = base_models
        self.cache = {}

    @torch.inference_mode()
    def with_loras(self, loras):
        # Loras
        print("Loras...", flush=True)
        tag = "loras"
        cached_key, cached_value = self.cache.get(tag, (None, None))
        key = ".".join(
            f"{strength_model}_{strength_clip}_{lora_path}" for lora_path, strength_model, strength_clip in loras
        )
        updated = cached_key != key
        if updated:
            print("... invalid cache", flush=True)
            unet = self.base_models.unet
            clip = self.base_models.clip
            for lora_path, strength_model, strength_clip in loras:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                unet, clip = comfy.sd.load_lora_for_models(unet, clip, lora, strength_model, strength_clip)
            self.cache[tag] = (key, (unet, clip))
        else:
            print("... reusing cache", flush=True)
            unet, clip = cached_value

        return updated, unet, clip

    @torch.inference_mode()
    def encode_text(self, clip, text, tag, force_update):
        key = text
        cached_key, cached_value = self.cache.get(tag, (None, None))
        updated = cached_key != key
        if updated or force_update:
            print("... invalid cache", flush=True)
            cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            self.cache[tag] = (key, cond)
        else:
            print("... reusing cache", flush=True)
            cond = cached_value
        return updated, cond


@torch.inference_mode()
def wan_latent(vae, positive, negative, width, height, length, batch_size, strength, control_video=None,
               control_masks=None, reference_image=None):
    latent_length = ((length - 1) // 4) + 1
    if control_video is not None:
        control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height,
                                                   "bilinear", "center").movedim(1, -1)
        if control_video.shape[0] < length:
            control_video = torch.nn.functional.pad(control_video,
                                                    (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]),
                                                    value=0.5)
    else:
        control_video = torch.ones((length, height, width, 3)) * 0.5

    if reference_image is not None:
        reference_image = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height,
                                                     "bilinear", "center").movedim(1, -1)
        # reference_image = torch.cat([
        #     vae.encode(ref_image.unsqueeze(0))
        #     for ref_image in torch.unbind(reference_image[:, :, :, :3], dim=0)
        # ], dim=2)
        reference_image = vae.encode(reference_image[:, :, :, :3])
        reference_image = torch.cat(
            [reference_image, comfy.latent_formats.Wan21().process_out(torch.zeros_like(reference_image))],
            dim=1)

    if control_masks is None:
        mask = torch.ones((length, height, width, 1))
    else:
        mask = control_masks
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        mask = comfy.utils.common_upscale(mask[:length], width, height, "bilinear", "center").movedim(1, -1)
        if mask.shape[0] < length:
            mask = torch.nn.functional.pad(mask, (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]), value=1.0)

    control_video = control_video - 0.5
    inactive = (control_video * (1 - mask)) + 0.5
    reactive = (control_video * mask) + 0.5

    print("Encoding latent...", flush=True)
    inactive = vae.encode(inactive[:, :, :, :3])
    reactive = vae.encode(reactive[:, :, :, :3])

    print("inactive", inactive.shape)
    print("reactive", reactive.shape)
    

    control_video_latent = torch.cat((inactive, reactive), dim=1)
    if reference_image is not None:
        control_video_latent = torch.cat((reference_image, control_video_latent), dim=2)

    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask = mask.permute(2, 4, 0, 1, 3)
    mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask),
                                           mode='nearest-exact').squeeze(0)

    trim_latent = 0
    if reference_image is not None:
        mask_pad = torch.zeros_like(mask[:, :reference_image.shape[2], :, :])
        mask = torch.cat((mask_pad, mask), dim=1)
        latent_length += reference_image.shape[2]
        trim_latent = reference_image.shape[2]

    mask = mask.unsqueeze(0)

    positive = node_helpers.conditioning_set_values(positive,
                                                    {"vace_frames": [control_video_latent], "vace_mask": [mask],
                                                     "vace_strength": [strength]}, append=True)
    negative = node_helpers.conditioning_set_values(negative,
                                                    {"vace_frames": [control_video_latent], "vace_mask": [mask],
                                                     "vace_strength": [strength]}, append=True)

    latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8],
                         device=comfy.model_management.intermediate_device())
    return positive, negative, latent, trim_latent


@torch.inference_mode()
def gen(base_models, cache, pos, neg, loras, width, height, length, seed, steps, shift, denoise,
         control_video=None, control_masks=None, reference_image=None):
    batch_size = 1
    strength = 1.0
    cfg = 1.0
    sampler = "uni_pc"
    scheduler = "simple"

    models_updated, unet, clip = cache.with_loras(loras)
    print("Encode prompts...", flush=True)
    mm.load_model_gpu(clip.patcher)
    pos_updated, pos_tokens = cache.encode_text(clip, text=pos, tag="pos", force_update=models_updated)
    neg_updated, neg_tokens = cache.encode_text(clip, text=neg, tag="neg", force_update=models_updated)
    positive, negative, latent, trim_latent = wan_latent(base_models.vae, pos_tokens, neg_tokens, width, height,
                                                             length, batch_size, strength, control_video=control_video,
                                                             control_masks=control_masks, reference_image=reference_image)

    noise = torch.randn(latent.size(), dtype=latent.dtype, layout=latent.layout,
                        generator=torch.manual_seed(seed), device="cpu")

    print("Sampling...", flush=True)
    # unet.get_model_object("model_sampling").set_parameters(shift=shift)
    m = unet.clone()

    sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
    sampling_type = comfy.model_sampling.CONST

    class ModelSamplingAdvanced(sampling_base, sampling_type):
        pass

    model_sampling = ModelSamplingAdvanced(unet.model.model_config)
    model_sampling.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", model_sampling)
    unet = m
    
    samples = comfy.sample.sample(
        model=unet,
        noise=noise,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=denoise)
    print(trim_latent, samples.shape)
    samples = samples[:, :, trim_latent:]

    print("Decoding latent...", flush=True)
    images = base_models.vae.decode(samples)
    images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

    return images, samples


@torch.inference_mode()
def save(images, out, metadata=None, fps=16):
    if len(images) > 1:
        video = VideoFromComponents(VideoComponents(images=images, frame_rate=Fraction(fps), metadata=metadata))
        p = out if out.endswith(".mp4") else out + ".mp4"
        video.save_to(p)
        print(f"Saved: {p}")
    else:
        img = Image.fromarray(np.clip(images[0].detach().numpy() * 255, 0, 255).astype(np.uint8))
        p = out if out.endswith(".png") else out + ".png"
        img.save(p)
        print(f"Saved: {p}")

def load_video(video_path):
    return VideoFromFile(video_path).get_components().images

def load_image(image_path):
    image = ImageOps.exif_transpose( Image.open(image_path))
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

def resize(img, width, height, upscale_method, crop):
    return comfy.utils.common_upscale(img.movedim(-1, 1), width, height, upscale_method, crop).movedim(1, -1)

def resize_and_pad(image, target_width, target_height, padding_color, interpolation):
    batch_size, orig_height, orig_width, channels = image.shape

    scale_w = target_width / orig_width
    scale_h = target_height / orig_height
    scale = min(scale_w, scale_h)

    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    image_permuted = image.permute(0, 3, 1, 2)

    resized = comfy.utils.common_upscale(image_permuted, new_width, new_height, interpolation, "disabled")

    pad_value = 0.0 if padding_color == "black" else 1.0
    padded = torch.full(
        (batch_size, channels, target_height, target_width),
        pad_value,
        dtype=image.dtype,
        device=image.device
    )

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2

    padded[:, :, y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

    return padded.permute(0, 2, 3, 1)
    
def mem_status():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")


def free_memory():
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()

def video_chunks(
    path,
    chunk_size=16,
    start_frame=0,
    frame_step=1,
    max_frames=None,
    device="cpu",
):
    """
    Yields chunks of video frames as torch tensors of shape (chunk_size, H, W, C).

    Args:
        path (str): Path to video file.
        chunk_size (int): Number of frames per yielded batch.
        start_frame (int): Index of first frame to start reading from.
        frame_step (int): Take 1 frame every `frame_step` frames.
        max_frames (int | None): Optional total frames to read.
        device (str): "cpu" or "cuda" for tensor device.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    # Seek to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    total_read = 0
    frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Take every j-th frame
        if (frame_idx - start_frame) % frame_step == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            total_read += 1

            # Yield full chunk
            if len(frames) == chunk_size:
                tensor = torch.from_numpy(np.stack(frames)).float().to(device)
                yield tensor, frame_idx
                frames = []

            if max_frames is not None and total_read >= max_frames:
                break

        frame_idx += 1

    # Yield leftover frames
    if frames:
        tensor = torch.from_numpy(np.stack(frames)).float().to(device)
        yield tensor, frame_idx

    cap.release()


class ComModels():
    def __init__(self):
        print("VAE...")
        self.vae, = nodes.VAELoader().load_vae(
            "wan_2.1_vae.safetensors")
        print("CLIP...")
        clip, = nodes.CLIPLoader().load_clip(
            "umt5_xxl_fp16.safetensors", type="wan")
        print("VACE UNet...")
        unet, = nodes.UNETLoader().load_unet(
            "wan2.1_vace_14B_fp8_e4m3fn.safetensors", "default")
        print("VACE CausVid...")
        self.base_unet = unet
        self.base_clip = clip
        self.vace_unet, self.vace_clip = nodes.LoraLoader().load_lora(unet, clip, 
            "wan21t2v/Wan21_CausVid_14B_T2V_lora_rank32.safetensors", 0.5, 1.0)
        print("...Done")

class ComCache():
    def __init__(self):
        self.cache = {}

    def get(self, tag, key, gen):
        ck, cv = self.cache.get(tag, (None, None))
        if ck == key:
            print("  cache hit")
        else:
            print("  cache miss")
            ck = key
            cv = gen()
            self.cache[tag] = (ck, cv)
        return cv

def com_vace(models, cache, seed, steps, cfg, sampler, scheduler, shift, pos_text, neg_text,
            width, height, length, control_video, control_masks, reference_image, strength, 
            loras, causvid=True):
    importlib.reload(nodes)
    importlib.reload(nodes_wan)

    unet, clip = (models.vace_unet, models.vace_clip) if causvid else (models.base_unet, models.base_clip)

    print("Loras")
    def load_loras():
        lora_unet = unet
        for strength, lpath in loras:
            lora_unet = nodes.LoraLoader().load_lora(lora_unet, None, lpath, strength, strength_clip=0.0)[0]
        return lora_unet
    loras_key = ",".join(f"{strength},{lpath}" for strength, lpath in loras)
    unet = cache.get("loras", (unet, loras_key), load_loras)

    print("Set shift")
    vace, = nodes_adv.ModelSamplingSD3().patch(unet, shift)

    print("Encoding pos")
    positive = cache.get("positive", (clip, pos_text), 
        lambda: nodes.CLIPTextEncode().encode(clip=clip, text=pos_text)[0])
    print("Encoding neg")
    negative = cache.get("negative", (clip, neg_text), 
        lambda: nodes.CLIPTextEncode().encode(clip=clip, text=neg_text)[0])

    print("Encoding latent")
    batch_size = 1
    denoise = 1.0
    pos_cond, neg_cond, input_latent, trim_latent = nodes_wan.WanVaceToVideo().execute(
        positive=positive, negative=negative, vae=models.vae, width=width, height=height, length=length, 
        batch_size=batch_size, strength=strength, 
        control_video=control_video, control_masks=control_masks, reference_image=reference_image)

    print("Sampling")
    samples, = nodes.KSampler().sample(model=vace, seed=seed, steps=steps, cfg=cfg, 
        sampler_name=sampler, scheduler=scheduler, positive=pos_cond, negative=neg_cond, 
        latent_image=input_latent, denoise=denoise)
    samples, = nodes_wan.TrimVideoLatent().execute(samples=samples, trim_amount=trim_latent)

    print("Decoding latent")
    out_images, = nodes.VAEDecode().decode(models.vae, samples)

    return out_images


def human(image, face=False, hair=False, glasses=False, top_clothes=False, bottom_clothes=False,
        torso_skin=False, left_arm=False, right_arm=False, left_leg=False, right_leg=False, left_foot=False, 
        right_foot=False, detail_method = "VITMatte",
        detail_erode = 8,
        detail_dilate = 6,
        black_point=0.01,
        white_point=0.99,
        process_detail=True,
        device="cuda",
        max_megapixels=2):
    from custom_nodes.comfyui_layerstyle.py.human_parts_ultra import LS_HumanPartsUltra
    hpu = LS_HumanPartsUltra()
    
    return hpu.human_parts_ultra(image, face, hair, glasses, top_clothes, bottom_clothes,
                            torso_skin, left_arm, right_arm, left_leg, right_leg, left_foot, right_foot,
                            detail_method, detail_erode, detail_dilate, black_point, white_point,
                            process_detail, device, max_megapixels)[0]

def draw_mask(image, color):
    return torch.lerp(torch.Tensor(color), image, image[:,:,:,3:])

print("... Done")

