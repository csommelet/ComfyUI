import comfy.sd
import comfy.utils
import torch

torch.inference_mode()


# /workspace/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp8_e4m3fn.safetensors
# /workspace/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp16.safetensors
# /workspace/ComfyUI/models/loras/wan21t2v/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors
# /workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors
# /workspace/ComfyUI/models/vae/wan2_1_vae_fp8.safetensors
# /workspace/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors
unet_path = "/workspace/ComfyUI/models/diffusion_models/wan2.1_vace_14B_fp8_e4m3fn.safetensors"
lora_path = "/workspace/ComfyUI/models/loras/wan21t2v/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"
clip_path = "/workspace/ComfyUI/models/text_encoders/umt5_xxl_fp16.safetensors"
vae_path = "/workspace/ComfyUI/models/vae/wan_2.1_vae.safetensors"

model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
clip = comfy.sd.load_clip([clip_path],  clip_type=comfy.sd.CLIPType.WAN)
lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

pos_text = "rat riding a bicycle. Ultra realitic. HD 4K"
neg_text = "low_quality"
pos_tokens = clip_lora.encode_from_tokens_scheduled(clip_lora.tokenize(pos_text))
neg_tokens = clip_lora.encode_from_tokens_scheduled(clip_lora.tokenize(nrg_text))

vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

def wan_latent(positive, negative, vae, width, height, length, batch_size, strength):
    latent_length = ((length - 1) // 4) + 1
    control_video = torch.ones((length, height, width, 3)) * 0.5 - 0.5
    mask = torch.ones((length, height, width, 1))

    inactive = (control_video * (1 - mask)) + 0.5
    reactive = (control_video * mask) + 0.5

    inactive = vae.encode(inactive[:, :, :, :3])
    reactive = vae.encode(reactive[:, :, :, :3])

    control_video_latent = torch.cat((inactive, reactive), dim=1)

    vae_stride = 8
    height_mask = height // vae_stride
    width_mask = width // vae_stride
    mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
    mask = mask.permute(2, 4, 0, 1, 3)
    mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
    mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(latent_length, height_mask, width_mask), mode='nearest-exact').squeeze(0)

    mask = mask.unsqueeze(0)

    positive = node_helpers.conditioning_set_values(positive, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)
    negative = node_helpers.conditioning_set_values(negative, {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]}, append=True)

    latent = torch.zeros([batch_size, 16, latent_length, height // 8, width // 8], device=comfy.model_management.intermediate_device())
    out_latent = {"samples": latent}

    return positive, negative, out_latent
