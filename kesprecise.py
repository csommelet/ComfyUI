import kes
import importlib
importlib.reload(kes)

import numpy as np
from typing import List
import cv2
from controlnet_aux.open_pose import util, draw_poses, PoseResult
from controlnet_aux.open_pose.body import Keypoint, BodyResult
import json
import time
import pathlib
import random
import torch
import os.path
import importlib.util, sys, os

class DWPose:
    def __init__(self):
        from custom_nodes.comfyui_controlnet_aux.node_wrappers.dwpose import DWPose_Preprocessor
        self.dwp = DWPose_Preprocessor()

    def run(self, image, detect_hand=True, detect_body=True, detect_face=True, resolution=512):
        res = self.dwp.estimate_pose(image,
                detect_hand=("enable" if detect_hand else "disable"), 
                detect_body=("enable" if detect_body else "disable"), 
                detect_face=("disable" if detect_face else "disable"), 
                resolution=(min(image.shape[1:3]) if resolution == 0 else resolution))
        return res["result"]


class Depth:
    def __init__(self):
        from custom_nodes.comfyui_controlnet_aux.node_wrappers.depth_anything_v2 import Depth_Anything_V2_Preprocessor
        self.dat = Depth_Anything_V2_Preprocessor()

    def run(self, image, resolution=512):
        res = self.dat.execute(image, 
                resolution=(min(image.shape[1:3]) if resolution == 0 else resolution))
        return res[0]

def md5sum(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

class Source():
    def __init__(self, path, tag):
        self.path = path
        self.images = kes.load_image(path) if os.path.splitext(path)[1].lower() in (".png", ".jpg", ".jpeg") else kes.load_video(path)
        self.hash = md5sum(path)[:6]
        self.prefix = f"/workspace/out-precise/{tag}-{self.hash}"


def load_or_gen(path, gen, reset=False):
    if not reset and os.path.exists(path):
        return kes.load_video(path) if path.endswith(".mp4") else kes.load_image(path)
    vid = gen()
    kes.save(vid, path)
    return vid

def with_ext(images, path):
    return path + ".png" if len(images) == 1 else path + ".mp4"

def pose(images, prefix):
    pose_path = with_ext(images, prefix + "-dwpose")
    kps_path = prefix + "-kps.txt"
    if os.path.exists(pose_path) and os.path.exists(kps_path):
        pose = kes.load_video(pose_path)
        with open(kps_path, "r") as fin:
            kps = json.load(fin)
    else:
        pose, kps = DWPose().run(images, 
            detect_hand=True, 
            detect_body=True,
            detect_face=False, 
            resolution=0)

        kes.save(pose, pose_path)
        with open(kps_path, "w") as fout:
            json.dump(kps, fout, indent=4)
    return pose, kps

def importanylib(alias, pkg_dir):
    mod = sys.modules.get(alias, None)
    if mod is None:
        init_file = os.path.join(pkg_dir, "__init__.py")
        spec = importlib.util.spec_from_file_location(
            alias, init_file, submodule_search_locations=[pkg_dir]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[alias] = mod
        spec.loader.exec_module(mod)
    return mod

def rembg(images):
    was_node_suite_comfyui = importanylib("was_node_suite_comfyui", "/workspace/ComfyUI/custom_nodes/was-node-suite-comfyui")
    rembg = was_node_suite_comfyui.NODE_CLASS_MAPPINGS["Image Rembg (Remove Background)"]()
    return rembg.image_rembg(
        images,
        transparency=True,
        model="u2net",
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_size=10,
        post_processing=False,
        only_mask=False,
        background_color="none",
    )[0]

def depth(images, prefix):
    return load_or_gen(with_ext(images, prefix + "-depth"), lambda: Depth().run(images, resolution=0))

def ctrl(path, start, length):
    vid = load_or_gen(path)

def kps_guide(source_kps, target_kps, length, prefix):
    js = [source_kps, target_kps]

    ks = [np.asarray(j["people"][0]["pose_keypoints_2d"]).reshape(-1, 3) for j in js]
    ss = [(j["canvas_height"], j["canvas_width"]) for j in js]

    h, w = ss[1]

    def bb(a):
        a = a[a[:, 2] > 0, :2]
        xymin = a.min(axis=0)
        xymax = a.max(axis=0)
        return xymin, xymax
    sel = [0,1,2,5,14,15,16,17]
    bbs = [bb(k[sel,:]) for k in ks]
    print("######")
    print(bbs)

    cts = [((bbs[i][1] + bbs[i][0])) * 0.5 for i in range(2)]

    for i in range(2):
        ks[i][~np.isin(np.arange(18), sel), 2] = 0
        md = np.max(bbs[i][1] - bbs[i][0]) * (1.0 + i * 0.4)
        
        sc = 1 #np.max(bbs[0][1] - bbs[0][0]) / md
        tr = 0 #cts[0] - cts[i] * sc

        ks[i][sel,:2] = (ks[i][sel,:2] * sc + tr) / (w, h)

    # h = h // 2 // 16 * 16
    # w = w // 2 // 16 * 16
    # h, w = 720, 720
    print(h, w)
    # print([k.astype(int) for k in ks])
    # ks[0][~np.isin(np.arange(18), [0,14,15,16,17]), 2] = 0
    # ks[1][5:,2] = 0
    frames = []
    for i, s in enumerate(np.linspace(0, 1, length)):
        t = np.clip(s * 1, 0.0, 1.0)
        body_xyc = (ks[0] * (1-t) + ks[1] * t)
        if t != 0 and t != 1:
            body_xyc[:,2] *= ks[0][:,2] * ks[1][:,2]
        # print(body_xyc.astype(int))

        keypoints: List[Keypoint] = []
        for idx, (x, y, s) in enumerate(body_xyc):
            if s <= 0.0:
                keypoints.append(None)
            else:
                keypoints.append(Keypoint(x=float(x), y=float(y), score=float(s), id=idx))
        body_result = BodyResult(
            keypoints=keypoints,
            total_score=float(body_xyc[:, 2].sum()),
            total_parts=int((body_xyc[:, 2] > 0).sum()),
        )

        pose = PoseResult(
            body=body_result,
            left_hand=None,
            right_hand=None,
            face=None,
        )

        # This uses util.draw_bodypose / draw_handpose / draw_facepose internally
        canvas = draw_poses([pose], h, w, draw_body=True, draw_hand=False, draw_face=False)
        frames.append(canvas)
    guide = torch.cat([torch.from_numpy(f).unsqueeze(0) / 255 for f in frames], dim=0)
    kes.save(guide, prefix + "-moving-pose")
    return guide


def move_head(head, guide, dim, prefix, models, cache):
    ts = int(time.time())

    pos_text = """
    close-up shot on woman's face.
    she stops smiling, and show are think long tongue.
    camera starts at eye-level view, then slowly moves upward into an ultra-high  
    angle top-down view, smooth cinematic transition, stable motion, continuous  
    focus on subject, natural perspective shift
    """
    pos_text=""
    loras = [
            (1.0, "moni/adapter_model_2048_e30.safetensors"),
            # (1.0, "wan21t2v/wan_female_masturbation.safetensors"),
            # (0.8, "wan21t2v/wan_dr34mj0b_t2v_HD.safetensors"),
            # (0.4, "wan21t2v/ultimateblowjob.safetensors"),
            # (0.8, "wan21t2v/wan_t2v_pov_blowjob_v1.2.safetensors"),
            # (0.8, "wan21t2v/Pov_Blowjob_wan_v1.0.safetensors"),
            # (0.9+0*random.uniform(0.8, 1.2), "wan21t2v/BetterTitfuck_v4_July2025.safetensors"),
        ]

    cfg = 1.0 #int(random.uniform(1.0, 2.0) * 100) / 100
    steps = 5
    shift = 2
    height, width = guide.shape[1:3]

    height, width = dim, dim * width // height
    height, width = (height + 8) // 16 * 16, (width + 8) // 16 * 16

    # start_masked = kes.load_image("/workspace/out-precise/start-masked.png")

    # start_mask = mask = start_masked[:,:,:,3:].repeat(1,1,1,3)
    # start_masked[:,:,:,:3] = start_masked[:,:,:,:3] * (1.0 - start_mask) + start_mask

    strength = 1.2
    extra_frames = 8
    control_masks = None
    control_video = torch.cat([
        head.repeat(1 + extra_frames, 1, 1, 1),
        guide[1:],
        # start_masked.repeat(1 + extra_frames, 1, 1, 1),
    ], dim=0)
    length = control_video.shape[0]

    kes.save(control_video, prefix + "-control")

    # path = random.choice(list(pathlib.Path("input/facelles").rglob("*.[pj][pn]g"))[:1])
    ref_img = None #[ herp ]
    out_images = kes.com_vace(
            seed=ts, steps=steps, shift=shift,
            width=width, height=height, length=length,
            cfg=cfg, sampler="unipc", scheduler="simple",
            control_video = control_video,
            control_masks = control_masks,
            reference_image = ref_img,
            loras=loras,
            strength=strength,
            pos_text = pos_text,
            neg_text = kes.DEFAULT_NEG,
            models=models, cache=cache, causvid=True
        )[extra_frames:]

    kes.save(out_images, prefix + "-moving-head", fps=16)

    kes.save(torch.lerp(out_images, 
        kes.resize(guide, width, height, "lanczos", "center"), 0.5),
        prefix + "-moving-head-view", fps=16)

    return out_images[-1:]


def swap(ref, placed_head_first, placed_head_last, width, height, pose, depth, prefix, models, cache):
    ts = int(time.time())
    seed = ts
    pos_text = """
    fully naked woman, kneeling between the legs of a man,
    she is masturbating and sucking man's erect penis,
    she has hyper glossy transparent lipstick.
    in the middle of dense rainy forest, under heavy rain
    
    """ + """
    ultra detailed, cinematic lighting, 8k, RAW photo, ultra sharp focus, 
    skin pores visible, realistic skin texture, detailed eyes and lips, 
    perfect face proportions, subsurface scattering, high dynamic range, 
    ray tracing reflections, depth of field, global illumination, 
    finely detailed hair strands, film grain, tone mapped, physically-based rendering
    """ + """
    studio lighting, soft directional light, rim light, key light and fill light, 
    three-point lighting setup, gentle shadows, perfect exposure, 
    volumetric light, subtle bounce light, cinematic contrast, 
    HDR balanced tones, warm key light and cool fill light, 
    specular highlights on skin, light diffusion through skin, 
    reflective catchlights in eyes, precise shading and contouring
    """
    loras = [
            # (1.0, "moni/adapter_model_2048_e30.safetensors"),
            # (1.0, "wan21t2v/wan_female_masturbation.safetensors"),
            (0.5, "wan21t2v/wan_dr34mj0b_t2v_HD.safetensors"),
            # (0.4, "wan21t2v/ultimateblowjob.safetensors"),
            # (0.8, "wan21t2v/wan_t2v_pov_blowjob_v1.2.safetensors"),
            # (0.8, "wan21t2v/Pov_Blowjob_wan_v1.0.safetensors"),
            # (0.9+0*random.uniform(0.8, 1.2), "wan21t2v/BetterTitfuck_v4_July2025.safetensors"),
        ]
    # self.refq[start:start+length], self.refd[start:start+length]
    print(pose.shape, depth.shape)
    length = pose.shape[0]
    cfg = 1.0 #int(random.uniform(1.0, 2.0) * 100) / 100
    steps = 5
    shift = 2
    if height == 0:
        height, width = pose.shape[1:3]
    height, width = (height + 8) // 16 * 16, (width + 8) // 16 * 16
    strength = 1.2
    control_video = kes.resize(torch.lerp(pose, depth, 0.01), width, height, "bilinear", "center")
    control_masks = None
    # img = kes.load_image("/workspace/out5/DSC07410.JPG")
    # img = comfy.utils.common_upscale(img.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)[:,:,:,:3]
    # control_video = torch.ones((length, height, width, 3)) #, dtype=vid.dtype, device=vid.device)
    # control_masks = torch.ones((length, height, width)) #, dtype=vid.dtype, device=vid.device)

    if placed_head_first is not None:
        control_video[:placed_head_first.shape[0]] = kes.resize_and_pad(placed_head_first, width, height, "white", "lanczos")[:,:,:,:3]
    if placed_head_last is not None:
        control_video[-placed_head_last.shape[0]:] = kes.resize_and_pad(placed_head_last, width, height, "white", "lanczos")[:,:,:,:3]
    
    # control_masks[0] = 0

    ref_img = [
        kes.resize_and_pad(ref, width, height, "white", "lanczos"),
    # ] * 4 + [
        kes.resize_and_pad(kes.load_image("input/mydk2.png"), width, height, "white", "lanczos"),
    ]
    # ref_img = None
    out_images = kes.com_vace(
            seed=seed, steps=steps, shift=shift,
            width=width, height=height, length=length,
            cfg=cfg, sampler="unipc", scheduler="simple",
            control_video = control_video,
            control_masks = control_masks,
            reference_image = ref_img,
            loras=loras,
            strength=strength,
            pos_text = pos_text,
            neg_text = kes.DEFAULT_NEG,
            models=models, cache=cache, causvid=True
        )
    kes.save(out_images, prefix + "-swap", fps=16)
    return out_images

def extract_face(head, prefix):
    human_parts = kes.human(head, face=True, hair=True, glasses=True, torso_skin=False, white_point=1.0)
    human_masked = kes.draw_mask(human_parts, torch.Tensor([1, 1, 1, 1]))
    kes.save(human_masked[:,:,:,:3], prefix + "-head-hpu")
    kes.save(human_parts[:,:,:,3:].repeat(1,1,1,3), prefix + "-head-hpu-mask")
    return human_parts, human_masked

def her_head_naked(face, prefix, models, cache):
    seed = int(time.time())
    print(f"her_head_naked : {seed}")
    pos_text = """
    hd 8k portrait close-up fully naked woman, pure white background,
    hyper glossy transparent lipstick.

    subject perfectly still, no motion blur, frozen in time, absolute stillness, 
    steady camera, sharp edges, zero movement, static composition
    """ + """
    ultra detailed, cinematic lighting, 8k, RAW photo, ultra sharp focus, 
    skin pores visible, realistic skin texture, detailed eyes and lips, 
    perfect face proportions, subsurface scattering, high dynamic range, 
    ray tracing reflections, depth of field, global illumination, 
    finely detailed hair strands, film grain, tone mapped, physically-based rendering
    """ + """
    studio lighting, soft directional light, rim light, key light and fill light, 
    three-point lighting setup, gentle shadows, perfect exposure, 
    volumetric light, subtle bounce light, cinematic contrast, 
    HDR balanced tones, warm key light and cool fill light, 
    specular highlights on skin, light diffusion through skin, 
    reflective catchlights in eyes, precise shading and contouring
    """

    neg_text = """
    low quality, blurry, flicker, low contrast, overexposed, underexposed, noise,  
    distorted anatomy, stiff motion, unrealistic skin, painting, cartoon, CGI,  
    3D render, extra limbs, watermark, text, logo, color banding, compression  
    artifacts, unnatural camera shake, camera movement, model movement
    """

    loras = [
            # (1.0, "moni/adapter_model_2048_e30.safetensors"),
            # (1.0, "wan21t2v/wan_female_masturbation.safetensors"),
            # (0.8, "wan21t2v/wan_dr34mj0b_t2v_HD.safetensors"),
            # (0.4, "wan21t2v/ultimateblowjob.safetensors"),
            # (0.8, "wan21t2v/wan_t2v_pov_blowjob_v1.2.safetensors"),
            # (0.8, "wan21t2v/Pov_Blowjob_wan_v1.0.safetensors"),
            # (0.9+0*random.uniform(0.8, 1.2), "wan21t2v/BetterTitfuck_v4_July2025.safetensors"),
        ]
    length = 17
    cfg = 1.0 #int(random.uniform(1.0, 2.0) * 100) / 100
    steps = 5
    shift = 2
    height, width = face.shape[1:3]

    strength = 1.0
    control_video = torch.ones((length, height, width, 3)) #, dtype=vid.dtype, device=vid.device)
    control_masks = torch.ones((length, height, width)) #, dtype=vid.dtype, device=vid.device)
    control_video[0] = face[0, :, :, :3]
    control_masks[0] = face[0, :, :, 3]
    ref_image = None
    head_naked = kes.com_vace(
            seed=seed, steps=steps, shift=shift,
            width=width, height=height, length=length,
            cfg=cfg, sampler="unipc", scheduler="simple",
            control_video = control_video,
            control_masks = control_masks,
            reference_image = ref_image,
            loras=loras,
            strength=strength,
            pos_text = pos_text,
            neg_text = neg_text,
            models=models, cache=cache, causvid=True
        )
    kes.save(head_naked, prefix+"-head-naked", fps=16)
    return head_naked[-1:]
            

def swap_frame(first, last, length, prefix, models, cache):
    ts = int(time.time())
    seed = ts
    pos_text = """
woman kneeling between the legs of a man,
she is masturbating and sucking man's erect penis, always looking at the camera
"""
    loras = [
            # (1.0, "moni/adapter_model_2048_e30.safetensors"),
            # (1.0, "wan21t2v/wan_female_masturbation.safetensors"),
            (0.8, "wan21t2v/wan_dr34mj0b_t2v_HD.safetensors"),
            # (0.4, "wan21t2v/ultimateblowjob.safetensors"),
            # (0.8, "wan21t2v/wan_t2v_pov_blowjob_v1.2.safetensors"),
            # (0.8, "wan21t2v/Pov_Blowjob_wan_v1.0.safetensors"),
            # (0.9+0*random.uniform(0.8, 1.2), "wan21t2v/BetterTitfuck_v4_July2025.safetensors"),
        ]
    cfg = 1.0 #int(random.uniform(1.0, 2.0) * 100) / 100
    steps = 5
    shift = 2
    height, width = 720, 1024 # first.shape[1:3]
    height, width = (height + 8) // 16 * 16, (width + 8) // 16 * 16
    strength = 1.0

    first = kes.resize(first, width, height, "bilinear", "center")
    last = kes.resize(last, width, height, "bilinear", "center").repeat(17, 1, 1, 1)
    length += 17
    control_video = torch.ones((length, height, width, 3)) #, dtype=vid.dtype, device=vid.device)
    control_masks = torch.ones((length, height, width)) #, dtype=vid.dtype, device=vid.device)
    control_video[:1] = first[:,:,:,:3]
    control_masks[:1] = 1.0 - first[:,:,:,3]
    control_video[-17:] = last[:,:,:,:3]
    control_masks[-17:] = 1.0 - last[:,:,:,3]
    ref_img = [first]
    out_images = kes.com_vace(
            seed=seed, steps=steps, shift=shift,
            width=width, height=height, length=length,
            cfg=cfg, sampler="unipc", scheduler="simple",
            control_video = control_video,
            control_masks = control_masks,
            reference_image = ref_img,
            loras=loras,
            strength=strength,
            pos_text = pos_text,
            neg_text = kes.DEFAULT_NEG,
            models=models, cache=cache, causvid=True
        )
    kes.save(out_images, prefix + "-swap-frame", fps=16)
    return out_images[-1:]


def warp_image_by_eye_alignment(
    target_img,
    target_eyes,
    source_img,
    source_eyes,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REPLICATE,
    border_value=0,
):
    """
    Warp `source_img` so that its eye coordinates map to those of `target_img`,
    returning an image of the same size as `target_img`.
    Uses direct vector relationships without angle or normalization.
    """
    # Extract coordinates
    target_left, target_right = np.asarray(target_eyes, dtype=np.float64)
    source_left, source_right = np.asarray(source_eyes, dtype=np.float64)

    # Centers between eyes
    target_center = (target_left + target_right) / 2.0
    source_center = (source_left + source_right) / 2.0

    # Eye direction vectors
    vt = target_right - target_left
    vs = source_right - source_left

    # Safety check
    if np.allclose(vs, 0) or np.allclose(vt, 0):
        raise ValueError("Eye points are degenerate; cannot compute transform.")

    # Compute 2D rotation + scale matrix directly
    dot = np.dot(vs, vt)
    cross = vs[0] * vt[1] - vs[1] * vt[0]

    denom = np.dot(vs, vs)
    if denom < 1e-12:
        raise ValueError("Source eye vector too small for reliable transform.")

    # Derive rotation-scale matrix R so that vs maps to vt
    a = (dot / denom)
    b = (cross / denom)
    R = np.array([[a, -b],
                  [b,  a]], dtype=np.float64)

    # Translation so that source_center maps to target_center
    t = target_center - R @ source_center

    # Final affine transform
    M = np.hstack([R, t.reshape(2, 1)])

    h, w = target_img.shape[:2]
    warped_img = cv2.warpAffine(
        source_img,
        M,
        (w, h),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )

    return torch.Tensor(warped_img).unsqueeze(0)

def pipe_init(her_path, ref_path):
    #### HER #####
    print("###### HER...")
    her = Source(her_path, "her")

    her_pose, her_kps = pose(her.images, her.prefix)
    her_face_alpha, her_face = extract_face(her.images, her.prefix)

    #### REF ####
    print("###### REF...")
    ref = Source(ref_path, "ref")
    print("###### REF.POSE...")
    ref_pose, ref_kps = pose(ref.images, ref.prefix)
    print("###### REF.DEPTH...")
    ref_depth = depth(ref.images, ref.prefix)

    return her, her_pose, her_kps, her_face_alpha, her_face, ref, ref_pose, ref_kps, ref_depth

def pipe_step1(pctx, width, height, first_frame, length, models, cache):
    her, her_pose, her_kps, her_face_alpha, her_face, ref, ref_pose, ref_kps, ref_depth = pctx

    #### JOIN #####
    print("###### JOIN.ALIGN ...")
    join_prefix = ref.prefix + f"-{her.hash}"
    head_pts = np.asarray(her_kps[0]["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)

    # 0: Nose
    # 1: Neck
    # 2: Shoulder R
    # 5: Shoulder L
    # 14: Eye R
    # 15: Eye L
    # 16: Ear R
    # 17: Ear L
    # print(pts[[0,1,2,5,14,15,16,17]])
    placed_heads = []
    for fls in (first_frame, first_frame+length-1):
        ref_pts = np.asarray(ref_kps[fls]["people"][0]["pose_keypoints_2d"]).reshape(-1, 3)
        placed_head = warp_image_by_eye_alignment(ref.images[fls].numpy(), ref_pts[14:16,:2], 
            her_face[0].numpy(), head_pts[14:16,:2],
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            border_value=[1,1,1,1])
        placed_heads.append(placed_head)
        kes.save(placed_head, join_prefix + f"-f{fls}-warp.png")
        kes.save(0.5 * (placed_head[:,:,:,:3] + ref.images[fls:fls+1][:,:,:,:3]), join_prefix + f"-f{fls}-warp-view.png")

    print("###### JOIN.SWAP ...")
    ids = range(first_frame, first_frame + length)
    swp = swap(her_face, placed_heads[0], None, width, height, ref_pose[ids], ref_depth[ids], ref.prefix + f"-{her.hash}", models, cache)
    vi = 0.5 * (kes.resize(swp, ref.images.shape[2], ref.images.shape[1], "lanczos", "center") + ref.images[ids])
    kes.save(vi, ref.prefix + "-out")