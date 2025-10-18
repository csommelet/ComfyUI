from comfy_api.latest import io
import logging
import torch
import numpy as np
import os
import color_match
from PIL import Image
import importlib
import color_match
importlib.reload(color_match)

from scipy.ndimage import distance_transform_edt

def shrink_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    dist = distance_transform_edt(mask > 0.5)
    grown = dist >= radius
    return grown

def hist_match(src, ref):
    for ch in range(3):
        src_vec = src[..., ch].ravel()
        ref_vec = ref[..., ch].ravel()

        _, src_idxs, src_cnts = np.unique(src_vec, return_inverse=True, return_counts=True)
        ref_vals, ref_cnts = np.unique(ref_vec, return_counts=True)

        src_cdf = np.cumsum(src_cnts).astype(np.float64) / src_vec.size
        ref_cdf = np.cumsum(ref_cnts).astype(np.float64) / ref_vec.size

        interp_vals = np.interp(src_cdf, ref_cdf, ref_vals)

    return res

# res[..., ch] = interp_vals[src_idxs].reshape(src[..., ch].shape)

def average_color_alpha_mask(img: Image.Image, mask: Image.Image):
    mask = np.asarray(mask)[...,3] > 128
    if not np.any(mask):
        return (0, 0, 0)
    arr = np.asarray(img, dtype=np.float32)
    return tuple(arr[mask].mean(axis=0) .astype(int))

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    # Move to CPU
    t = t[0].detach().cpu()

    # If float, squeeze to 0–255
    if t.dtype == torch.float32 or t.dtype == torch.float64:
        t = t.clamp(0, 1) * 255
        t = t.byte()

    logging.info(f"#KES# {t.shape}")
    return Image.fromarray(t.numpy())


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0   # H×W×C float32
    t = torch.from_numpy(arr)                       # H×W×C
    return t.unsqueeze(0)


def paste_resized_clipped(
    target,
    target_face_mask,
    target_matching_mask,
    new_face,
    new_matching_mask,
    rect: tuple[int, int, int, int],
    ref_img: Image.Image | None = None,
):
    ref_img = None
    source = tensor_to_pil(new_face)
    source_match = tensor_to_pil(new_matching_mask)

    """
    Paste `source` resized into rect on target.
    If ref_img is None, use target region under the paste area as reference,
    masked by the alpha of the source.
    """
    tgt_h, tgt_w = target.shape[1:3]
    print(tgt_h, tgt_w)
    left, top, right, bottom = rect

    rect_w = right - left
    rect_h = bottom - top
    if rect_w <= 0 or rect_h <= 0:
        return target

    # 1) Resize source
    src_resized = source.resize((rect_w, rect_h), Image.LANCZOS)
    source_match_rsz = source_match.resize((rect_w, rect_h), Image.LANCZOS)

    # 2) Visible region
    vis_left   = max(left, 0)
    vis_top    = max(top, 0)
    vis_right  = min(right, tgt_w)
    vis_bottom = min(bottom, tgt_h)

    if vis_left >= vis_right or vis_top >= vis_bottom:
        return target

    # 3) Crop visible part from source
    sx1 = vis_left - left
    sy1 = vis_top - top
    sx2 = sx1 + (vis_right - vis_left)
    sy2 = sy1 + (vis_bottom - vis_top)

    src_crop = src_resized.crop((sx1, sy1, sx2, sy2)).convert("RGBA")
    src_match_crop = source_match_rsz.crop((sx1, sy1, sx2, sy2))

    # 4) Build reference image
    # ---------------------------------------------
    src_np = np.asarray(src_crop, dtype=np.float32) / 255.0
    src_match_np = np.atleast_3d(np.asarray(src_match_crop, dtype=np.float32) / 255.0)

    target_img = tensor_to_pil(target)
    tgt_crop = target_img.crop((vis_left, vis_top, vis_right, vis_bottom)).convert("RGBA")
    ref_np = np.asarray(tgt_crop, dtype=np.float32) / 255.0

    target_match_img = tensor_to_pil(target_matching_mask)
    tgt_match_crop = target_match_img.crop((vis_left, vis_top, vis_right, vis_bottom))
    ref_match_np = np.atleast_3d(np.asarray(tgt_match_crop, dtype=np.float32) / 255.0)

    target_masked = target[0].detach().cpu().numpy()
    target_masked = np.concatenate([target_masked, 
            np.ones((*target_masked.shape[:2], 1), dtype=target_masked.dtype)], axis=-1)

    target_masked[vis_top:vis_bottom,vis_left:vis_right,:] *= (1.0 - src_np[..., -1:])
    target_masked[..., :] *= (1.0 - np.atleast_3d(target_face_mask[0].detach().cpu().numpy())[...,-1:])
    
    target_masked[..., :] *= shrink_mask(target_masked[..., -1:], 0.1 * max(rect_w, rect_h))
    
    target_masked = torch.tensor(target_masked).unsqueeze(0)
    
    

    src_np[:,:,:] *= src_match_np[:,:, -1:]
    ref_np[:,:,:] *= ref_match_np[:,:, -1:]
    
    # else:
    #     ref_img = ref_img.convert("RGBA")
    #     ref_np = np.asarray(ref_img, dtype=np.float32) / 255.0

    # tensor_to_pil(torch.tensor(src_np).unsqueeze(0)).save("src_np.png")
    # tensor_to_pil(torch.tensor(ref_np).unsqueeze(0)).save("ref_np.png")

    # matched_rgb = color_match.color_match(src_np, src_np, ref_np)
    params = color_match.build_params(src_np, ref_np)
    matched_rgb = color_match.apply_params(src_np, params)

    out_np = np.clip(matched_rgb * 255.0, 0, 255).astype(np.uint8)


    src_crop_matched_img = Image.fromarray(out_np, "RGBA")
    src_crop.paste(src_crop_matched_img, (0, 0), src_match_crop)

    target_img.paste(src_crop, (vis_left, vis_top), src_crop)

    new_color_matched = torch.zeros_like(target_masked)
    new_color_matched[0,vis_top:vis_bottom,vis_left:vis_right,:] = torch.tensor(np.asarray(src_crop, dtype=np.float32) / 255.0)
    

    target_pasted = torch.tensor(np.asarray(target_img, dtype=np.float32) / 255.0).unsqueeze(0)
    return new_color_matched, target_masked, target_pasted

class PasteImageNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSomPasteImagee",
            display_name="CSom Paste Image Node",
            category="",
            inputs=[
                io.Image.Input("image"),
                io.Image.Input("image_to_paste"),
                io.Custom("CSOM_PASTE_DATA").Input("paste_data")
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image, image_to_paste, paste_data) -> io.NodeOutput:
        out = image.clone()
        # c0 = torch.maximum(torch.tensor((0, 0)), torch.tensor(paste_data[0:2])).int()
        # c1 = torch.maximum(torch.tensor((image.shape[1], image.shape[0])), torch.tensor(paste_data[2:4])).int()
        # out[:, c0[1]:c1[1], c0[0]:c1[0],:] = 0
        i0 = tensor_to_pil(image.clone())
        i1 = tensor_to_pil(image_to_paste)
        logging.info(f"#KES# {image.shape} {image_to_paste.shape}")
        res = pil_to_tensor(paste_resized_clipped(
            i0,
            i1,
            [int(round(v)) for v in paste_data],
        ))
        return io.NodeOutput(
            res,
        )

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    @classmethod
    def fingerprint_inputs(*args, **kwargs):
        v = int(os.path.getmtime(__file__))
        logging.info(f"#KES3# fingerprint_inputs {v}")
        return v


class PasteImageNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CSomPasteImagee",
            display_name="CSom Paste Image Node",
            category="",
            inputs=[
                io.Image.Input("target"),
                io.Mask.Input("target_face_mask"),
                io.Mask.Input("target_matching_mask"),
                io.Image.Input("new_face"),
                io.Mask.Input("new_matching_mask"),
                io.Custom("CSOM_PASTE_DATA").Input("paste_data")
            ],
            outputs=[
                io.Image.Output(display_name="new_color_matched"),
                io.Image.Output(display_name="target_masked"),
                io.Image.Output(display_name="target_pasted"),
            ],
        )

    @classmethod
    def execute(cls, target, target_face_mask, target_matching_mask, 
        new_face, new_matching_mask, paste_data) -> io.NodeOutput:
        new_color_matched, target_masked, target_pasted = paste_resized_clipped(
            target,
            target_face_mask,
            target_matching_mask,
            new_face,
            new_matching_mask,
            [int(round(v)) for v in paste_data],
        )
        return io.NodeOutput(
            new_color_matched,
            target_masked,
            target_pasted,
        )

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    @classmethod
    def fingerprint_inputs(*args, **kwargs):
        v = int(os.path.getmtime(__file__))
        return f"{v}-{color_match.modiftime()}"
