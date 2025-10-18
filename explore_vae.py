import os
import numpy as np
import safetensors
import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from PIL import Image
import lpips  # pip install lpips

import comfy.sd
import comfy.utils
from comfy.ldm.wan.vae import WanVAE

# ============================================================
# 🔧 CONFIGURATION
# ============================================================

# --- Paths ---
VAE_PATH = "models/vae/wan_2.1_vae.safetensors"
LATENT_PATH = "/workspace/war/ComfyUITrim_00010_.latent"
REF_IMG_DIR = "/workspace/war/AnimateDiff3_00005"         # folder with img_001.png ... img_009.png
OUT_LATENT_PATH = "/workspace/war/optimized_latent.pt"
OUT_RECON_DIR = "/workspace/war/rr2"       # folder for saving reconstructions

# --- Optimization parameters ---
NUM_REFERENCE_FRAMES = 9          # how many frames have reference images
LR = 1e-3                         # learning rate
NUM_STEPS = 5000                   # optimization steps
LAMBDA_PERCEPTUAL = 1.0           # perceptual term weight
LAMBDA_SMOOTH = 0.1               # temporal smoothness regularization
DTYPE = torch.float16             # enforce FP32 (safe for MPS/CUDA)

os.makedirs(OUT_RECON_DIR, exist_ok=True)

# ============================================================
# ⚙️ DEVICE SETUP
# ============================================================
if torch.backends.cuda.is_built() and torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"🧠 Using device: {device}")

# ============================================================
# 🧩 LOAD MODEL
# ============================================================
vae_base = comfy.sd.VAE(
    comfy.utils.load_torch_file(VAE_PATH),
    device=device,
    dtype=DTYPE
)
vae = vae_base.first_stage_model
assert isinstance(vae, WanVAE)
vae.requires_grad_(False)
vae.to(device, dtype=DTYPE)
print(f"✅ Loaded VAE ({next(vae.parameters()).dtype}, {next(vae.parameters()).device})")

# ============================================================
# 🖼️ LOAD REFERENCE IMAGES
# ============================================================
to_tensor = transforms.Compose([transforms.ToTensor()])
ref_imgs = []

for i in range(NUM_REFERENCE_FRAMES):
    img_path = f"{REF_IMG_DIR}/img_{i+1:03d}.png"
    img = Image.open(img_path).convert("RGB")
    ref = to_tensor(img).unsqueeze(0).to(device, dtype=DTYPE)
    # Normalize to [-1, 1] for VAE decoding
    ref = ref * 2.0 - 1.0
    ref_imgs.append(ref)

ref_imgs = torch.stack(ref_imgs, dim=2)  # [1,3,T,H,W]
print(f"✅ Loaded {NUM_REFERENCE_FRAMES} reference frames:", ref_imgs.shape)

# ============================================================
# 💾 LOAD LATENT SEQUENCE
# ============================================================
latent_data = safetensors.torch.load_file(LATENT_PATH)
z_init = latent_data["latent_tensor"].to(device, dtype=DTYPE)
print("✅ Loaded latent:", z_init.shape)

lpips_fn = lpips.LPIPS(net='vgg').to(device)
lpips_fn.requires_grad_(False)

def the_loss(pred, ref):
    # LPIPS perceptual similarity + pixel MSE
    with torch.inference_mode():
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, 3, pred.shape[3], pred.shape[4])
        ref_2d  = ref.permute(0, 2, 1, 3, 4).reshape(-1, 3, ref.shape[3], ref.shape[4])
        perceptual_loss = lpips_fn(pred_2d, ref_2d).mean()
    pixel_loss = F.mse_loss(pred, ref)
    return LAMBDA_PERCEPTUAL * perceptual_loss + pixel_loss
    
@torch.inference_mode()
def optim2():
    z_opt = z_init.clone().detach().requires_grad_(False)
    sigma = 0.01
    num_dirs = 4
    lr = 0.01

    for step in range(NUM_STEPS):
        base = vae.decode(z_opt)
        loss_base = the_loss(base[:, :, :9], ref_imgs[:, :, :9])
        grads = torch.zeros_like(z_opt)

        for _ in range(num_dirs):
            u = torch.randn_like(z_opt)
            loss_pos = the_loss(vae.decode(z_opt + sigma * u)[:, :, :9], ref_imgs[:, :, :9])
            grads += (loss_pos - loss_base) * u / sigma

        grads /= num_dirs
        z_opt = z_opt - lr * grads  # manual update

        print(f"Step {step:04d} | total={loss_base.item():.6f} ")

        # Save decoded preview frames (first 9)
        recon_vis = base[0, :, :NUM_REFERENCE_FRAMES, :, :].detach().cpu().movedim(1, 0)
        for i, frame in enumerate(recon_vis):
            frame_np = ((frame.movedim(0, -1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            Image.fromarray(frame_np).save(os.path.join(OUT_RECON_DIR, f"img_{i+1:03d}.png"))

optim2()

def optim1():
    z_opt = z_init.clone().detach().requires_grad_(True)
    # ============================================================
    # 🎯 PREPARE OPTIMIZER & LOSSES
    # ============================================================
    optimizer = optim.Adam([z_opt], lr=LR)

    # ============================================================
    # 🔁 OPTIMIZATION LOOP
    # ============================================================
    for step in range(NUM_STEPS):
        optimizer.zero_grad()

        # Decode the video
        recon = vae.decode(z_opt)  # [1,3,T,H,W]

        # --- Reconstruction (Perceptual + Pixel) loss for first 9 frames ---
        loss_recon = 0.0
        for t in range(NUM_REFERENCE_FRAMES):
            ref = ref_imgs[:, :, t, :, :]
            pred = recon[:, :, t, :, :]
            loss_recon += the_loss(pred, ref)

        loss_recon /= NUM_REFERENCE_FRAMES

        # --- Temporal smoothness loss ---
        loss_smooth = ((z_opt[:, :, 1:, :, :] - z_opt[:, :, :-1, :, :]) ** 2).mean()

        # --- Total loss ---
        loss = loss_recon + LAMBDA_SMOOTH * loss_smooth

        # --- Backprop + step ---
        loss.backward()
        optimizer.step()

        # --- Logging & image dump ---
        if step % 10 == 0:
            print(f"Step {step:04d} | total={loss.item():.6f} "
                f"recon={loss_recon.item():.6f} smooth={loss_smooth.item():.6f}")

            # Save decoded preview frames (first 9)
            recon_vis = recon[0, :, :NUM_REFERENCE_FRAMES, :, :].detach().cpu().movedim(1, 0)
            for i, frame in enumerate(recon_vis):
                frame_np = ((frame.movedim(0, -1).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                Image.fromarray(frame_np).save(os.path.join(OUT_RECON_DIR, f"step{step:04d}_img_{i+1:03d}.png"))

# ============================================================
# 💾 SAVE OPTIMIZED LATENT
# ============================================================
torch.save(z_opt.detach().cpu(), OUT_LATENT_PATH)
print(f"✅ Optimization complete. Saved to {OUT_LATENT_PATH}")
