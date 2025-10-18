import cv2
import torch
import numpy as np
from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2

# -------- CONFIG --------
input_video = "input.mp4"
output_video = "depth_output.mp4"
model_size = "small"  # choose from "small", "base", or "large"
device = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

# Load model
model = DepthAnythingV2.from_pretrained(model_size).to(device).eval()

# Load video
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Prepare output video writer
out = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# Process frames
for _ in tqdm(range(total_frames), desc="Processing frames"):
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_t = img_t.to(device)

    # Predict depth
    with torch.no_grad():
        depth = model(img_t)[0, 0].cpu().numpy()

    # Normalize and colorize depth
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Write frame
    out.write(depth_color)

cap.release()
out.release()

print(f"✅ Depth video saved as {output_video}")
