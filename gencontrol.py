#!/usr/bin/env python3
"""
Create a 5s MP4 of a white disk moving periodically between two points
(10 full oscillations). Requires: opencv-python, numpy.

pip install opencv-python numpy
"""

import numpy as np
import cv2

def main():
    path = "/workspace/ComfyUI/input/1663177568_1-boombo-biz-p-huge-tits-handjobs-krasivaya-erotika-1.jpg"
    bg = cv2.imread(path)

    # -------- Parameters --------
    width, height = bg.shape[1], bg.shape[0]          # output resolution
    fps = 16                          # frames per second
    duration_s = 5.0                  # total duration (seconds)
    oscillations = 13                 # number of full cycles in the video
    radius = 14                       # disk radius (pixels)
    p0 = np.array([870, 360]) # start point (x, y)
    p1 = np.array([690, 300]) # end point (x, y)
    bg_color = (0, 0, 0)              # black background
    disk_color = np.ones((3,)) * 255.0
    out_path = "white_disk_oscillation.mp4"
    # ----------------------------

    pp = [
        np.array([[315, 100], [761, 150]]),
        np.array([[249, 138], [661, 130]]),
    ]
    pp2 = [
        np.array([[428, 580], [700, 530]]),
    ]

    # Derived values
    n_frames = int(round(duration_s * fps)) // 4 * 4 + 1
    f_hz = oscillations / duration_s       # cycles per second
    omega = 2 * np.pi * f_hz               # angular frequency
    omega2 = 0.5 * 2 * np.pi * f_hz               # angular frequency

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter. Check codec/folder permissions.")

    center = (p0 + p1) / 2.0
    half_vec = (p1 - p0) / 2.0
    w = np.asarray((-half_vec[1], half_vec[0])) * 0.3

    for i in range(n_frames):
        t = i / fps
        omega1 = omega * (1 + 0.2 * np.sin(t / n_frames))
        # Sinusoidal interpolation: position ranges exactly between p0 and p1
        pos = center - 0.9 * half_vec * np.cos(omega1 * t) + 0.3 * w * np.sin(omega1 * t)
        q = p0 + 2.0 * np.linalg.norm(half_vec) * (pos - p0) / np.linalg.norm(pos - p0)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # frame = bg.copy()
        if bg_color != (0, 0, 0):
            frame[:] = bg_color
        if i < 50:
            cv2.line(frame, (int(round(p0[0])), int(round(p0[1]))), (int(round(q[0])), int(round(q[1]))), (255, 128, 0), thickness=30)
            cv2.circle(frame, (int(round(pos[0])), int(round(pos[1]))), 50, disk_color, thickness=-1)
        j = 0
        for p, q in pp:
            u = i/(n_frames-1) * 5.5
            v = min(1.0, u) ** 0.9
            pq = p + v * (q - p)
            if u >= 1:
                pq[0] += np.sin((u - 1) * 6.0) * (20 + j * 5)
                pq[1] += np.sin((u - 1) * 8.0) * (j * 5)
            cv2.circle(frame, (int(round(pq[0])), int(round(pq[1]))), 20, disk_color, thickness=-1)
            j += 1
        
        # for qq in pp2:
        #     s = i/(n_frames-1)
        #     u = i/(n_frames-1) * 5.5
        #     v = min(1.0, u) ** 0.9
        #     pr = qq[0] + v * (qq[1] - qq[0])
        #     cv2.circle(frame, (int(round(pr[0])), int(round(pr[1]))), 10, (0, 0, 255), thickness=-1)


        # if t < 1.0:
        #     cv2.circle(frame, (int(250 + t * 140), int(100)), radius, disk_color, thickness=-1)
        # frame = cv2.blur(frame, (5, 5))
        writer.write(frame)

    writer.release()
    print(f"Wrote {out_path} ({width}x{height} @ {fps}fps, {duration_s}s, {oscillations} oscillations).")

if __name__ == "__main__":
    main()
