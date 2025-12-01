import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict


def moving_average(arr, window=7):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='same')


pth_path = "/root/HVTC/checkpoints/cross_stitch_pure_yyx/latest_net_Cross_Stitch.pth"
save_dir = "cs_vis_4lines_smooth"
os.makedirs(save_dir, exist_ok=True)

ckpt = torch.load(pth_path, map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

layers = defaultdict(lambda: defaultdict(dict))

# Collect cross-stitch params
for k, v in state_dict.items():
    if "cross" in k.lower() or "stitch" in k.lower():
        parts = k.split(".")
        layer = parts[0]
        task_i = parts[-3]
        task_j = parts[-2]
        layers[layer][task_i][task_j] = v.detach().cpu().numpy()


window = 7  # smoothing window

for layer, data in layers.items():

    tran_tran = moving_average(data["tran"]["tran"], window)
    tran_seg  = moving_average(data["tran"]["seg"], window)
    seg_seg   = moving_average(data["seg"]["seg"], window)
    seg_tran  = moving_average(data["seg"]["tran"], window)

    # ---- 图 1：tran 输出 ----
    plt.figure(figsize=(10,4))
    plt.title(f"{layer} - tran output")
    plt.plot(tran_tran, label="tran←tran", linewidth=1.5)
    plt.plot(tran_seg, label="tran←seg", linewidth=1.5)
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{layer}_tran.png"))
    plt.close()

    # ---- 图 2：seg 输出 ----
    plt.figure(figsize=(10,4))
    plt.title(f"{layer} - seg output")
    plt.plot(seg_seg, label="seg←seg", linewidth=1.5)
    plt.plot(seg_tran, label="seg←tran", linewidth=1.5)
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{layer}_seg.png"))
    plt.close()

    print(f"✓ Saved: {layer}_tran.png, {layer}_seg.png")

print("🎉 Done! 4 curves (smoothed) saved per layer.")
