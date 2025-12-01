import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import os

pth_path = "/root/HVTC/checkpoints/cross_stitch_pure_yyx/latest_net_Cross_Stitch.pth"
save_dir = "cs_line_plots"
os.makedirs(save_dir, exist_ok=True)

ckpt = torch.load(pth_path, map_location="cpu")
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

layers = defaultdict(lambda: defaultdict(dict))

# 收集参数
for k, v in state_dict.items():
    if "cross" in k.lower() or "stitch" in k.lower():
        parts = k.split(".")
        layer = parts[0]
        task_i = parts[-3]
        task_j = parts[-2]
        layers[layer][task_i][task_j] = v.detach().cpu()

# 绘制
for layer, data in layers.items():
    plt.figure(figsize=(10,5))
    plt.title(f"Cross-stitch per-channel summed weights - {layer}")
    plt.xlabel("Channel")
    plt.ylabel("Weight Value")

    # 每通道求 self + cross
    tran_curve = (data["tran"]["tran"] + data["tran"]["seg"]).numpy()
    seg_curve  = (data["seg"]["seg"]  + data["seg"]["tran"]).numpy()

    plt.plot(tran_curve, label="tran_output", linewidth=1.5)
    plt.plot(seg_curve, label="seg_output", linewidth=1.5)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{layer}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✓ Saved: {save_path}")

print("🎉 Done! Two smooth curves per layer saved.")
