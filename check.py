import torch
import matplotlib.pyplot as plt
import os

# ============================
# 1. 加载 pth
# ============================
ckpt = torch.load(
    "/root/HVTC/checkpoints/cross_stitch_yyx/latest_net_Cross_Stitch.pth",
    map_location="cpu"
)

if isinstance(ckpt, dict):
    state_dict = ckpt
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    raise ValueError("找不到 state_dict，请检查你的 pth 格式")


# ============================
# 2. 提取 cross-stitch 参数并按层组织
# ============================
layers = {}  # {layer_name: {param_type: np.array}}

for k, v in state_dict.items():
    if "cross" in k.lower() or "stitch" in k.lower() or "cs" in k.lower():
        
        parts = k.split(".")
        layer_name = parts[0]  # 例如 conv1
        param_type = ".".join(parts[-3:])  # 例如 seg.seg.param

        if layer_name not in layers:
            layers[layer_name] = {}
        layers[layer_name][param_type] = v.detach().cpu().numpy()


# ============================
# 3. 画图，每层一张，4 条线
# ============================
save_dir = "./cs_line_plots"
os.makedirs(save_dir, exist_ok=True)

param_order = ["tran.tran.param", "tran.seg.param",
               "seg.tran.param", "seg.seg.param"]

titles = {
    "tran.tran.param": "TT (trans→trans)",
    "tran.seg.param":  "TS (seg→trans)",
    "seg.tran.param":  "ST (trans→seg)",
    "seg.seg.param":   "SS (seg→seg)"
}

for layer, param_dict in layers.items():

    plt.figure(figsize=(12, 6))

    for p in param_order:
        if p in param_dict:
            values = param_dict[p]  # shape = (channels,)
            plt.plot(range(1, len(values)+1), values, label=titles[p], linewidth=2)

    plt.title(f"Cross-Stitch Weights - {layer}", fontsize=18)
    plt.xlabel("Channel Index", fontsize=14)
    plt.ylabel("Weight", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)

    save_path = os.path.join(save_dir, f"{layer}_cs.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved → {save_path}")
