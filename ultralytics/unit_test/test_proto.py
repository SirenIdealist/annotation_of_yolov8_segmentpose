import os
import math
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.utils as vutils

from ultralytics.nn.modules.block import Proto


# ========== 构造模拟多尺度特征 ==========
def make_fake_pyramid(batch=2, base_size=640, c3=128, device="cpu"):
    """
    构造伪多尺度特征 (stride 8/16/32)，模拟 YOLO FPN/Neck 输出。
    返回 dict: {8: feat_s8, 16: feat_s16, 32: feat_s32}
    """
    h8 = base_size // 8
    h16 = base_size // 16
    h32 = base_size // 32
    feats = {
        8: torch.randn(batch, c3, h8, h8, device=device),
        16: torch.randn(batch, c3 * 2, h16, h16, device=device),
        32: torch.randn(batch, c3 * 4, h32, h32, device=device)
    }
    return feats


# ========== 原型归一化与可视化 ==========
def _normalize_protos(protos: torch.Tensor, mode="per_channel", clip_percent=0.0, gamma=None):
    """
    protos: [P,H,W]
    mode: per_channel | global | percentile
    clip_percent: 百分位裁剪百分比（仅 percentile 模式有效，如 1.0 表示裁掉上下 1%）
    gamma: 伽马校正 (<1 增强暗部)
    """
    p = protos
    if mode == "global":
        mn = p.min()
        mx = p.max()
        x = (p - mn) / (mx - mn + 1e-6)
    elif mode == "percentile":
        lp = clip_percent / 100.0
        low_q = torch.quantile(p, lp)
        high_q = torch.quantile(p, 1 - lp)
        x = (p - low_q) / (high_q - low_q + 1e-6)
        x = x.clamp(0, 1)
    else:  # per_channel
        mn = p.amin(dim=(1, 2), keepdim=True)
        mx = p.amax(dim=(1, 2), keepdim=True)
        span = (mx - mn)
        x = (p - mn) / (span + 1e-6)
        low_var = span < 1e-4
        x[low_var.squeeze()] = 0.0
    if gamma is not None:
        x = x.clamp(0, 1) ** gamma
    return x.clamp(0, 1)


def visualize_prototypes(proto_tensor: torch.Tensor,
                         out_dir="runs/debug_proto",
                         prefix="proto",
                         norm_mode="per_channel",
                         clip_percent=0.0,
                         gamma=None,
                         use_colormap=True,
                         colormap="viridis",
                         max_channels=None):
    """
    proto_tensor: [B,P,H,W]
    norm_mode: per_channel | global | percentile
    max_channels: 仅显示前 N 个通道（调试时可减少）
    """
    os.makedirs(out_dir, exist_ok=True)
    protos = proto_tensor[0].detach().cpu()  # [P,H,W]

    if max_channels is not None:
        protos = protos[:max_channels]

    # 归一化
    nm = "percentile" if norm_mode == "percentile" else norm_mode
    protos_norm = _normalize_protos(protos, mode=nm, clip_percent=clip_percent, gamma=gamma)

    if use_colormap:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        tiles = []
        for p in protos_norm:
            arr = p.numpy()
            colored = (cmap(arr)[:, :, :3] * 255).astype("uint8")  # H,W,3
            tiles.append(torch.from_numpy(colored).permute(2, 0, 1))  # 3,H,W
        stack = torch.stack(tiles)  # [P,3,H,W]
        grid = vutils.make_grid(stack, nrow=int(math.ceil(len(stack) ** 0.5)))
        img = grid.permute(1, 2, 0).numpy().astype("uint8")
    else:
        grid = vutils.make_grid(protos_norm.unsqueeze(1),
                                nrow=int(math.ceil(protos_norm.shape[0] ** 0.5)))
        img = (grid.permute(1, 2, 0).numpy() * 255).astype("uint8")

    out_path = Path(out_dir) / f"{prefix}_grid.png"
    from PIL import Image
    Image.fromarray(img).save(out_path)
    return out_path


# ========== 单项测试：前向、梯度、可视化 ==========
def test_proto_forward_and_shapes(device="cpu",
                                  base=640,
                                  c_in=128,
                                  c_hidden=256,
                                  num_protos=32,
                                  norm_mode="per_channel",
                                  clip_percent=0.0,
                                  gamma=None,
                                  cmap="viridis",
                                  use_colormap=True):
    """
    1. 构造 stride=8 伪特征
    2. 前向：输出形状检查
    3. 反向：梯度检查
    4. 可视化：原型网格
    """
    batch = 2
    feats = make_fake_pyramid(batch=batch, base_size=base, c3=c_in, device=device)
    x = feats[8]  # [B,c_in, base/8, base/8]

    proto_module = Proto(c1=c_in, c_=c_hidden, c2=num_protos).to(device)
    proto_module.train()

    t0 = time.time()
    out = proto_module(x)  # 期望: [B, num_protos, (base/8)*2, (base/8)*2]
    dt = (time.time() - t0) * 1000

    expect_h = (base // 8) * 2
    assert out.shape == (batch, num_protos, expect_h, expect_h), \
        f"输出形状不符: {out.shape} vs {(batch, num_protos, expect_h, expect_h)}"
    print(f"[Proto] forward OK shape={tuple(out.shape)} time={dt:.2f} ms")

    # 构造假损失做反向
    fake_target = torch.randn_like(out)
    loss = (out - fake_target).pow(2).mean()
    loss.backward()

    for name, p in proto_module.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"参数 {name} 无梯度"
    print(f"[Proto] backward OK loss={loss.item():.4f}")

    # 可视化
    grid_path = visualize_prototypes(out.detach(),
                                     out_dir="runs/debug_proto",
                                     prefix=f"proto_{num_protos}",
                                     norm_mode=norm_mode,
                                     clip_percent=clip_percent,
                                     gamma=gamma,
                                     use_colormap=use_colormap,
                                     colormap=cmap)
    print(f"[Proto] prototype grid saved to: {grid_path}")


# ========== 不同输入尺寸测试 ==========
def test_proto_different_input(device="cpu"):
    batch = 1
    base = 512
    c_in = 96
    feats = make_fake_pyramid(batch=batch, base_size=base, c3=c_in, device=device)
    x = feats[8]  # [1,96,64,64]

    proto_module = Proto(c1=c_in, c_=192, c2=24).to(device)
    y = proto_module(x)
    assert y.shape == (batch, 24, 128, 128)
    print(f"[Proto@{base}] OK shape={tuple(y.shape)}")


# ========== 命令行入口 ==========
def parse_args():
    ap = argparse.ArgumentParser("Proto 单元/可视化测试")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base", type=int, default=640, help="模拟输入原图尺寸（正方形）")
    ap.add_argument("--cin", type=int, default=128, help="输入通道（stride=8特征）")
    ap.add_argument("--chidden", type=int, default=256, help="Proto 中间通道 c_")
    ap.add_argument("--protos", type=int, default=32, help="原型个数 c2")
    ap.add_argument("--norm", default="per_channel", choices=["per_channel", "global", "percentile"])
    ap.add_argument("--clip", type=float, default=0.0, help="percentile 模式裁剪百分比（如 1.0）")
    ap.add_argument("--gamma", type=float, default=None, help="伽马校正，例如 0.8")
    ap.add_argument("--cmap", default="viridis", help="伪彩色 colormap 名")
    ap.add_argument("--no-color", action="store_true", help="使用灰度而不是伪彩色")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device
    torch.manual_seed(0)

    test_proto_forward_and_shapes(device=device,
                                  base=args.base,
                                  c_in=args.cin,
                                  c_hidden=args.chidden,
                                  num_protos=args.protos,
                                  norm_mode=args.norm,
                                  clip_percent=args.clip,
                                  gamma=args.gamma,
                                  cmap=args.cmap,
                                  use_colormap=not args.no_color)

    test_proto_different_input(device=device)