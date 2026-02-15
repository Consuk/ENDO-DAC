from __future__ import absolute_import, division, print_function

import os
import time
import argparse
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

import wandb
import matplotlib.cm as cm

from utils.layers import disp_to_depth
from utils.utils import readlines
from options import MonodepthOptions

import datasets
from datasets.hamlyn_dataset import HamlynDataset
import models.endodac as endodac
import models.encoders as encoders
import models.decoders as decoders


# -----------------------------
# Helpers: image conversions
# -----------------------------
_DEPTH_CMAP = cm.get_cmap("plasma")


def to_uint8_rgb_from_tensor(t_chw_01: torch.Tensor) -> np.ndarray:
    """Tensor [3,H,W] in [0,1] -> uint8 RGB [H,W,3]."""
    img = t_chw_01.detach().cpu().permute(1, 2, 0).numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return img


def colorize_map(x_hw: np.ndarray, robust=True) -> np.ndarray:
    """
    Colorize a single-channel map (depth or disp) -> uint8 RGB [H,W,3].
    Uses robust percentiles like your trainer.
    """
    x = x_hw.astype(np.float32)

    if robust:
        vmin = np.percentile(x[np.isfinite(x)], 5) if np.isfinite(x).any() else 0.0
        vmax = np.percentile(x[np.isfinite(x)], 95) if np.isfinite(x).any() else 1.0
    else:
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    x_norm = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    colored = _DEPTH_CMAP(x_norm)[..., :3]  # [H,W,3] in [0,1]
    return (colored * 255).astype(np.uint8)


def read_original_rgb_from_disk(data_path: str, folder: str, frame_idx: int, side: str) -> np.ndarray:
    """
    Hamlyn: images live in <data_path>/<folder>/image01 or image02 with {:010d}.jpg
    side: 'l' -> image01, 'r' -> image02
    """
    side_dir = "image01" if side == "l" else "image02"
    img_path = os.path.join(data_path, folder, side_dir, f"{frame_idx:010d}.jpg")
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def make_2x2_panel(tl: np.ndarray, bl: np.ndarray, tr: np.ndarray, br: np.ndarray, pad: int = 6) -> np.ndarray:
    """
    Combine 4 RGB uint8 images into a 2x2 panel:
      [ tl | tr ]
      [ bl | br ]
    Adds padding between cells.
    """
    def pad_img(img):
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    tlp, blp, trp, brp = map(pad_img, (tl, bl, tr, br))

    top = np.concatenate([tlp, trp], axis=1)
    bot = np.concatenate([blp, brp], axis=1)
    panel = np.concatenate([top, bot], axis=0)
    return panel


# -----------------------------
# Model loading (mirrors evaluate_depth.py behavior)
# -----------------------------
def load_depth_model(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    if not os.path.isdir(opt.load_weights_folder):
        raise FileNotFoundError(f"Cannot find folder: {opt.load_weights_folder}")

    print(f"-> Loading weights from {opt.load_weights_folder}")

    if opt.model_type == "endodac":
        # EndoDAC model wrapper
        model = endodac.EndoDAC(opt)
        model.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "depth_model.pth"), map_location="cpu"))
        model.eval()
        return model

    elif opt.model_type == "afsfm":
        # Optional: if you want to debug AF-SfMLearner style models, keep as reference
        encoder = encoders.ResnetEncoder(opt.num_layers, opt.weights_init == "pretrained")
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=opt.scales)
        encoder.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "encoder.pth"), map_location="cpu"))
        depth_decoder.load_state_dict(torch.load(os.path.join(opt.load_weights_folder, "depth.pth"), map_location="cpu"))

        def forward_fn(x):
            feats = encoder(x)
            out = depth_decoder(feats)
            return out

        # Wrap into a callable that returns dict with ("disp",0)
        class Wrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.decoder = depth_decoder

            def forward(self, x):
                feats = self.encoder(x)
                return self.decoder(feats)

        model = Wrapper()
        model.eval()
        return model

    else:
        raise ValueError(f"Unknown model_type: {opt.model_type}")


def build_hamlyn_loader(opt):
    # Use eval_filelist exactly like evaluate_depth.py
    if getattr(opt, "eval_filelist", None):
        fpath = os.path.expanduser(opt.eval_filelist)
    else:
        splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        fpath = os.path.join(splits_dir, opt.eval_split, "test_files.txt")

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Missing eval filelist: {fpath}")

    filenames = readlines(fpath)

    dataset = HamlynDataset(
        opt.data_path,
        filenames,
        opt.height,
        opt.width,
        frame_idxs=[0],
        num_scales=4,
        is_train=False,
        img_ext=".jpg",
    )

    loader = DataLoader(
        dataset,
        batch_size=getattr(opt, "eval_batch_size", 8),
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return filenames, dataset, loader, fpath


def load_gt_npz(opt):
    if not getattr(opt, "gt_depths_path", None):
        raise ValueError("Pass --gt_depths_path pointing to your .npz")
    gt_path = os.path.expanduser(opt.gt_depths_path)
    data_npz = np.load(gt_path, allow_pickle=True)
    gt_depths = data_npz["data"]
    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)
    return gt_depths, gt_path


# -----------------------------
# Main
# -----------------------------
def parse_args():
    # Reuse your MonodepthOptions so all the usual flags work
    options = MonodepthOptions()
    opt = options.parse()

    # Extra debug args
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--max_panels", type=int, default=80, help="How many samples to log total.")
    p.add_argument("--panel_every", type=int, default=1, help="Log every Nth sample.")
    p.add_argument("--wandb_project", type=str, default="debug-depth-panels")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    extra, _ = p.parse_known_args()

    # attach
    opt.max_panels = extra.max_panels
    opt.panel_every = extra.panel_every
    opt.wandb_project = extra.wandb_project
    opt.wandb_run_name = extra.wandb_run_name
    opt.wandb_entity = extra.wandb_entity
    opt.wandb_mode = extra.wandb_mode

    if not hasattr(opt, "model_type") or opt.model_type is None:
        opt.model_type = "endodac"

    return opt


def main():
    opt = parse_args()

    # W&B init
    if opt.wandb_mode != "disabled":
        wandb.init(
            project=opt.wandb_project,
            name=opt.wandb_run_name,
            entity=opt.wandb_entity,
            mode=opt.wandb_mode,
            config=vars(opt),
        )

    device = torch.device("cuda" if torch.cuda.is_available() and (not opt.no_cuda) else "cpu")

    model = load_depth_model(opt).to(device)
    filenames, dataset, loader, filelist_path = build_hamlyn_loader(opt)
    gt_depths, gt_path = load_gt_npz(opt)

    print(f"-> Using eval filelist: {filelist_path}")
    print(f"-> Using gt depths:    {gt_path}")
    print(f"-> num_split_lines: {len(filenames)} | num_gt: {len(gt_depths)}")

    # Safety check: counts must match for correct alignment
    if len(filenames) != len(gt_depths):
        print("WARNING: filenames and gt_depths counts differ. Panels will still log, but alignment is suspect.")

    logged = 0
    sample_idx_global = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            # batch inputs
            input_color = batch[("color", 0, 0)].to(device)  # [B,3,H,W]

            out = model(input_color)
            if not isinstance(out, dict) or ("disp", 0) not in out:
                raise RuntimeError("Model output does not contain ('disp', 0).")

            # Get disp/depth tensors (B,1,H,W)
            scaled_disp, pred_depth = disp_to_depth(out[("disp", 0)], opt.min_depth, opt.max_depth)

            B = input_color.shape[0]
            for j in range(B):
                i = sample_idx_global

                if (i % opt.panel_every) != 0:
                    sample_idx_global += 1
                    continue
                if logged >= opt.max_panels:
                    break

                # Parse filename line: "<folder> <frame> <side>"
                parts = filenames[i].strip().split()
                folder = parts[0]
                frame_idx = int(parts[1]) if len(parts) > 1 else 0
                side = parts[2] if len(parts) > 2 else "l"

                # Left/top: pred depth (colored)
                pred_depth_hw = pred_depth[j, 0].detach().cpu().numpy()
                pred_depth_vis = colorize_map(pred_depth_hw, robust=True)

                # Left/bottom: disp (colored) (use raw network disp for intuition)
                disp_hw = out[("disp", 0)][j, 0].detach().cpu().numpy()
                disp_vis = colorize_map(disp_hw, robust=True)

                # Right/top: ORIGINAL RGB from disk
                try:
                    rgb_orig = read_original_rgb_from_disk(opt.data_path, folder, frame_idx, side)
                except Exception as e:
                    # Fallback: resized tensor image (still useful)
                    rgb_orig = to_uint8_rgb_from_tensor(input_color[j])

                # Resize orig RGB to match model size for nice panel layout
                rgb_orig_resized = cv2.resize(rgb_orig, (opt.width, opt.height), interpolation=cv2.INTER_AREA)

                # Right/bottom: GT depth from NPZ (colored)
                if i < len(gt_depths):
                    gt = gt_depths[i].astype(np.float32)
                    # Resize GT to model size so it matches in the panel
                    gt_resized = cv2.resize(gt, (opt.width, opt.height), interpolation=cv2.INTER_NEAREST)
                    gt_vis = colorize_map(gt_resized, robust=True)
                    gt_stats = (float(np.min(gt)), float(np.max(gt)))
                else:
                    gt_vis = np.zeros((opt.height, opt.width, 3), dtype=np.uint8)
                    gt_stats = (float("nan"), float("nan"))

                panel = make_2x2_panel(
                    tl=pred_depth_vis,
                    bl=disp_vis,
                    tr=rgb_orig_resized,
                    br=gt_vis,
                    pad=6,
                )

                caption = (
                    f"{i}: {folder} {frame_idx} {side} | "
                    f"GT[min,max]={gt_stats[0]:.2f},{gt_stats[1]:.2f} | "
                    f"pred_depth[min,max]={float(np.min(pred_depth_hw)):.2f},{float(np.max(pred_depth_hw)):.2f}"
                )

                if opt.wandb_mode != "disabled":
                    wandb.log(
                        {
                            "debug/panel": wandb.Image(panel, caption=caption),
                            "debug/index": i,
                            "debug/folder": folder,
                            "debug/frame": frame_idx,
                            "debug/side": side,
                        },
                        step=i,
                    )

                print("logged:", caption)
                logged += 1
                sample_idx_global += 1

            if logged >= opt.max_panels:
                break

    dt = time.time() - t0
    print(f"Done. Logged {logged} panels in {dt:.1f}s")


if __name__ == "__main__":
    main()
