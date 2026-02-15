from __future__ import absolute_import, division, print_function

import os
import time
import numpy as np
import cv2
from tqdm import tqdm

import torch
import wandb
import matplotlib.cm as cm

from options import MonodepthOptions
from utils.layers import disp_to_depth
from utils.utils import readlines

# Reuse your exact evaluate_depth wiring (model + loader + gt loader)
import evaluate_depth as evalmod


# -----------------------------
# Config (no new CLI args needed)
# -----------------------------
DEFAULT_MAX_PANELS = int(os.environ.get("DEBUG_MAX_PANELS", "120"))
DEFAULT_PANEL_EVERY = int(os.environ.get("DEBUG_PANEL_EVERY", "1"))

_DEPTH_CMAP = cm.get_cmap("plasma")


def colorize_map(x_hw: np.ndarray, robust=True) -> np.ndarray:
    x = x_hw.astype(np.float32)
    mask = np.isfinite(x)
    if robust and mask.any():
        vmin = np.percentile(x[mask], 5)
        vmax = np.percentile(x[mask], 95)
    else:
        vmin = np.nanmin(x) if mask.any() else 0.0
        vmax = np.nanmax(x) if mask.any() else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    x_norm = np.clip((x - vmin) / (vmax - vmin), 0.0, 1.0)
    colored = _DEPTH_CMAP(x_norm)[..., :3]
    return (colored * 255).astype(np.uint8)


def pil_to_rgb_uint8(pil_img) -> np.ndarray:
    # PIL -> RGB np.uint8
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr.astype(np.uint8)


def make_2x2_panel(tl, bl, tr, br, pad=6):
    def pad_img(img):
        return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    tlp, blp, trp, brp = map(pad_img, (tl, bl, tr, br))
    top = np.concatenate([tlp, trp], axis=1)
    bot = np.concatenate([blp, brp], axis=1)
    return np.concatenate([top, bot], axis=0)


@torch.no_grad()
def main():
    options = MonodepthOptions()
    opt = options.parse()

    if not hasattr(opt, "model_type") or opt.model_type is None:
        opt.model_type = "endodac"

    # --- W&B init (use existing flags in your repo: --use_wandb, --wandb_project, --wandb_run_name)
    use_wandb = getattr(opt, "use_wandb", False)
    if use_wandb:
        wandb.init(
            project=getattr(opt, "wandb_project", "endodac-debug"),
            name=getattr(opt, "wandb_run_name", None),
            config=vars(opt),
        )

    device = torch.device("cuda" if (torch.cuda.is_available() and (not opt.no_cuda)) else "cpu")

    # --- Exactly like evaluate_depth.py
    depther = evalmod.load_model(opt)
    dataset, dataloader, filenames, eval_filelist_path = evalmod.build_dataset_and_loader(opt)
    gt_depths, gt_depths_path = evalmod.load_gt_depths_npz(
        opt.eval_split, getattr(opt, "gt_depths_path", None)
    )

    print(f"-> Using eval filelist: {eval_filelist_path}")
    print(f"-> Using gt depths:    {gt_depths_path}")
    print(f"-> num_split_lines: {len(filenames)} | num_gt: {len(gt_depths)}")

    max_panels = DEFAULT_MAX_PANELS
    panel_every = DEFAULT_PANEL_EVERY

    logged = 0
    global_idx = 0
    t0 = time.time()

    for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        input_color = data[("color", 0, 0)].to(device)

        output = depther(input_color)

        if not isinstance(output, dict) or ("disp", 0) not in output:
            raise RuntimeError("Model output does not contain ('disp', 0).")

        # scaled_disp is what evaluate_depth uses (then 1/disp => depth)
        scaled_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
        scaled_disp = scaled_disp.detach().cpu()[:, 0].numpy()  # (B,H,W)

        raw_disp = output[("disp", 0)].detach().cpu()[:, 0].numpy()  # (B,H,W)

        B = input_color.shape[0]
        for j in range(B):
            i = global_idx

            # stop condition
            if i % panel_every != 0:
                global_idx += 1
                continue
            if logged >= max_panels:
                break

            # Split line format: "<folder> <frame_index> <side>"
            parts = filenames[i].strip().split()
            folder = parts[0]
            frame_idx = int(parts[1]) if len(parts) > 1 else 0
            side = parts[2] if len(parts) > 2 else "l"

            # ---- left/top: predicted depth = 1 / scaled_disp
            pred_disp = scaled_disp[j]
            pred_depth = 1.0 / np.maximum(pred_disp, 1e-6)
            pred_depth_vis = colorize_map(pred_depth, robust=True)

            # ---- left/bottom: predicted disp (raw network output)
            disp_vis = colorize_map(raw_disp[j], robust=True)

            # ---- right/top: ORIGINAL image via dataset.get_color (not the resized tensor)
            try:
                pil_img = dataset.get_color(folder, frame_idx, side, do_flip=False)
                rgb_orig = pil_to_rgb_uint8(pil_img)
            except Exception as e:
                # fallback: use the resized tensor if something goes wrong
                rgb_orig = (input_color[j].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            rgb_orig_resized = cv2.resize(rgb_orig, (opt.width, opt.height), interpolation=cv2.INTER_AREA)

            # ---- right/bottom: GT depth from NPZ (resize to model size)
            gt = gt_depths[i].astype(np.float32)
            gt_resized = cv2.resize(gt, (opt.width, opt.height), interpolation=cv2.INTER_NEAREST)
            gt_vis = colorize_map(gt_resized, robust=True)

            panel = make_2x2_panel(
                tl=pred_depth_vis,
                bl=disp_vis,
                tr=rgb_orig_resized,
                br=gt_vis,
                pad=6,
            )

            caption = (
                f"{i}: {folder} {frame_idx} {side} | "
                f"GT[min,max]={float(np.min(gt)):.2f},{float(np.max(gt)):.2f} | "
                f"pred_depth[min,max]={float(np.min(pred_depth)):.2f},{float(np.max(pred_depth)):.2f}"
            )

            print("logged:", caption)

            if use_wandb:
                wandb.log({"debug/panel": wandb.Image(panel, caption=caption)}, step=i)

            logged += 1
            global_idx += 1

        if logged >= max_panels:
            break

    dt = time.time() - t0
    print(f"Done. Logged {logged} panels in {dt:.1f}s")


if __name__ == "__main__":
    main()
