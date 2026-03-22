from __future__ import absolute_import, division, print_function

import os
import cv2
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import scipy.stats as st

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from options import MonodepthOptions

import datasets
from datasets.hamlyn_dataset import HamlynDataset  # trainer-style explicit import
try:
    from datasets.c3vd_dataset import (
        C3VDDataset,
        DEFAULT_C3VD_DEPTH_SCALE,
        build_c3vd_default_filelists,
    )
except Exception:
    C3VDDataset = None
    DEFAULT_C3VD_DEPTH_SCALE = 100.0 / 65535.0
    build_c3vd_default_filelists = None

from datasets.scared_dataset import SCAREDRAWDataset

import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac


cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


def load_gt_depths_npz(eval_split: str, gt_depths_path: str = None):
    """
    Default: splits/<eval_split>/gt_depths.npz
    Override: --gt_depths_path /path/to/custom_gt_depths.npz
    """
    if gt_depths_path is not None:
        gt_path = os.path.expanduser(gt_depths_path)
    else:
        gt_path = os.path.join(splits_dir, eval_split, "gt_depths.npz")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Missing GT file: {gt_path}\n"
            f"Generate it with export_gt_depth.py for split='{eval_split}', "
            f"or pass --gt_depths_path to point to your custom .npz."
        )

    data_npz = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)
    gt_depths = data_npz["data"]
    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)
    gt_depths = [np.asarray(g, dtype=np.float32) for g in gt_depths]

    return gt_depths, gt_path


def build_dataset_and_loader(opt):
    """
    Trainer-style dataset wiring:
      - dataset selection by opt.dataset (fallback to opt.eval_split)
      - filenames from splits/<eval_split>/test_files.txt
      - img_ext: .jpg for hamlyn, else .png (matches trainer behavior)
      - frame_ids used for evaluation: [0]
    """
    # Choose dataset key similar to trainer:
    dataset_key = getattr(opt, "dataset", None) or opt.eval_split
    if dataset_key == "endovis" and opt.eval_split != "endovis":
        dataset_key = opt.eval_split

    # Keep explicit mapping; do not silently fall back to a different dataset.
    datasets_dict = {
        "endovis": SCAREDRAWDataset,
        "hamlyn": HamlynDataset,
    }
    if C3VDDataset is not None:
        datasets_dict["c3vd"] = C3VDDataset

    if dataset_key not in datasets_dict:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Expected one of: {sorted(datasets_dict.keys())}. "
            f"Set --dataset correctly (e.g., --dataset hamlyn)."
        )

    dataset_cls = datasets_dict[dataset_key]

    # Load test filenames (default: splits/<eval_split>/test_files.txt, override: --eval_filelist)
    if getattr(opt, "eval_filelist", None):
        fpath = os.path.expanduser(opt.eval_filelist)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing split file: {fpath}")
        filenames = readlines(fpath)
    else:
        fpath = os.path.join(splits_dir, opt.eval_split, "test_files.txt")
        if os.path.exists(fpath):
            filenames = readlines(fpath)
        elif dataset_key == "c3vd" and build_c3vd_default_filelists is not None:
            auto_lists = build_c3vd_default_filelists(
                opt.data_path,
                write_to_splits_dir=os.path.join(splits_dir, opt.eval_split),
            )
            filenames = auto_lists["test"]
            fpath = f"<auto:{opt.eval_split}/test_files.txt>"
        else:
            raise FileNotFoundError(
                f"Missing split file: {fpath}\n"
                f"Either create splits/{opt.eval_split}/test_files.txt or pass --eval_filelist."
            )

    if len(filenames) == 0:
        raise RuntimeError(
            f"No evaluation samples found for split '{opt.eval_split}' using filelist '{fpath}'."
        )

    # img_ext matches trainer's hamlyn override
    img_ext = ".jpg" if opt.eval_split == "hamlyn" else ".png"

    # For evaluation we only need frame 0 (single image per sample)
    frame_ids = [0]
    num_scales = 4

    dataset_kwargs = {"img_ext": img_ext}
    if dataset_key == "c3vd":
        dataset_kwargs.update(
            {
                "use_intrinsics_file": (not getattr(opt, "learn_intrinsics", True))
                and getattr(opt, "c3vd_use_intrinsics_file", True),
                "intrinsics_path": getattr(opt, "c3vd_intrinsics_path", None),
                "depth_scale": getattr(opt, "c3vd_depth_scale", DEFAULT_C3VD_DEPTH_SCALE),
            }
        )

    dataset = dataset_cls(
        opt.data_path,
        filenames,
        opt.height,
        opt.width,
        frame_ids,
        num_scales,
        is_train=False,
        **dataset_kwargs,
    )

    # Use a batch size if available; default to 16 for speed
    batch_size = getattr(opt, "eval_batch_size", 16)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return dataset, loader, filenames, fpath


def load_model(opt):
    """
    Same model-loading behavior as your existing evaluate_depth.py:
      - If model_type == endodac: load depth_model.pth
      - If model_type == afsfm: load encoder.pth + depth.pth
    """
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    if not os.path.isdir(opt.load_weights_folder):
        raise FileNotFoundError(f"Cannot find folder: {opt.load_weights_folder}")

    print(f"-> Loading weights from {opt.load_weights_folder}")

    if opt.model_type == "endodac":
        depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
        depther_dict = torch.load(depther_path, map_location="cpu")

        depther = endodac.endodac(
            backbone_size="base",
            r=opt.lora_rank,
            lora_type=opt.lora_type,
            image_shape=(224, 280),
            pretrained_path=opt.pretrained_path,
            residual_block_indexes=opt.residual_block_indexes,
            include_cls_token=opt.include_cls_token,
        )
        model_dict = depther.state_dict()
        depther.load_state_dict(
            {k: v for k, v in depther_dict.items() if k in model_dict},
            strict=False,
        )
        depther.cuda().eval()
        return depther

    if opt.model_type == "afsfm":
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path, map_location="cpu")

        encoder = encoders.ResnetEncoder(opt.num_layers, False)
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict(
            {k: v for k, v in encoder_dict.items() if k in model_dict},
            strict=False,
        )
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        encoder.cuda().eval()
        depth_decoder.cuda().eval()

        def depther(image):
            return depth_decoder(encoder(image))

        return depther

    raise ValueError("You must set --model_type endodac or --model_type afsfm")


def evaluate(opt):
    MIN_DEPTH = opt.min_depth
    MAX_DEPTH = opt.max_depth

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Choose mono or stereo with --eval_mono or --eval_stereo"

    # ----------------------------
    # Load predictions or model
    # ----------------------------
    if opt.ext_disp_to_eval is None:
        depther = load_model(opt)
        pred_disps = None
    else:
        print(f"-> Loading predictions from {opt.ext_disp_to_eval}")
        pred_disps = np.load(opt.ext_disp_to_eval)
        depther = None

    # ----------------------------
    # Trainer-style dataset creation
    # ----------------------------
    dataset, dataloader, filenames, eval_filelist_path = build_dataset_and_loader(opt)

    # Load GT depths (allow override)
    gt_depths, gt_depths_path = load_gt_depths_npz(opt.eval_split, getattr(opt, "gt_depths_path", None))

    print(f"-> Using eval filelist: {eval_filelist_path}")
    print(f"-> Using gt depths:    {gt_depths_path}")

    # ----------------------------
    # Predict
    # ----------------------------
    inference_times = []
    if pred_disps is None:
        print(f"-> Computing predictions with size {opt.width}x{opt.height}")
        pred_disps_list = []

        printed_intrinsics_debug = False

        with torch.no_grad():
            for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                # ---- DEBUG: verify Hamlyn intrinsics are actually loaded from intrinsics.txt ----
                if (not printed_intrinsics_debug) and ("intrinsics_from_file" in data):
                    try:
                        intr_flag = data["intrinsics_from_file"]
                        uniq = torch.unique(intr_flag).detach().cpu().tolist()
                        print(f"[DEBUG] Intrinsics loaded from file? unique flags in batch: {uniq} (1=file, 0=fallback)")
                        if int(torch.max(intr_flag).item()) == 1:
                            if (("K", 0) in data):
                                print(f"[DEBUG] Sample K (first item):\n{data[('K', 0)][0].detach().cpu().numpy()}")
                        else:
                            print("[WARNING] Fallback intrinsics in use — check intrinsics.txt path / hamlyn_use_intrinsics_file flags!")
                    except Exception as e:
                        print(f"[DEBUG] Could not print intrinsics debug info: {e}")
                    printed_intrinsics_debug = True

                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                t0 = time.time()
                output = depther(input_color)
                inference_times.append(time.time() - t0)

                if not isinstance(output, dict) or ("disp", 0) not in output:
                    raise RuntimeError("Model output does not contain ('disp', 0).")

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()  # (B,H,W)

                pred_disps_list.append(pred_disp)

        pred_disps = np.concatenate(pred_disps_list, axis=0)

    # ----------------------------
    # Sanity check ordering
    # ----------------------------
    print(f"-> num_pred: {pred_disps.shape[0]} | num_gt: {len(gt_depths)} | num_split_lines: {len(filenames)}")
    if pred_disps.shape[0] != len(gt_depths):
        raise AssertionError(
            f"Mismatch: {pred_disps.shape[0]} predictions vs {len(gt_depths)} gt depth maps.\n"
            f"Check that the GT .npz was generated from the SAME filelist used here:\n"
            f"  filelist: {eval_filelist_path}\n"
            f"  gt_npz:   {gt_depths_path}"
        )

    # ----------------------------
    # Scaling mode
    # ----------------------------
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling")
        opt.disable_median_scaling = True
    else:
        print("   Mono evaluation - using median scaling")

    # ----------------------------
    # Evaluate metrics
    # ----------------------------
    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = np.asarray(gt_depths[i], dtype=np.float32)

        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-6)

        gt_depth[gt_depth >= 65535 - 1e-3] = 0.0

        mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_valid = gt_depth[mask]

        if not opt.disable_median_scaling:
            ratio = np.median(gt_valid) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH



        gt_valid = np.asarray(gt_valid, dtype=np.float32)
        pred_depth = np.asarray(pred_depth, dtype=np.float32)

        err = compute_errors(gt_valid, pred_depth)

        errors.append(err)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)

    # Confidence intervals
    cls = []
    for k in range(len(mean_errors)):
        cl = st.t.interval(
            alpha=0.95,
            df=len(errors) - 1,
            loc=mean_errors[k],
            scale=st.sem(errors[:, k]),
        )
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    if len(inference_times) > 0:
        print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times)) * 1000))
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()

    # Default model_type if missing
    if not hasattr(opt, "model_type") or opt.model_type is None:
        opt.model_type = "endodac"

    evaluate(opt)
