from __future__ import absolute_import, division, print_function

import os
import csv
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import scipy.stats as st

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors

from datasets.hamlyn_dataset import HamlynDataset
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

import models.endodac as endodac


cv2.setNumThreads(0)


class EvalOptions:
    pass


def load_gt_depths_npz(eval_split: str, splits_dir: str, gt_depths_path: str = None):
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


def build_dataset(opt):
    dataset_key = getattr(opt, "dataset", None) or opt.eval_split
    if dataset_key == "endovis" and opt.eval_split != "endovis":
        dataset_key = opt.eval_split

    datasets_dict = {
        "endovis": SCAREDRAWDataset,
        "scared": SCAREDRAWDataset,
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

    if getattr(opt, "eval_filelist", None):
        fpath = os.path.expanduser(opt.eval_filelist)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing split file: {fpath}")
        filenames = readlines(fpath)
    else:
        fpath = os.path.join(opt.splits_dir, opt.eval_split, "test_files.txt")
        if os.path.exists(fpath):
            filenames = readlines(fpath)
        elif dataset_key == "c3vd" and build_c3vd_default_filelists is not None:
            auto_lists = build_c3vd_default_filelists(
                opt.data_path,
                write_to_splits_dir=os.path.join(opt.splits_dir, opt.eval_split),
            )
            filenames = auto_lists["test"]
            fpath = f"<auto:{opt.eval_split}/test_files.txt>"
        else:
            raise FileNotFoundError(
                f"Missing split file: {fpath}\n"
                f"Either create {opt.splits_dir}/{opt.eval_split}/test_files.txt or pass --eval_filelist."
            )

    if len(filenames) == 0:
        raise RuntimeError(
            f"No evaluation samples found for split '{opt.eval_split}' using filelist '{fpath}'."
        )

    img_ext = ".jpg" if opt.eval_split == "hamlyn" else ".png"
    if getattr(opt, "img_ext", None) is not None:
        img_ext = opt.img_ext

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

    return dataset, filenames, fpath


def load_model(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    if not os.path.isdir(opt.load_weights_folder):
        raise FileNotFoundError(f"Cannot find folder: {opt.load_weights_folder}")

    print(f"-> Loading EndoDAC weights from {opt.load_weights_folder}")

    depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
    if not os.path.isfile(depther_path):
        raise FileNotFoundError(f"Cannot find EndoDAC weights: {depther_path}")

    if opt.height % 14 != 0 or opt.width % 14 != 0:
        raise ValueError(
            f"EndoDAC requires height and width to be multiples of 14. "
            f"Got height={opt.height}, width={opt.width}."
        )

    depther_dict = torch.load(depther_path, map_location="cpu")

    depther = endodac.endodac(
        backbone_size="base",
        r=opt.lora_rank,
        lora_type=opt.lora_type,
        image_shape=(opt.height, opt.width),
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


def compute_confidence_intervals(errors):
    errors = np.asarray(errors, dtype=np.float64)
    mean_errors = np.mean(errors, axis=0)

    if len(errors) < 2:
        cls = np.array([np.nan] * (2 * len(mean_errors)), dtype=np.float64)
        return mean_errors, cls

    cls = []
    for k in range(len(mean_errors)):
        sem = st.sem(errors[:, k])
        if np.isnan(sem):
            cls.extend([np.nan, np.nan])
            continue
        cl = st.t.interval(
            confidence=0.95,
            df=len(errors) - 1,
            loc=mean_errors[k],
            scale=sem,
        )
        cls.append(cl[0])
        cls.append(cl[1])
    return mean_errors, np.asarray(cls, dtype=np.float64)


def resolve_eval_depth_range(opt):
    dataset_key = getattr(opt, "dataset", None) or opt.eval_split
    if dataset_key == "endovis" and opt.eval_split != "endovis":
        dataset_key = opt.eval_split

    if dataset_key == "c3vd" or opt.eval_split == "c3vd":
        min_depth = float(getattr(opt, "c3vd_eval_min_depth", 1e-3))
        max_depth = float(getattr(opt, "c3vd_eval_max_depth", 100.0))
    else:
        min_depth = float(opt.min_depth)
        max_depth = float(opt.max_depth)

    if max_depth <= min_depth:
        raise ValueError(
            f"Invalid evaluation depth range: min={min_depth}, max={max_depth}. "
            "Expected max > min."
        )
    return min_depth, max_depth, dataset_key


def evaluate_one_root(opt, depther, gt_depths):
    dataset, filenames, eval_filelist_path = build_dataset(opt)
    eval_min_depth, eval_max_depth, dataset_key = resolve_eval_depth_range(opt)

    inference_times = []
    pred_disps_list = []
    kept_indices = []

    print(f"-> Using eval filelist: {eval_filelist_path}")
    print(f"-> Computing predictions with size {opt.width}x{opt.height}")
    print(f"-> Eval depth range: [{eval_min_depth:.6f}, {eval_max_depth:.6f}] ({dataset_key})")

    buffer_imgs = []
    buffer_ids = []

    missing = 0
    debug_shown = 0
    total = len(filenames)

    def flush_buffer():
        nonlocal pred_disps_list, inference_times

        if not buffer_imgs:
            return

        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).cuda()

            if getattr(opt, "post_process", False):
                batch_pp = torch.cat((batch, torch.flip(batch, [3])), 0)
            else:
                batch_pp = batch

            t0 = time.time()
            output = depther(batch_pp)
            inference_times.append(time.time() - t0)

            if not isinstance(output, dict) or ("disp", 0) not in output:
                raise RuntimeError("Model output does not contain ('disp', 0).")

            pred_disp, _ = disp_to_depth(output[("disp", 0)], eval_min_depth, eval_max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if getattr(opt, "post_process", False):
                pred_disp = pred_disp[:len(buffer_imgs)]

            pred_disps_list.append(pred_disp)

    for i in tqdm(range(total), total=total):
        try:
            data = dataset[i]

            input_color = data[("color", 0, 0)]
            if not isinstance(input_color, torch.Tensor):
                input_color = torch.as_tensor(input_color)

            buffer_imgs.append(input_color)
            buffer_ids.append(i)

            if len(buffer_imgs) == opt.eval_batch_size:
                flush_buffer()
                kept_indices.extend(buffer_ids)
                buffer_imgs.clear()
                buffer_ids.clear()

        except FileNotFoundError as e:
            missing += 1
            if debug_shown < 10:
                print(f"[DEBUG] Missing sample idx={i} file='{filenames[i]}' -> {e}")
                debug_shown += 1
        except Exception as e:
            missing += 1
            if debug_shown < 10:
                print(f"[DEBUG] Error sample idx={i} file='{filenames[i]}' -> {repr(e)}")
                debug_shown += 1

    if buffer_imgs:
        flush_buffer()
        kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        raise RuntimeError(
            f"No usable samples found in {opt.data_path}. "
            f"Missing/errors: {missing}/{total}"
        )

    if missing > 0:
        print(
            f"-> Using {len(kept_indices)}/{total} samples from split "
            f"(skipped {missing} missing/error samples)"
        )

    pred_disps = np.concatenate(pred_disps_list, axis=0)

    if len(pred_disps) != len(kept_indices):
        raise AssertionError(
            f"Mismatch after filtering: {len(pred_disps)} predictions vs "
            f"{len(kept_indices)} kept indices."
        )

    if isinstance(gt_depths, list):
        gt_depths_sel = [gt_depths[idx] for idx in kept_indices]
    else:
        gt_depths_sel = gt_depths[kept_indices]

    print(f"-> num_pred: {pred_disps.shape[0]} | num_gt_filtered: {len(gt_depths_sel)} | num_split_lines: {len(filenames)}")

    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling")
        opt.disable_median_scaling = True
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = np.asarray(gt_depths_sel[i], dtype=np.float32)
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-6)

        gt_depth[gt_depth >= 65535 - 1e-3] = 0.0
        mask = (gt_depth > eval_min_depth) & (gt_depth < eval_max_depth)

        pred_depth = pred_depth[mask]
        gt_valid = gt_depth[mask]

        if gt_valid.size == 0 or pred_depth.size == 0:
            continue

        if not opt.disable_median_scaling:
            ratio = np.median(gt_valid) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < eval_min_depth] = eval_min_depth
        pred_depth[pred_depth > eval_max_depth] = eval_max_depth

        gt_valid = np.asarray(gt_valid, dtype=np.float32)
        pred_depth = np.asarray(pred_depth, dtype=np.float32)
        err = compute_errors(gt_valid, pred_depth)
        errors.append(err)

    if len(errors) == 0:
        raise RuntimeError(f"No valid depth metrics could be computed for {opt.data_path}")

    if not opt.disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors, cls = compute_confidence_intervals(errors)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    if len(inference_times) > 0:
        print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times)) * 1000))

    return {
        "mean_errors": mean_errors,
        "confidence_intervals": cls,
        "num_samples": len(errors),
        "avg_inference_ms": float(np.mean(np.array(inference_times)) * 1000) if len(inference_times) > 0 else np.nan,
        "num_kept": len(kept_indices),
        "num_missing": missing,
    }

def list_corruption_dirs(root):
    if not os.path.isdir(root):
        return []

    severities = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")
    ]
    if len(severities) > 0:
        return [root]

    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_opt_from_args(args, data_path_root):
    opt = EvalOptions()
    opt.data_path = data_path_root
    opt.eval_split = args.split
    opt.dataset = args.dataset
    opt.height = args.height
    opt.width = args.width
    opt.num_workers = args.num_workers
    opt.eval_batch_size = args.batch_size
    opt.min_depth = args.min_depth
    opt.max_depth = args.max_depth
    opt.eval_stereo = args.eval_stereo
    opt.eval_mono = not args.eval_stereo
    opt.disable_median_scaling = args.eval_stereo
    opt.post_process = args.post_process
    opt.eval_filelist = args.eval_filelist
    opt.gt_depths_path = args.gt_depths_path
    opt.img_ext = args.img_ext
    opt.splits_dir = args.splits_dir
    opt.learn_intrinsics = bool(args.learn_intrinsics)
    opt.c3vd_use_intrinsics_file = bool(args.c3vd_use_intrinsics_file)
    opt.c3vd_intrinsics_path = args.c3vd_intrinsics_path
    opt.c3vd_depth_scale = float(args.c3vd_depth_scale)
    opt.c3vd_eval_min_depth = float(args.c3vd_eval_min_depth)
    opt.c3vd_eval_max_depth = float(args.c3vd_eval_max_depth)

    opt.load_weights_folder = args.load_weights_folder
    opt.lora_rank = args.lora_rank
    opt.lora_type = args.lora_type
    opt.pretrained_path = args.pretrained_path
    opt.residual_block_indexes = args.residual_block_indexes
    opt.include_cls_token = args.include_cls_token
    return opt


def main():
    parser = argparse.ArgumentParser("Evaluate corruptions using EndoDAC with evaluate_depth.py logic")
    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Root containing corruption folders or a single corruption folder with severity_* subfolders")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Folder containing EndoDAC depth_model.pth")
    parser.add_argument("--split", type=str, default="hamlyn")
    parser.add_argument("--splits_dir", type=str, default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--dataset", type=str, default="hamlyn", choices=["hamlyn", "endovis", "scared", "c3vd"])
    parser.add_argument("--data_subdir", type=str, default="",
                        help="Optional subdir inside each severity folder for non-Hamlyn datasets")
    parser.add_argument("--eval_filelist", type=str, default=None)
    parser.add_argument("--gt_depths_path", type=str, default=None)

    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_ext", type=str, default=None,
                        help="Force image extension, e.g. .jpg or .png. By default matches evaluate_depth.py logic")
    parser.add_argument(
        "--learn_intrinsics",
        type=lambda v: str(v).lower() in ("1", "true", "yes", "y"),
        default=True,
    )
    parser.add_argument("--c3vd_use_intrinsics_file", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"), default=True)
    parser.add_argument("--c3vd_intrinsics_path", type=str, default=None)
    parser.add_argument("--c3vd_depth_scale", type=float, default=DEFAULT_C3VD_DEPTH_SCALE)
    parser.add_argument("--c3vd_eval_min_depth", type=float, default=1e-3)
    parser.add_argument("--c3vd_eval_max_depth", type=float, default=100.0)

    parser.add_argument("--eval_stereo", action="store_true")
    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--min_depth", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=50.0)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", type=str, default="dvlora")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--residual_block_indexes", type=int, nargs="*", default=[])
    parser.add_argument("--include_cls_token", action="store_true")

    parser.add_argument("--run_name", type=str, default="endodac_corruptions_eval")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--summary_filename", type=str, default="summary_by_severity.csv")
    parser.add_argument("--per_corruption_filename", type=str, default="summary_by_corruption.csv")
    parser.add_argument("--global_avg_filename", type=str, default="global_average.csv")
    parser.add_argument("--ci_filename", type=str, default="confidence_intervals_by_severity.csv")
    args = parser.parse_args()

    args.splits_dir = os.path.expanduser(args.splits_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("This script expects CUDA just like the current evaluate_depth.py flow.")

    gt_depths, gt_depths_path = load_gt_depths_npz(args.split, args.splits_dir, args.gt_depths_path)
    print(f"-> Using gt depths:    {gt_depths_path}")

    first_opt = build_opt_from_args(args, data_path_root="")
    depther = load_model(first_opt)

    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(f"No corruption directories found in {args.corruptions_root}")

    run_output_dir = os.path.join(args.output_dir, args.run_name)
    safe_makedirs(run_output_dir)

    rows = []
    ci_rows = []

    print("-> Starting corruption evaluation")
    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))
        severities = sorted(
            [
                d for d in os.listdir(corr_dir)
                if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")
            ],
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999
        )

        for sev in severities:
            if args.dataset.lower() == "hamlyn":
                data_root = os.path.join(corr_dir, sev)
            else:
                data_root = os.path.join(corr_dir, sev, args.data_subdir) if args.data_subdir else os.path.join(corr_dir, sev)

            print(f"\n>> {corr_name} / {sev} :: {data_root}")
            if not os.path.isdir(data_root):
                print(f"   [WARN] Missing directory: {data_root}, skipping.")
                continue

            opt = build_opt_from_args(args, data_root)
            result = evaluate_one_root(opt, depther, gt_depths)
            mean_errors = result["mean_errors"]
            cls = result["confidence_intervals"]

            abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
            rows.append([
                corr_name, sev, result["num_samples"], result["avg_inference_ms"],
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
            ])
            ci_rows.append([
                corr_name, sev,
                cls[0], cls[1], cls[2], cls[3], cls[4], cls[5], cls[6], cls[7],
                cls[8], cls[9], cls[10], cls[11], cls[12], cls[13]
            ])

            print(
                f"   abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}"
            )

    if not rows:
        raise RuntimeError("No results were generated.")

    header = ["corruption", "severity", "num_samples", "avg_inference_ms", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    summary_csv = os.path.join(run_output_dir, args.summary_filename)
    save_csv(summary_csv, header, rows)

    ci_header = [
        "corruption", "severity",
        "abs_rel_ci_low", "abs_rel_ci_high",
        "sq_rel_ci_low", "sq_rel_ci_high",
        "rmse_ci_low", "rmse_ci_high",
        "rmse_log_ci_low", "rmse_log_ci_high",
        "a1_ci_low", "a1_ci_high",
        "a2_ci_low", "a2_ci_high",
        "a3_ci_low", "a3_ci_high",
    ]
    ci_csv = os.path.join(run_output_dir, args.ci_filename)
    save_csv(ci_csv, ci_header, ci_rows)

    bucket = defaultdict(list)
    for r in rows:
        bucket[r[0]].append(r)

    per_corr_rows = []
    for corr in sorted(bucket.keys()):
        vals = np.array([r[2:] for r in bucket[corr]], dtype=np.float64)
        means = vals.mean(axis=0).tolist()
        per_corr_rows.append([corr] + means)

    per_corr_header = ["corruption", "num_samples", "avg_inference_ms", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    per_corr_csv = os.path.join(run_output_dir, args.per_corruption_filename)
    save_csv(per_corr_csv, per_corr_header, per_corr_rows)

    all_vals = np.array([r[2:] for r in rows], dtype=np.float64)
    global_means = all_vals.mean(axis=0).tolist()
    global_csv = os.path.join(run_output_dir, args.global_avg_filename)
    save_csv(global_csv, per_corr_header, [["global"] + global_means])

    print("\n======= SUMMARY =======")
    print("By severity         :", summary_csv)
    print("Confidence intervals:", ci_csv)
    print("By corruption       :", per_corr_csv)
    print("Global              :", global_csv)


if __name__ == "__main__":
    main()
