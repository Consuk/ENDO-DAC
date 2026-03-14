from __future__ import absolute_import, division, print_function

import os
import csv
import time
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import scipy.stats as st
from tqdm import tqdm

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors

from datasets.hamlyn_dataset import HamlynDataset
from datasets.scared_dataset import SCAREDRAWDataset
try:
    from datasets.c3vd_dataset import C3VDDataset
except Exception:
    C3VDDataset = None

import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac


cv2.setNumThreads(0)
STEREO_SCALE_FACTOR = 5.4
SPLITS_DIR = os.path.join(os.path.dirname(__file__), "splits")


def load_gt_depths_npz(split, gt_depths_path=None, splits_dir=None):
    """
    Default: splits/<split>/gt_depths.npz
    Override: --gt_depths_path /path/to/custom_gt_depths.npz
    """
    if gt_depths_path is not None:
        gt_path = os.path.expanduser(gt_depths_path)
    else:
        base_splits_dir = os.path.expanduser(splits_dir) if splits_dir is not None else SPLITS_DIR
        gt_path = os.path.join(base_splits_dir, split, "gt_depths.npz")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Missing GT file: {gt_path}\n"
            f"Generate it with export_gt_depth.py for split='{split}', "
            f"or pass --gt_depths_path to point to your custom .npz."
        )

    data_npz = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)
    gt_depths = data_npz["data"]
    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)
    gt_depths = [np.asarray(g, dtype=np.float32) for g in gt_depths]

    return gt_depths, gt_path


def build_dataset(dataset_name, data_path_root, filenames, height, width, img_ext=None):
    """
    Same dataset mapping philosophy as evaluate_depth.py.
    """
    dataset_key = dataset_name.lower()

    datasets_dict = {
        "endovis": SCAREDRAWDataset,
        "scared": SCAREDRAWDataset,
        "hamlyn": HamlynDataset,
    }
    if C3VDDataset is not None:
        datasets_dict["c3vd"] = C3VDDataset

    if dataset_key not in datasets_dict:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Expected one of: {sorted(datasets_dict.keys())}."
        )

    dataset_cls = datasets_dict[dataset_key]

    if img_ext is None:
        img_ext = ".jpg" if dataset_key == "hamlyn" else ".png"

    return dataset_cls(
        data_path_root,
        filenames,
        height,
        width,
        [0],
        4,
        is_train=False,
        img_ext=img_ext,
    )


def load_model(args):
    """
    Mirror evaluate_depth.py behavior:
      - endodac -> depth_model.pth
      - afsfm   -> encoder.pth + depth.pth
    Returns a callable depther(image) -> dict containing ('disp', 0)
    """
    load_weights_folder = os.path.expanduser(args.load_weights_folder)
    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"Cannot find folder: {load_weights_folder}")

    print(f"-> Loading weights from {load_weights_folder}")

    if args.model_type == "endodac":
        depther_path = os.path.join(load_weights_folder, "depth_model.pth")
        if not os.path.isfile(depther_path):
            raise FileNotFoundError(f"Missing EndoDAC weights: {depther_path}")

        depther_dict = torch.load(depther_path, map_location="cpu")

        depther = endodac.endodac(
            backbone_size=args.backbone_size,
            r=args.lora_rank,
            lora_type=args.lora_type,
            image_shape=(args.height, args.width),
            pretrained_path=args.pretrained_path,
            residual_block_indexes=args.residual_block_indexes,
            include_cls_token=args.include_cls_token,
        )
        model_dict = depther.state_dict()
        depther.load_state_dict(
            {k: v for k, v in depther_dict.items() if k in model_dict},
            strict=False,
        )
        depther.cuda().eval()
        return depther

    if args.model_type == "afsfm":
        encoder_path = os.path.join(load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(load_weights_folder, "depth.pth")
        if not os.path.isfile(encoder_path):
            raise FileNotFoundError(f"Missing encoder weights: {encoder_path}")
        if not os.path.isfile(decoder_path):
            raise FileNotFoundError(f"Missing decoder weights: {decoder_path}")

        encoder_dict = torch.load(encoder_path, map_location="cpu")

        encoder = encoders.ResnetEncoder(args.num_layers, False)
        depth_decoder = models_depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict(
            {k: v for k, v in encoder_dict.items() if k in model_dict},
            strict=False,
        )
        models_depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        encoder.cuda().eval()
        models_depth_decoder.cuda().eval()

        def depther(image):
            return models_depth_decoder(encoder(image))

        return depther

    raise ValueError("You must set --model_type endodac or --model_type afsfm")


def list_corruption_dirs(root):
    """
    If root already points to a single corruption (contains severity_*), returns [root].
    Else returns its subdirectories.
    """
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


def predict_disps_for_root(
    depther,
    dataset,
    filenames,
    min_depth,
    max_depth,
    batch_size=16,
    num_workers=4,
    post_process=False,
    strict=False,
):
    """
    Robust prediction loop that skips missing/broken samples in lenient mode,
    matching the spirit of your original corruption script while using EndoDAC/AFSfM inference.
    Returns:
        pred_disps: (N_kept, H, W)
        kept_indices: original sample indices retained
        inference_times: per-batch inference times
    """
    from torch.utils.data import DataLoader

    preds_list = []
    kept_indices = []
    inference_times = []

    valid_items = []
    missing = 0
    debug_shown = 0
    total = len(filenames)

    for i in range(total):
        try:
            _ = dataset[i]
            valid_items.append(i)
        except FileNotFoundError as e:
            missing += 1
            if debug_shown < 5:
                print(f"   [DEBUG] idx={i} file='{filenames[i]}' FileNotFoundError: {e}")
                debug_shown += 1
            if strict:
                raise FileNotFoundError(f"[STRICT] Missing sample idx={i}: {e}")
        except Exception as e:
            missing += 1
            if debug_shown < 5:
                print(f"   [DEBUG] idx={i} file='{filenames[i]}' error={repr(e)}")
                debug_shown += 1
            if strict:
                raise RuntimeError(f"[STRICT] Error loading sample idx={i}: {e}")

    if len(valid_items) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] No usable samples found. Missing/errors: {missing}/{total}"
        )

    if (not strict) and missing > 0:
        print(f"   [INFO] Using {len(valid_items)}/{total} frames (missing {missing}).")

    class IndexedSubset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, indices):
            self.base_dataset = base_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            base_idx = self.indices[idx]
            sample = self.base_dataset[base_idx]
            sample["__orig_idx__"] = base_idx
            return sample

    subset = IndexedSubset(dataset, valid_items)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    printed_intrinsics_debug = False

    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
            if (not printed_intrinsics_debug) and ("intrinsics_from_file" in data):
                try:
                    intr_flag = data["intrinsics_from_file"]
                    uniq = torch.unique(intr_flag).detach().cpu().tolist()
                    print(f"   [DEBUG] Intrinsics loaded from file? unique flags in batch: {uniq} (1=file, 0=fallback)")
                    if int(torch.max(intr_flag).item()) == 1 and (("K", 0) in data):
                        print(f"   [DEBUG] Sample K (first item):\n{data[('K', 0)][0].detach().cpu().numpy()}")
                    elif int(torch.max(intr_flag).item()) == 0:
                        print("   [WARNING] Fallback intrinsics in use — check intrinsics.txt path / flags!")
                except Exception as e:
                    print(f"   [DEBUG] Could not print intrinsics debug info: {e}")
                printed_intrinsics_debug = True

            input_color = data[("color", 0, 0)].cuda()
            orig_indices = data["__orig_idx__"].cpu().numpy().tolist()

            if post_process:
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            t0 = time.time()
            output = depther(input_color)
            inference_times.append(time.time() - t0)

            if not isinstance(output, dict) or ("disp", 0) not in output:
                raise RuntimeError("Model output does not contain ('disp', 0).")

            pred_disp, _ = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            if post_process:
                N = pred_disp.shape[0] // 2
                pred_disp = 0.5 * (pred_disp[:N] + np.flip(pred_disp[N:], axis=2))
                orig_indices = orig_indices[:N]

            preds_list.append(pred_disp)
            kept_indices.extend(orig_indices)

    pred_disps = np.concatenate(preds_list, axis=0)
    return pred_disps, kept_indices, inference_times


def evaluate_one_root(
    data_path_root,
    filenames,
    gt_depths,
    depther,
    dataset_name="hamlyn",
    height=256,
    width=320,
    batch_size=16,
    num_workers=4,
    img_ext=None,
    disable_median_scaling=False,
    pred_depth_scale_factor=1.0,
    strict=False,
    min_depth=1e-3,
    max_depth=150.0,
    post_process=False,
):
    dataset = build_dataset(
        dataset_name=dataset_name,
        data_path_root=data_path_root,
        filenames=filenames,
        height=height,
        width=width,
        img_ext=img_ext,
    )

    pred_disps, kept_indices, inference_times = predict_disps_for_root(
        depther=depther,
        dataset=dataset,
        filenames=filenames,
        min_depth=min_depth,
        max_depth=max_depth,
        batch_size=batch_size,
        num_workers=num_workers,
        post_process=post_process,
        strict=strict,
    )

    if isinstance(gt_depths, list):
        sel_gt = [gt_depths[idx] for idx in kept_indices]
    else:
        sel_gt = gt_depths[kept_indices]

    if pred_disps.shape[0] != len(sel_gt):
        raise AssertionError(
            f"Mismatch after filtering: {pred_disps.shape[0]} predictions vs {len(sel_gt)} GT maps"
        )

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = np.asarray(sel_gt[i], dtype=np.float32)
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / np.maximum(pred_disp, 1e-6)

        gt_depth[gt_depth >= 65535 - 1e-3] = 0.0
        mask = (gt_depth > min_depth) & (gt_depth < max_depth)

        gt_valid = gt_depth[mask]
        pred_valid = pred_depth[mask]

        if pred_valid.size == 0 or gt_valid.size == 0:
            continue

        if pred_depth_scale_factor != 1.0:
            pred_valid *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(gt_valid) / np.median(pred_valid)
            ratios.append(ratio)
            pred_valid *= ratio

        pred_valid[pred_valid < min_depth] = min_depth
        pred_valid[pred_valid > max_depth] = max_depth

        errors.append(compute_errors(np.asarray(gt_valid, dtype=np.float32), np.asarray(pred_valid, dtype=np.float32)))

    if len(errors) == 0:
        raise RuntimeError(f"No valid metrics could be computed for {data_path_root}")

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)

    cls = []
    for k in range(len(mean_errors)):
        if len(errors) > 1:
            cl = st.t.interval(
                confidence=0.95,
                df=len(errors) - 1,
                loc=mean_errors[k],
                scale=st.sem(errors[:, k]),
            )
        else:
            cl = (mean_errors[k], mean_errors[k])
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)

    scaling_stats = None
    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        scaling_stats = (med, np.std(ratios / med))

    avg_inference_ms = None
    if len(inference_times) > 0:
        avg_inference_ms = float(np.mean(np.array(inference_times)) * 1000.0)

    return {
        "mean_errors": mean_errors,
        "confidence_intervals": cls,
        "avg_inference_ms": avg_inference_ms,
        "num_used": len(errors),
        "num_requested": len(filenames),
        "num_pred": pred_disps.shape[0],
        "scaling_stats": scaling_stats,
    }


def main():
    parser = argparse.ArgumentParser("Evaluate corruption benchmark for EndoDAC / AFSfM")

    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Root of corruptions or one specific corruption directory")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Folder with weights: depth_model.pth OR encoder.pth+depth.pth")

    parser.add_argument("--splits_dir", type=str, default=SPLITS_DIR,
                        help="Directory containing splits/<split>/test_files.txt and gt_depths.npz")
    parser.add_argument("--split", type=str, default="hamlyn",
                        help="Split name inside splits/")
    parser.add_argument("--eval_filelist", type=str, default=None,
                        help="Optional custom test_files.txt")
    parser.add_argument("--gt_depths_path", type=str, default=None,
                        help="Optional custom gt_depths.npz")

    parser.add_argument("--dataset", type=str, default="hamlyn",
                        choices=["hamlyn", "endovis", "scared", "c3vd"],
                        help="Dataset loader to use")
    parser.add_argument("--data_subdir", type=str, default="",
                        help="Subfolder inside severity_X for non-Hamlyn datasets if needed")
    parser.add_argument("--img_ext", type=str, default=None,
                        help="Force image extension, e.g. .png or .jpg. Default follows evaluate_depth.py logic")

    parser.add_argument("--model_type", type=str, default="endodac", choices=["endodac", "afsfm"])
    parser.add_argument("--num_layers", type=int, default=18,
                        help="Only used for --model_type afsfm")

    # EndoDAC args mirroring evaluate_depth.py
    parser.add_argument("--backbone_size", type=str, default="base")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_type", type=str, default="dvlora")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--include_cls_token", action="store_true")
    parser.add_argument("--residual_block_indexes", nargs="+", type=int, default=[])

    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--post_process", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--eval_stereo", action="store_true")
    parser.add_argument("--eval_mono", action="store_true")

    parser.add_argument("--min_depth", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=50.0)

    parser.add_argument("--run_name", type=str, default="corruptions_eval")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--summary_filename", type=str, default="summary_by_severity.csv")
    parser.add_argument("--per_corruption_filename", type=str, default="summary_by_corruption.csv")
    parser.add_argument("--global_avg_filename", type=str, default="global_average.csv")
    parser.add_argument("--ci_filename", type=str, default="confidence_intervals_by_severity.csv")

    args = parser.parse_args()

    splits_dir = os.path.expanduser(args.splits_dir)

    if not (args.eval_mono or args.eval_stereo):
        args.eval_mono = True

    assert sum((args.eval_mono, args.eval_stereo)) == 1, \
        "Choose mono or stereo with --eval_mono or --eval_stereo"

    test_files_path = os.path.expanduser(args.eval_filelist) if args.eval_filelist else os.path.join(splits_dir, args.split, "test_files.txt")
    if not os.path.isfile(test_files_path):
        raise FileNotFoundError(f"Missing split file: {test_files_path}")

    test_files = readlines(test_files_path)
    gt_depths, gt_path = load_gt_depths_npz(args.split, args.gt_depths_path, splits_dir=splits_dir)

    print(f"-> Using eval filelist: {test_files_path}")
    print(f"-> Using gt depths:    {gt_path}")

    if len(test_files) != len(gt_depths):
        print(
            "[WARN] test_files and gt_depths do not have the same length. "
            "The script will continue and filter according to actually usable samples."
        )

    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = STEREO_SCALE_FACTOR if args.eval_stereo else 1.0

    depther = load_model(args)

    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(f"No corruption folders found in {args.corruptions_root}")

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
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999,
        )

        for sev in severities:
            if args.dataset.lower() == "hamlyn":
                data_root = os.path.join(corr_dir, sev)
            else:
                data_root = os.path.join(corr_dir, sev, args.data_subdir) if args.data_subdir else os.path.join(corr_dir, sev)

            print(f"\n>> {corr_name} / {sev} :: {data_root}")

            if not os.path.isdir(data_root):
                print(f"   [WARN] Missing directory {data_root}, skipping.")
                continue

            try:
                result = evaluate_one_root(
                    data_path_root=data_root,
                    filenames=test_files,
                    gt_depths=gt_depths,
                    depther=depther,
                    dataset_name=args.dataset,
                    height=args.height,
                    width=args.width,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    img_ext=args.img_ext,
                    disable_median_scaling=disable_median_scaling,
                    pred_depth_scale_factor=pred_depth_scale_factor,
                    strict=args.strict,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    post_process=args.post_process,
                )

                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = result["mean_errors"].tolist()
                rows.append([corr_name, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, result["num_pred"], result["num_used"], result["avg_inference_ms"]])

                ci = result["confidence_intervals"].tolist()
                ci_rows.append([corr_name, sev] + ci)

                if result["scaling_stats"] is not None:
                    med, std = result["scaling_stats"]
                    print(f"   Scaling ratios | med: {med:0.3f} | std: {std:0.3f}")

                print(
                    f"   abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                    f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}"
                )
                if result["avg_inference_ms"] is not None:
                    print(f"   avg inference time: {result['avg_inference_ms']:.1f} ms")

            except Exception as e:
                print(f"   [SKIP] {e}")

    if not rows:
        print("\n-> No results were generated.")
        return

    header = ["corruption", "severity", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "num_pred", "num_eval", "avg_inference_ms"]
    summary_csv = os.path.join(run_output_dir, args.summary_filename)
    save_csv(summary_csv, header, rows)
    print(f"\n-> Main CSV saved to: {summary_csv}")

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
    print(f"-> Confidence intervals CSV saved to: {ci_csv}")

    bucket = defaultdict(list)
    for r in rows:
        bucket[r[0]].append(r)

    per_corr_rows = []
    for corr in sorted(bucket.keys()):
        vals = np.array([r[2:9] for r in bucket[corr]], dtype=np.float64)
        means = vals.mean(axis=0).tolist()
        per_corr_rows.append([corr] + means)

    per_corr_header = ["corruption", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    per_corr_csv = os.path.join(run_output_dir, args.per_corruption_filename)
    save_csv(per_corr_csv, per_corr_header, per_corr_rows)
    print(f"-> Per-corruption average saved to: {per_corr_csv}")

    all_vals = np.array([r[2:9] for r in rows], dtype=np.float64)
    global_means = all_vals.mean(axis=0).tolist()
    global_csv = os.path.join(run_output_dir, args.global_avg_filename)
    save_csv(global_csv, per_corr_header, [["global"] + global_means])
    print(f"-> Global average saved to: {global_csv}")

    print("\n======= SUMMARY =======")
    print("Main file     :", summary_csv)
    print("CI file       :", ci_csv)
    print("Per corruption:", per_corr_csv)
    print("Global        :", global_csv)


if __name__ == "__main__":
    main()
