# export_gt_depth.py
from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import cv2

from utils.utils import readlines


def _clean_invalid(gt: np.ndarray, invalid_0: bool = True, invalid_65535: bool = True) -> np.ndarray:
    """Set common invalid sentinel values to 0."""
    if gt is None:
        return gt
    m = np.zeros_like(gt, dtype=bool)
    if invalid_0:
        m |= (gt == 0)
    if invalid_65535:
        m |= (gt == 65535)
    if m.any():
        gt = gt.copy()
        gt[m] = 0.0
    return gt


def _decode_hamlyn_gt(raw: np.ndarray,
                     gt_format: str,
                     depth_scale: float,
                     invalid_0: bool,
                     invalid_65535: bool,
                     clip_max: float,
                     fx: float,
                     baseline: float,
                     inv_scale: float,
                     warn_if_max_gt: float,
                     depth_path: str) -> np.ndarray:
    """
    Decode Hamlyn GT to a consistent float32 depth map.

    Supported formats:
      - depth_mm   : raw values are depth in millimeters
      - depth_m    : raw values are depth in meters
      - kitti_256  : raw values are depth_meters * 256 (KITTI-like encoding)
      - disparity  : raw values are disparity in pixels -> depth = fx * baseline / disp
      - inv_depth  : raw values are inverse depth -> depth = inv_scale / inv

    Output depth is in:
      - same units as chosen format AFTER applying depth_scale
        (commonly meters if you set depth_scale appropriately)
    """
    gt = raw.astype(np.float32)

    # If multi-channel, take first channel
    if gt.ndim == 3:
        gt = gt[:, :, 0]

    # Clean invalids before conversion
    gt = _clean_invalid(gt, invalid_0=invalid_0, invalid_65535=invalid_65535)

    if gt_format == "depth_mm":
        # raw millimeters -> optionally scale (e.g., 0.001 to meters)
        gt = gt * depth_scale

    elif gt_format == "depth_m":
        # raw meters -> optional additional scaling
        gt = gt * depth_scale

    elif gt_format == "kitti_256":
        # KITTI-style: depth_png = depth_meters * 256
        gt = (gt / 256.0) * depth_scale

    elif gt_format == "disparity":
        if fx is None or baseline is None:
            raise ValueError("hamlyn_gt_format=disparity requires --hamlyn_fx and --hamlyn_baseline")
        disp = gt
        disp = disp.astype(np.float32)
        disp[disp <= 0] = np.nan
        depth = (fx * baseline) / disp
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        gt = depth * depth_scale

    elif gt_format == "inv_depth":
        inv = gt.astype(np.float32)
        inv[inv <= 0] = np.nan
        depth = (inv_scale / inv)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        gt = depth * depth_scale

    else:
        raise ValueError(f"Unknown hamlyn_gt_format: {gt_format}")

    # Optional outlier clamp: values above clip_max become invalid
    if clip_max is not None:
        gt = gt.copy()
        gt[gt > clip_max] = 0.0

    mx = float(np.max(gt)) if gt.size else 0.0
    if warn_if_max_gt is not None and mx > warn_if_max_gt:
        print(f"[WARN] Large GT max after decode: {mx:.2f} at {depth_path}")

    return gt.astype(np.float32)


def export_gt_depths_kitti():
    parser = argparse.ArgumentParser(description="export_gt_depth")

    parser.add_argument("--data_path", type=str, required=True,
                        help="path to the root of the data")

    parser.add_argument("--split", type=str, required=True,
                        choices=["eigen", "eigen_benchmark", "endovis", "hamlyn"],
                        help="which split to export gt from")

    parser.add_argument("--useage", type=str, required=True,
                        choices=["eval", "3d_recon"],
                        help="gt depth use for evaluation or 3d reconstruction")

    # Optional custom split file
    parser.add_argument("--split_file_path", type=str, default=None,
                        help="optional path to a custom split file (e.g. test_files2.txt). Overrides default splits/<split>/...")

    # Optional explicit output path
    parser.add_argument("--output_path", type=str, default=None,
                        help="Optional: explicit output npz path (overrides default naming).")

    # Hamlyn frame index offset
    parser.add_argument("--hamlyn_frame_id_offset", type=int, default=-1,
                        help="Offset to apply to Hamlyn frame_id when reading depth files (default: -1 for 1-based split files). Set 0 for 0-based.")

    # ===== Hamlyn GT decoding fixes (Issue A) =====
    parser.add_argument("--hamlyn_gt_format", type=str, default="depth_mm",
                        choices=["depth_mm", "depth_m", "kitti_256", "disparity", "inv_depth"],
                        help="How to interpret Hamlyn GT files before saving to NPZ.")
    parser.add_argument("--hamlyn_depth_scale", type=float, default=1.0,
                        help="Extra multiplicative scale after decoding. For depth_mm->meters use 0.001.")
    parser.add_argument("--hamlyn_invalid_0", action="store_true",
                        help="Treat 0 as invalid and set to 0.")
    parser.add_argument("--hamlyn_invalid_65535", action="store_true",
                        help="Treat 65535 as invalid and set to 0.")
    parser.add_argument("--hamlyn_clip_max", type=float, default=None,
                        help="If set: values > clip_max are set invalid (0).")
    parser.add_argument("--hamlyn_warn_if_max_gt", type=float, default=1e6,
                        help="Warn if GT max after decode exceeds this threshold.")

    # For disparity conversion
    parser.add_argument("--hamlyn_fx", type=float, default=None,
                        help="Required for disparity->depth conversion.")
    parser.add_argument("--hamlyn_baseline", type=float, default=None,
                        help="Required for disparity->depth conversion.")
    # For inverse depth conversion
    parser.add_argument("--hamlyn_inv_scale", type=float, default=1.0,
                        help="For inv_depth: depth = inv_scale / inv_value")

    opt = parser.parse_args()

    # Decide split file and output path
    if opt.split_file_path:
        split_file = os.path.expanduser(opt.split_file_path)
        base, _ = os.path.splitext(split_file)
        output_path = base + ("_gt_depths.npz" if opt.useage == "eval" else "_gt_depths_recon.npz")
    else:
        split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
        if opt.useage == "eval":
            split_file = os.path.join(split_folder, "test_files.txt")
            output_path = os.path.join(split_folder, "gt_depths.npz")
        else:
            split_file = os.path.join(split_folder, "3d_reconstruction.txt")
            output_path = os.path.join(split_folder, "gt_depths_recon.npz")

    if opt.output_path:
        output_path = os.path.expanduser(opt.output_path)

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    lines = readlines(split_file)

    # For fallback searches if user provided a split file somewhere else
    split_base_data_path = os.path.dirname(split_file) if opt.split_file_path else None

    print(f"Exporting ground truth depths for split='{opt.split}' useage='{opt.useage}'")
    print(f"Split file: {split_file}")
    print(f"Output:     {output_path}")

    if opt.split == "hamlyn":
        print("Hamlyn decode settings:")
        print(f"  hamlyn_gt_format={opt.hamlyn_gt_format}")
        print(f"  hamlyn_depth_scale={opt.hamlyn_depth_scale}")
        print(f"  hamlyn_frame_id_offset={opt.hamlyn_frame_id_offset}")
        print(f"  invalid_0={opt.hamlyn_invalid_0} invalid_65535={opt.hamlyn_invalid_65535}")
        print(f"  clip_max={opt.hamlyn_clip_max}")

    gt_depths = []

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line in split file '{split_file}' at line {i + 1}: '{line}'")

        folder = parts[0]
        frame_id = int(parts[1])
        side = parts[2] if len(parts) > 2 else None

        print(f"[{i+1:05d}] {folder} frame {frame_id} side {side}")

        if opt.split == "eigen":
            # original logic commented out in your version (kept for compatibility)
            # calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            # velo_filename = os.path.join(opt.data_path, folder, "velodyne_points/data", "{:010d}.bin".format(frame_id))
            # gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
            gt_depth = None

        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(
                opt.data_path, folder, "proj_depth", "groundtruth", "image_02", "{:010d}.png".format(frame_id)
            )
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256.0

        elif opt.split == "endovis":
            f_str = "scene_points{:06d}.tiff".format(frame_id - 1)
            sequence = folder[7]
            data_splt = "train" if int(sequence) < 8 else "test"
            gt_depth_path = os.path.join(opt.data_path, data_splt, folder, "data", "scene_points", f_str)

            gt_depth = cv2.imread(gt_depth_path, 3)
            if gt_depth is None:
                raise RuntimeError(f"cv2.imread failed for {gt_depth_path}")
            gt_depth = gt_depth[:, :, 0]
            gt_depth = gt_depth[0:1024, :].astype(np.float32)

        elif opt.split == "hamlyn":
            # Normalize folder and build depth path
            norm_folder = folder.strip("/")
            folder_parts = norm_folder.split("/")
            depth_base = None

            # Case 1: folder includes image01/image02 -> replace with depth01/depth02
            if any(p.startswith("image0") for p in folder_parts):
                parts_copy = []
                for p in folder_parts:
                    if p == "image01":
                        parts_copy.append("depth01")
                    elif p == "image02":
                        parts_copy.append("depth02")
                    else:
                        parts_copy.append(p)
                depth_base = os.path.join(*parts_copy)
            else:
                # Case 2: folder doesn't include image dir
                if len(folder_parts) == 1:
                    seq_path = os.path.join(folder_parts[0], folder_parts[0])
                elif len(folder_parts) >= 2:
                    seq_path = os.path.join(folder_parts[-2], folder_parts[-1])
                else:
                    seq_path = norm_folder

                depth_sub = "depth02" if (side and side.lower().startswith("r")) else "depth01"
                depth_base = os.path.join(seq_path, depth_sub)

            # Adjust frame index (default -1 for 1-based lists)
            frame_id_adj = frame_id + opt.hamlyn_frame_id_offset
            if frame_id_adj < 0:
                frame_id_adj = 0

            fname = f"{frame_id_adj:010d}"

            # IMPORTANT: do NOT search .jpg/.jpeg for depth maps (prevents wrong file type)
            exts = [".png", ".tiff", ".tif"]

            candidate_paths = []
            for ext in exts:
                candidate_paths.append(os.path.join(opt.data_path, depth_base, fname + ext))
            if split_base_data_path:
                for ext in exts:
                    candidate_paths.append(os.path.join(split_base_data_path, depth_base, fname + ext))

            depth_path = None
            for cand in candidate_paths:
                if os.path.isfile(cand):
                    depth_path = cand
                    break

            if depth_path is None:
                search_paths = [os.path.join(opt.data_path, depth_base)]
                if split_base_data_path:
                    search_paths.append(os.path.join(split_base_data_path, depth_base))
                raise FileNotFoundError(
                    f"Could not find depth file for {folder} frame {frame_id} (adj={frame_id_adj}) in {', '.join(search_paths)}"
                )

            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise RuntimeError(f"cv2.imread failed for {depth_path}")

            # Decode to consistent representation
            gt_depth = _decode_hamlyn_gt(
                raw=raw,
                gt_format=opt.hamlyn_gt_format,
                depth_scale=opt.hamlyn_depth_scale,
                invalid_0=opt.hamlyn_invalid_0,
                invalid_65535=opt.hamlyn_invalid_65535,
                clip_max=opt.hamlyn_clip_max,
                fx=opt.hamlyn_fx,
                baseline=opt.hamlyn_baseline,
                inv_scale=opt.hamlyn_inv_scale,
                warn_if_max_gt=opt.hamlyn_warn_if_max_gt,
                depth_path=depth_path
            )

        else:
            raise ValueError(f"Unknown split {opt.split}")

        # Append if we have a depth map
        if opt.split != "eigen":
            gt_depths.append(gt_depth.astype(np.float32))

    print(f"Saving to {output_path}")
    gt_depths_array = np.array(gt_depths, dtype=object)
    np.savez_compressed(output_path, data=gt_depths_array)
    print("Done.")


if __name__ == "__main__":
    export_gt_depths_kitti()
