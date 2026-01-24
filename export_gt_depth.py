from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import PIL.Image as pil
import cv2

from utils.utils import readlines
# from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "endovis", "hamlyn"])
    parser.add_argument('--useage',
                        type=str,
                        help='gt depth use for evaluation or 3d reconstruction',
                        required=True,
                        choices=["eval", "3d_recon"])
    # optional external split file
    parser.add_argument('--split_file_path',
                        type=str,
                        default=None,
                        help='optional full path to a custom split file. If provided, this file is used instead of the default split file under splits/<split>/')

    opt = parser.parse_args()

    # Determine which split file to use and where to save results
    if opt.split_file_path:
        # Use external split file specified by user. Determine output path near the split file
        split_file = opt.split_file_path
        base, ext = os.path.splitext(split_file)
        if opt.useage == "eval":
            output_path = base + "_gt_depths.npz"
        else:
            output_path = base + "_gt_depths_recon.npz"
    else:
        # Use default split file located under splits/<split>/
        split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
        if opt.useage == "eval":
            split_file = os.path.join(split_folder, "test_files.txt")
            output_path = os.path.join(split_folder, "gt_depths.npz")
        else:
            split_file = os.path.join(split_folder, "3d_reconstruction.txt")
            output_path = os.path.join(split_folder, "gt_depths_recon.npz")

    # Read split lines
    lines = readlines(split_file)

    print("Exporting ground truth depths for {}".format(opt.split))
    gt_depths = []

    for i, line in enumerate(lines):
        # Each line may have 2 or 3 fields: <folder> <frame_id> [<side>].
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line in split file '{split_file}' at line {i + 1}: '{line}'")
        folder = parts[0]
        frame_id = int(parts[1])
        side = parts[2] if len(parts) > 2 else None

        # Print progress for user clarity
        print(f"[{i+1:05d}] {folder} frame {frame_id}")

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(
                opt.data_path, folder, "velodyne_points/data", "{:010d}.bin".format(frame_id)
            )
            # gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)

        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(
                opt.data_path, folder, "proj_depth",
                "groundtruth", "image_02", "{:010d}.png".format(frame_id)
            )
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256.0

        elif opt.split == "endovis":
            # original EndoVis logic
            f_str = "scene_points{:06d}.tiff".format(frame_id - 1)
            sequence = folder[7]
            data_splt = "train" if int(sequence) < 8 else "test"

            gt_depth_path = os.path.join(
                opt.data_path, data_splt, folder, "data", "scene_points", f_str
            )

            gt_depth = cv2.imread(gt_depth_path, 3)
            gt_depth = gt_depth[:, :, 0]
            gt_depth = gt_depth[0:1024, :]

        elif opt.split == "hamlyn":
            # Hamlyn dataset has a specific structure:
            # The depth maps are stored under <seq>/<seq>/depth0X where X is 1 for left (l) and 2 for right (r).
            # Historically, split files for Hamlyn could include 'image01' or 'image02' in the folder path.
            # They might also omit 'image01' and only include the sequence name (with or without repetition).
            # We'll handle both cases gracefully.

            # Determine the base path to the depth directory based on the provided folder and side.
            # First, normalise folder string to remove any trailing slash.
            folder = folder.strip('/')
            parts_path = folder.split('/')
            depth_base = None

            # Case 1: folder path contains 'image01' or 'image02'. Replace with 'depth01' or 'depth02'.
            if any(part.startswith('image0') for part in parts_path):
                # Determine replacement based on image folder name
                # If no side provided in split file, infer from image folder name
                for idx, p in enumerate(parts_path):
                    if p == 'image01':
                        parts_path[idx] = 'depth01'
                    elif p == 'image02':
                        parts_path[idx] = 'depth02'
                depth_base = os.path.join(*parts_path)
            else:
                # Case 2: folder path does not include image directories.
                # Determine the sequence name and build <seq>/<seq>/depth0X.
                if len(parts_path) == 1:
                    seq = parts_path[0]
                    depth_seq_path = os.path.join(seq, seq)
                elif len(parts_path) >= 2:
                    # Use the last two parts as the seq path, e.g., 'rectified08/rectified08'
                    depth_seq_path = os.path.join(parts_path[-2], parts_path[-1])
                else:
                    depth_seq_path = folder
                # Choose depth01 for left and depth02 for right; default to depth01 if side missing
                if side and side.lower().startswith('r'):
                    depth_sub = 'depth02'
                else:
                    depth_sub = 'depth01'
                depth_base = os.path.join(depth_seq_path, depth_sub)

            # Construct candidate depth file path by searching through known extensions
            fname = f"{frame_id:010d}"
            exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
            depth_path = None
            for ext in exts:
                cand = os.path.join(opt.data_path, depth_base, fname + ext)
                if os.path.isfile(cand):
                    depth_path = cand
                    break

            if depth_path is None:
                raise FileNotFoundError(
                    f"Could not find depth file for {folder} frame {frame_id} "
                    f"under {os.path.join(opt.data_path, depth_base)} "
                    f"with any of extensions {exts}"
                )

            # Load depth (maintain full precision); cv2.imread returns uint16 for png/tiff; we cast to float32
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if gt_depth is None:
                raise RuntimeError(f"cv2.imread failed for {depth_path}")
            gt_depth = gt_depth.astype(np.float32)
            # If the depth map has multiple channels (unlikely), use first channel
            if gt_depth.ndim == 3:
                gt_depth = gt_depth[:, :, 0]

        else:
            raise ValueError(f"Unknown split {opt.split}")

        # append only when we actually defined gt_depth
        if opt.split != "eigen":  # eigen case was commented out anyway
            gt_depths.append(gt_depth.astype(np.float32))

    print("Saving to {}".format(output_path))
    # np.savez_compressed(output_path, data=np.array(gt_depths))
    
    # Some Hamlyn depth maps have different shapes (H, W),
    # so we store them as an object array instead of forcing
    # a single (N, H, W) tensor.
    gt_depths_array = np.array(gt_depths, dtype=object)
    np.savez_compressed(output_path, data=gt_depths_array)


if __name__ == "__main__":
    export_gt_depths_kitti()
