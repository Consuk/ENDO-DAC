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
    # New optional argument to specify a custom split file path
    parser.add_argument('--split_file_path',
                        type=str,
                        default=None,
                        help='optional path to a custom split file (e.g. test_files.txt). Overrides the default file in splits/<split>/')

    parser.add_argument('--output_path',
                    type=str,
                    default=None,
                    help='Optional: explicit output npz path (overrides default naming).')

    # Hamlyn split files are typically 1-based (frame 1 corresponds to 0000000000.png).
    # This offset is applied ONLY when opt.split == "hamlyn".
    # Set to 0 if your split file uses 0-based indexing.
    parser.add_argument('--hamlyn_frame_id_offset',
                        type=int,
                        default=-1,
                        help='Offset to apply to Hamlyn frame_id when reading depth files (default: -1 for 1-based split files).')
    opt = parser.parse_args()

    # Decide which file list to use and where to save the output
    if opt.split_file_path:
        # Use the provided file list
        split_file = opt.split_file_path
        base, _ = os.path.splitext(split_file)
        # Save near the split file: append _gt_depths or _gt_depths_recon
        if opt.useage == "eval":
            output_path = base + "_gt_depths.npz"
        else:
            output_path = base + "_gt_depths_recon.npz"
    else:
        # Use the default split file under splits/<split>
        split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
        if opt.useage == "eval":
            split_file = os.path.join(split_folder, "test_files.txt")
            output_path = os.path.join(split_folder, "gt_depths.npz")
        else:
            split_file = os.path.join(split_folder, "3d_reconstruction.txt")
            output_path = os.path.join(split_folder, "gt_depths_recon.npz")
    if opt.output_path:
        output_path = os.path.expanduser(opt.output_path)

    # Read the list of files
    lines = readlines(split_file)

    # If a custom split file is provided, remember its directory for fallback searches
    split_base_data_path = None
    if opt.split_file_path:
        split_base_data_path = os.path.dirname(opt.split_file_path)

    print("Exporting ground truth depths for {}".format(opt.split))
    gt_depths = []

    for i, line in enumerate(lines):
        # Each line can have 2 or 3 entries: <folder> <frame_index> [<side>]
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line in split file '{split_file}' at line {i + 1}: '{line}'")
        folder = parts[0]
        frame_id = int(parts[1])
        side = parts[2] if len(parts) > 2 else None
        # Print progress
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
            # Hamlyn dataset specific logic.
            # Depth images live under <seq>/<seq>/depth01 or depth02.
            # The folder field might include image01 or image02, or might just be the sequence.

            # Normalize the folder path and split into components
            norm_folder = folder.strip('/')
            folder_parts = norm_folder.split('/')
            depth_base = None

            # Case 1: folder contains 'image01' or 'image02' -> replace with 'depth01' or 'depth02'
            if any(part.startswith('image0') for part in folder_parts):
                # Copy parts and replace image directory names
                parts_copy = []
                for p in folder_parts:
                    if p == 'image01':
                        parts_copy.append('depth01')
                    elif p == 'image02':
                        parts_copy.append('depth02')
                    else:
                        parts_copy.append(p)
                depth_base = os.path.join(*parts_copy)
            else:
                # Case 2: folder does not specify image directory.
                # Build <seq>/<seq>/depth01 or depth02 depending on side
                if len(folder_parts) == 1:
                    # e.g., 'rectified08' -> 'rectified08/rectified08'
                    seq_path = os.path.join(folder_parts[0], folder_parts[0])
                elif len(folder_parts) >= 2:
                    # Use last two segments as sequence path
                    seq_path = os.path.join(folder_parts[-2], folder_parts[-1])
                else:
                    seq_path = norm_folder
                # Determine left/right depth folder: default to depth01 if side unspecified
                depth_sub = 'depth02' if side and side.lower().startswith('r') else 'depth01'
                depth_base = os.path.join(seq_path, depth_sub)

            # ---- DEBUG: confirm which depth folder we are using (depth01 vs depth02) ----
            used_depth_dir = "depth02" if (os.sep + "depth02") in (os.sep + depth_base) or depth_base.endswith("depth02") else "depth01"
            side_flag = (side or "l").lower()[0]
            expected = "depth02" if side_flag == "r" else "depth01"
            print(f"[DEBUG] {folder} | frame_id {frame_id} | side '{side_flag}' | using {used_depth_dir} (expected {expected}) | depth_base={depth_base}")

            # Build candidate depth file names by trying multiple extensions
            # Hamlyn splits in this repo are 1-based (frame 1 -> 0000000000.png),
            # so apply an offset (default: -1). Clamp at 0 for safety.
            frame_id_adj = frame_id + getattr(opt, "hamlyn_frame_id_offset", -1)
            if frame_id_adj < 0:
                frame_id_adj = 0
            print(f"[DEBUG] {folder} | raw frame_id={frame_id} | hamlyn_frame_id_offset={getattr(opt,'hamlyn_frame_id_offset',-1)} | adjusted={frame_id_adj}")

            fname = f"{frame_id_adj:010d}"
            exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
            depth_path = None
            candidate_paths = []
            # Primary search relative to opt.data_path
            for ext in exts:
                candidate_paths.append(os.path.join(opt.data_path, depth_base, fname + ext))
            # Secondary search relative to split file directory (if provided)
            if split_base_data_path:
                for ext in exts:
                    candidate_paths.append(os.path.join(split_base_data_path, depth_base, fname + ext))
            # Find the first candidate that exists
            for cand in candidate_paths:
                if os.path.isfile(cand):
                    depth_path = cand
                    break
            if depth_path is None:
                search_paths = [os.path.join(opt.data_path, depth_base),
                                split_base_data_path and os.path.join(split_base_data_path, depth_base)]
                search_paths_str = ", ".join([p for p in search_paths if p])
                raise FileNotFoundError(
                    f"Could not find depth file for {folder} frame {frame_id} in {search_paths_str}"
                )
            # Read the depth map; keep original bit-depth; convert to float32
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if gt_depth is None:
                raise RuntimeError(f"cv2.imread failed for {depth_path}")
            gt_depth = gt_depth.astype(np.float32)
            # If multi-channel, keep first channel BORRAR LUEGO
            if gt_depth.ndim == 3:
                gt_depth = gt_depth[:, :, 0]

        else:
            raise ValueError(f"Unknown split {opt.split}")

        # append only when we actually defined gt_depth
        if opt.split != "eigen":  # eigen case was commented out anyway
            gt_depths.append(gt_depth.astype(np.float32))

    print("Saving to {}".format(output_path))

    # Some Hamlyn depth maps have different shapes (H, W),
    # so we store them as an object array instead of forcing
    # a single (N, H, W) tensor.
    gt_depths_array = np.array(gt_depths, dtype=object)
    np.savez_compressed(output_path, data=gt_depths_array)


if __name__ == "__main__":
    export_gt_depths_kitti()