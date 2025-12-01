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
                        # 🔴 added "hamlyn"
                        choices=["eigen", "eigen_benchmark", "endovis", "hamlyn"])
    parser.add_argument('--useage',
                        type=str,
                        help='gt depth use for evaluation or 3d reconstruction',
                        required=True,
                        choices=["eval", "3d_recon"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    if opt.useage == "eval":
        lines = readlines(os.path.join(split_folder, "test_files.txt"))
        output_path = os.path.join(split_folder, "gt_depths.npz")
    else:
        lines = readlines(os.path.join(split_folder, "3d_reconstruction.txt"))
        output_path = os.path.join(split_folder, "gt_depths_recon.npz")

    print("Exporting ground truth depths for {}".format(opt.split))
    gt_depths = []

    for i, line in enumerate(lines):
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        print(i + 1)
        print(folder)

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
            # 🔵 Hamlyn:
            # lines look like:
            #   rectified08/rectified08/image01 1805 l
            # And we want depth from:
            #   <data_path>/rectified08/rectified08/depth01/0000001805.<ext>

            # replace image01 -> depth01
            depth_folder = folder.replace("image01", "depth01")

            # try a few common depth extensions
            fname = f"{frame_id:010d}"
            exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]
            depth_path = None
            for ext in exts:
                cand = os.path.join(opt.data_path, depth_folder, fname + ext)
                if os.path.isfile(cand):
                    depth_path = cand
                    break

            if depth_path is None:
                raise FileNotFoundError(
                    f"Could not find depth file for {folder} frame {frame_id} "
                    f"under {os.path.join(opt.data_path, depth_folder)} "
                    f"with any of extensions {exts}"
                )

            # keep original values (16-bit etc), no normalization here
            gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if gt_depth is None:
                raise RuntimeError(f"cv2.imread failed for {depth_path}")

            gt_depth = gt_depth.astype(np.float32)

            # if it is 3-channel (unlikely), keep only first channel
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
