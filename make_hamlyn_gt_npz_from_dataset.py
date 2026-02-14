#!/usr/bin/env python3
import argparse
import numpy as np

def read_filelist(path):
    lines = []
    with open(path, "r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                raise ValueError(f"Bad line: {ln}")
            lines.append((parts[0], int(parts[1]), parts[2]))
    return lines

def swap_lr(s):
    return "l" if s == "r" else ("r" if s == "l" else s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--img_ext", default=".jpg")
    ap.add_argument("--swap_depth_dirs", action="store_true",
                    help="Only swaps which depth folder is used (depth01<->depth02) by swapping side when calling get_depth.")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--num_scales", type=int, default=4)
    args = ap.parse_args()

    # Important: import after args so it uses repo modules
    from datasets import HamlynDataset

    entries = read_filelist(args.filelist)
    # HamlynDataset expects filenames as strings "folder frame side"
    filenames = [f"{f} {i} {s}" for (f, i, s) in entries]

    ds = HamlynDataset(
        data_path=args.data_path,
        filenames=filenames,
        height=args.height,
        width=args.width,
        frame_ids=[0],
        num_scales=args.num_scales,
        is_train=False,
        img_ext=args.img_ext,
    )

    gt = []
    for k, (folder, frame, side) in enumerate(entries):
        side_for_depth = swap_lr(side) if args.swap_depth_dirs else side
        d = ds.get_depth(folder, frame, side_for_depth, do_flip=False)  # uint16 numpy
        gt.append(d)

        if (k + 1) % 100 == 0 or k == 0:
            print(f"[{k+1:05d}/{len(entries)}] {folder} frame {frame} side {side} -> depth_side {side_for_depth} shape={d.shape}")

    np.savez_compressed(args.out_npz, data=np.array(gt, dtype=object))
    print(f"Saved: {args.out_npz} maps: {len(gt)}")

if __name__ == "__main__":
    main()
