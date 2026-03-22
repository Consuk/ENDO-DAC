from __future__ import absolute_import, division, print_function

import argparse
import os

from datasets.c3vd_dataset import build_c3vd_default_filelists


def main():
    parser = argparse.ArgumentParser(
        description="Generate C3VD split files (train/val/test) from dataset folders."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to C3VD root containing training/validation/testing directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output split directory. Default: <repo>/splits/c3vd",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.expanduser(args.data_path))
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "splits", "c3vd")
    else:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    filelists = build_c3vd_default_filelists(
        data_path=data_path,
        write_to_splits_dir=output_dir,
    )

    print(f"[OK] Wrote C3VD splits to: {output_dir}")
    print(
        "Counts -> train: {train}, val: {val}, test: {test}".format(
            train=len(filelists["train"]),
            val=len(filelists["val"]),
            test=len(filelists["test"]),
        )
    )
    for key in ("train", "val", "test"):
        preview = filelists[key][:3]
        print(f"{key} preview:")
        for line in preview:
            print(f"  {line}")


if __name__ == "__main__":
    main()
