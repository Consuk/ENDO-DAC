from __future__ import absolute_import, division, print_function

import argparse
import os
import re


def _as_posix(path):
    return path.replace("\\", "/")


def _safe_listdir(path):
    if not os.path.isdir(path):
        return []
    try:
        return os.listdir(path)
    except Exception:
        return []


def _parse_color_idx(name):
    m = re.match(r"^(\d+)_color\.(png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)\.(png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _collect_color_indices(seq_abs):
    idxs = []
    for n in _safe_listdir(seq_abs):
        idx = _parse_color_idx(n)
        if idx is not None:
            idxs.append(idx)
    return sorted(set(idxs))


def _discover_sequences(subset_abs):
    seqs = []
    for name in sorted(_safe_listdir(subset_abs)):
        seq_abs = os.path.join(subset_abs, name)
        if not os.path.isdir(seq_abs):
            continue
        if len(_collect_color_indices(seq_abs)) > 0:
            seqs.append(name)
    return seqs


def _build_lines_for_subset(data_path, subset_name):
    subset_abs = os.path.join(data_path, subset_name)
    lines = []
    for seq in _discover_sequences(subset_abs):
        seq_rel = _as_posix(os.path.join(subset_name, seq))
        seq_abs = os.path.join(data_path, seq_rel)
        for idx in _collect_color_indices(seq_abs):
            lines.append(f"{seq_rel} {idx} l")
    return lines


def _write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def generate_splits(data_path, output_dir):
    has_train = os.path.isdir(os.path.join(data_path, "training"))
    has_val = os.path.isdir(os.path.join(data_path, "validation"))
    has_test = os.path.isdir(os.path.join(data_path, "testing"))

    if not (has_train or has_val or has_test):
        raise RuntimeError(
            f"No training/validation/testing folders found under '{data_path}'."
        )

    train_lines = _build_lines_for_subset(data_path, "training") if has_train else []
    val_lines = _build_lines_for_subset(data_path, "validation") if has_val else []
    test_lines = _build_lines_for_subset(data_path, "testing") if has_test else []

    if len(val_lines) == 0:
        val_lines = list(test_lines)

    _write_lines(os.path.join(output_dir, "train_files.txt"), train_lines)
    _write_lines(os.path.join(output_dir, "val_files.txt"), val_lines)
    _write_lines(os.path.join(output_dir, "test_files.txt"), test_lines)

    return {"train": train_lines, "val": val_lines, "test": test_lines}


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

    filelists = generate_splits(data_path, output_dir)

    print(f"[OK] Wrote C3VD splits to: {output_dir}")
    print(
        "Counts -> train: {train}, val: {val}, test: {test}".format(
            train=len(filelists["train"]),
            val=len(filelists["val"]),
            test=len(filelists["test"]),
        )
    )
    for key in ("train", "val", "test"):
        print(f"{key} preview:")
        for line in filelists[key][:5]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
