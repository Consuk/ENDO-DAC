from __future__ import absolute_import, division, print_function

import argparse
import os
import re
from typing import Dict, List, Optional


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
    for root in (seq_abs, os.path.join(seq_abs, "rgb")):
        for n in _safe_listdir(root):
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


# Count-consistent and disjoint split matching Mono-ViM reported frame counts:
# train=6849, val=1460, test=1706 (total=10015).
#
# Note:
# The Mono-ViM table text appears to include d_4a in train and test, which creates overlap
# and does not match the reported train frame count. This preset uses trans_t2_c in train
# so split counts are consistent and disjoint.
MONO_VIM_SPLITS = {
    "train": [
        "cecum_t1_a",
        "cecum_t1_b",
        "cecum_t2_a",
        "cecum_t2_b",
        "cecum_t2_c",
        "cecum_t3_a",
        "sigmoid_t1_a",
        "sigmoid_t2_a",
        "trans_t1_a",
        "trans_t1_b",
        "trans_t2_a",
        "trans_t2_b",
        "trans_t2_c",
        "trans_t3_a",
        "trans_t3_b",
    ],
    "val": [
        "cecum_t4_a",
        "sigmoid_t3_a",
        "trans_t4_a",
    ],
    "test": [
        "cecum_t4_b",
        "desc_t4_a",
        "sigmoid_t3_b",
        "trans_t4_b",
    ],
}


# Literal table text version (contains overlap: desc_t4_a appears in train and test).
MONO_VIM_REPORTED_SPLITS = {
    "train": [
        "cecum_t1_a",
        "cecum_t1_b",
        "cecum_t2_a",
        "cecum_t2_b",
        "cecum_t2_c",
        "cecum_t3_a",
        "desc_t4_a",
        "sigmoid_t1_a",
        "sigmoid_t2_a",
        "trans_t1_a",
        "trans_t1_b",
        "trans_t2_a",
        "trans_t2_b",
        "trans_t3_a",
        "trans_t3_b",
    ],
    "val": [
        "cecum_t4_a",
        "sigmoid_t3_a",
        "trans_t4_a",
    ],
    "test": [
        "cecum_t4_b",
        "desc_t4_a",
        "sigmoid_t3_b",
        "trans_t4_b",
    ],
}


# MonoLoT (JBHI 2024) official split protocol for C3VD.
# In MonoLoT repo these splits are stored as filtered triplets:
#   <sequence> <prev_idx> <center_idx> <next_idx>
# with gap = 20 and static-frame filtering.
#
# This script converts those lines to EndoDAC-compatible:
#   <sequence_without_under_review_suffix> <center_idx> l
#
# If --monolot_split_dir is not given, this sequence-only fallback is used.
MONO_LOT_SPLITS = {
    "train": [
        "cecum_t1_a",
        "cecum_t1_b",
        "cecum_t2_a",
        "cecum_t2_b",
        "cecum_t2_c",
        "cecum_t3_a",
        "sigmoid_t1_a",
        "sigmoid_t2_a",
        "trans_t1_a",
        "trans_t1_b",
        "trans_t2_a",
        "trans_t2_b",
        "trans_t2_c",
        "trans_t3_a",
        "trans_t3_b",
    ],
    "val": [
        "cecum_t4_a",
        "sigmoid_t3_a",
        "trans_t4_a",
    ],
    "test": [
        "cecum_t4_b",
        "desc_t4_a",
        "sigmoid_t3_b",
        "trans_t4_b",
    ],
}


def _discover_sequences_anywhere(data_path: str) -> Dict[str, str]:
    """
    Discover sequence folders across:
      - <data_path>/<sequence>
      - <data_path>/training/<sequence>
      - <data_path>/validation/<sequence>
      - <data_path>/testing/<sequence>
    Returns: {sequence_name: relative_path_from_data_path}
    """
    found: Dict[str, str] = {}
    duplicate: Dict[str, List[str]] = {}

    # deterministic priority for duplicate sequence names
    roots = [
        "",  # root-level sequences
        "training",
        "validation",
        "testing",
    ]

    for subset in roots:
        root_abs = os.path.join(data_path, subset) if subset else data_path
        if not os.path.isdir(root_abs):
            continue
        for name in sorted(_safe_listdir(root_abs)):
            seq_abs = os.path.join(root_abs, name)
            if not os.path.isdir(seq_abs):
                continue
            if len(_collect_color_indices(seq_abs)) == 0:
                continue

            rel = _as_posix(os.path.join(subset, name)) if subset else _as_posix(name)
            if name in found and found[name] != rel:
                duplicate.setdefault(name, []).append(rel)
                continue
            found[name] = rel

    if duplicate:
        print("[WARN] Duplicate sequence names found; using first discovered path:")
        for seq, other_paths in sorted(duplicate.items()):
            chosen = found[seq]
            print(f"  {seq}: chosen={chosen} ignored={other_paths}")

    return found


def _normalize_monolot_seq_name(folder_token: str) -> str:
    """
    Normalize sequence names from MonoLoT split files to C3VD folder names.
    Examples:
      cecum_t1_a_under_review            -> cecum_t1_a
      cecum_t1_a_under_review/c1v1       -> cecum_t1_a
      training/cecum_t1_a                -> cecum_t1_a
    """
    token = folder_token.strip().replace("\\", "/").strip("/")
    if token == "":
        return token

    head = token.split("/", 1)[0]

    # Drop optional leading subset names if present.
    if head in {"training", "validation", "testing"} and "/" in token:
        head = token.split("/", 2)[1]

    if head.endswith("_under_review"):
        head = head[: -len("_under_review")]

    return head


def _parse_monolot_triplet_line(line: str) -> Optional[str]:
    """
    Convert MonoLoT split line to EndoDAC single-frame line.
    Input examples:
      <seq> <prev> <center> <next>
      <seq> <center>
    Output:
      <normalized_seq> <center> l
    """
    parts = line.strip().split()
    if len(parts) < 2:
        return None

    seq = _normalize_monolot_seq_name(parts[0])
    if seq == "":
        return None

    center_token = None
    if len(parts) >= 4:
        center_token = parts[2]
    else:
        center_token = parts[1]

    try:
        center_idx = int(center_token)
    except Exception:
        return None

    return f"{seq} {center_idx} l"


def _build_lines_from_monolot_split_dir(
    split_dir: str,
    strict: bool = True,
) -> Dict[str, List[str]]:
    files = {
        "train": os.path.join(split_dir, "train_files.txt"),
        "val": os.path.join(split_dir, "val_files.txt"),
        "test": os.path.join(split_dir, "test_files.txt"),
    }

    missing = [k for k, p in files.items() if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(
            "MonoLoT split dir is missing required files for: "
            + ", ".join(missing)
            + f"\nExpected under: {split_dir}"
        )

    out = {"train": [], "val": [], "test": []}
    bad_lines = []
    for split_name in ("train", "val", "test"):
        with open(files[split_name], "r", encoding="utf-8") as f:
            for ln_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if line == "":
                    continue
                converted = _parse_monolot_triplet_line(line)
                if converted is None:
                    bad_lines.append((split_name, ln_no, line))
                    continue
                out[split_name].append(converted)

    if bad_lines and strict:
        preview = "\n".join(
            [f"  {sp}:{ln} -> {txt}" for sp, ln, txt in bad_lines[:10]]
        )
        raise RuntimeError(
            "Failed parsing some MonoLoT split lines:\n"
            + preview
            + ("\n  ..." if len(bad_lines) > 10 else "")
        )

    # Keep deterministic order while removing duplicates (if any).
    for split_name in ("train", "val", "test"):
        seen = set()
        deduped = []
        for line in out[split_name]:
            if line in seen:
                continue
            seen.add(line)
            deduped.append(line)
        out[split_name] = deduped

    return out


def _build_lines_for_named_sequences(
    data_path: str,
    split_spec: Dict[str, List[str]],
    strict: bool = True,
) -> Dict[str, List[str]]:
    seq_map = _discover_sequences_anywhere(data_path)

    missing = []
    filelists = {"train": [], "val": [], "test": []}
    used = set()

    for split_name in ("train", "val", "test"):
        for seq_name in split_spec[split_name]:
            if seq_name not in seq_map:
                missing.append(seq_name)
                continue

            seq_rel = seq_map[seq_name]
            seq_abs = os.path.join(data_path, seq_rel)
            idxs = _collect_color_indices(seq_abs)
            if len(idxs) == 0:
                missing.append(seq_name)
                continue

            filelists[split_name].extend([f"{seq_rel} {idx} l" for idx in idxs])
            used.add(seq_name)

    if missing and strict:
        missing_unique = sorted(set(missing))
        raise RuntimeError(
            "Missing required sequences for selected protocol:\n  "
            + "\n  ".join(missing_unique)
            + "\n\nTip: verify all C3VD zips are downloaded/extracted."
        )

    extras = sorted(set(seq_map.keys()) - used)
    if extras:
        print("[INFO] Sequences present but not used by selected protocol:")
        print("  " + ", ".join(extras))

    return filelists


def _write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def generate_splits(
    data_path,
    output_dir,
    protocol="folder_layout",
    strict=True,
    monolot_split_dir: Optional[str] = None,
):
    has_train = os.path.isdir(os.path.join(data_path, "training"))
    has_val = os.path.isdir(os.path.join(data_path, "validation"))
    has_test = os.path.isdir(os.path.join(data_path, "testing"))

    if protocol == "folder_layout":
        if not (has_train or has_val or has_test):
            raise RuntimeError(
                f"No training/validation/testing folders found under '{data_path}'."
            )

        train_lines = _build_lines_for_subset(data_path, "training") if has_train else []
        val_lines = _build_lines_for_subset(data_path, "validation") if has_val else []
        test_lines = _build_lines_for_subset(data_path, "testing") if has_test else []

        if len(val_lines) == 0:
            val_lines = list(test_lines)

        filelists = {"train": train_lines, "val": val_lines, "test": test_lines}
    elif protocol == "mono_vim":
        filelists = _build_lines_for_named_sequences(
            data_path, MONO_VIM_SPLITS, strict=strict
        )
    elif protocol == "mono_vim_reported":
        filelists = _build_lines_for_named_sequences(
            data_path, MONO_VIM_REPORTED_SPLITS, strict=strict
        )
    elif protocol == "mono_lot":
        if monolot_split_dir:
            filelists = _build_lines_from_monolot_split_dir(
                monolot_split_dir, strict=strict
            )
            print(
                "[INFO] Loaded official MonoLoT split files and converted triplets -> center frames."
            )
        else:
            print(
                "[WARN] --monolot_split_dir not set; using sequence-only MonoLoT split fallback.\n"
                "       This does NOT reproduce MonoLoT static-frame filtering."
            )
            filelists = _build_lines_for_named_sequences(
                data_path, MONO_LOT_SPLITS, strict=strict
            )
    else:
        raise ValueError(
            f"Unknown protocol '{protocol}'. "
            "Expected one of: folder_layout, mono_vim, mono_vim_reported, mono_lot."
        )

    _write_lines(os.path.join(output_dir, "train_files.txt"), filelists["train"])
    _write_lines(os.path.join(output_dir, "val_files.txt"), filelists["val"])
    _write_lines(os.path.join(output_dir, "test_files.txt"), filelists["test"])

    return filelists


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
    parser.add_argument(
        "--protocol",
        type=str,
        default="folder_layout",
        choices=["folder_layout", "mono_vim", "mono_vim_reported", "mono_lot"],
        help=(
            "Split protocol. "
            "folder_layout=use existing training/validation/testing folders; "
            "mono_vim=disjoint count-consistent protocol (6849/1460/1706); "
            "mono_vim_reported=literal table protocol (contains overlap); "
            "mono_lot=MonoLoT protocol (prefer with --monolot_split_dir for exact filtered split)."
        ),
    )
    parser.add_argument(
        "--monolot_split_dir",
        type=str,
        default=None,
        help=(
            "Path to MonoLoT split dir containing train/val/test_files.txt triplets. "
            "Used only when --protocol mono_lot."
        ),
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Allow missing sequences for the selected protocol and write partial lists.",
    )
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.expanduser(args.data_path))
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "splits", "c3vd")
    else:
        output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    filelists = generate_splits(
        data_path,
        output_dir,
        protocol=args.protocol,
        strict=(not args.allow_missing),
        monolot_split_dir=(
            os.path.abspath(os.path.expanduser(args.monolot_split_dir))
            if args.monolot_split_dir
            else None
        ),
    )

    print(f"[OK] Wrote C3VD splits to: {output_dir}")
    print(f"Protocol -> {args.protocol}")
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
