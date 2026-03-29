from __future__ import absolute_import, division, print_function

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import PIL.Image as pil

from .mono_dataset import MonoDataset

# C3VD depth specification from the project page:
# depth is clamped to 0-100 mm and linearly encoded to uint16 [0, 65535].
DEFAULT_C3VD_DEPTH_SCALE = 100.0 / 65535.0
DEFAULT_C3VD_NATIVE_WIDTH = 1350.0
DEFAULT_C3VD_NATIVE_HEIGHT = 1080.0


def _as_posix(path: str) -> str:
    return path.replace("\\", "/")


def _readlines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def _write_lines(path: str, lines: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def _safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    try:
        return os.listdir(path)
    except Exception:
        return []


def _parse_idx_suffix(name: str, suffix: str) -> Optional[int]:
    # Matches patterns like:
    #   0_color.png
    #   0000_depth.tiff
    m = re.match(rf"^(\d+)_{suffix}\.[^.]+$", name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_idx_plain(name: str) -> Optional[int]:
    # Matches patterns like:
    #   0000.png
    m = re.match(r"^(\d+)\.[^.]+$", name, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _collect_indexed_files(folder: str, token: str) -> Dict[int, str]:
    """
    Collect files in `folder` keyed by integer frame index.
    Supports both '<idx>_<token>.<ext>' and plain '<idx>.<ext>' naming.
    """
    indexed: Dict[int, str] = {}
    for name in _safe_listdir(folder):
        lower = name.lower()
        if token == "color":
            if not lower.endswith((".png", ".jpg", ".jpeg")):
                continue
        elif token == "depth":
            if not lower.endswith((".tiff", ".tif", ".png")):
                continue
        else:
            continue

        idx = _parse_idx_suffix(name, token)
        if idx is None:
            idx = _parse_idx_plain(name)
        if idx is None:
            continue
        indexed[idx] = name
    return indexed


def _discover_c3vd_roots(sequence_path: str) -> Tuple[str, str]:
    """
    Detect where color/depth files live for a sequence.
    Supports:
      - flat sequence folder (files directly inside sequence_path)
      - nested layout with rgb/ and depth/ subfolders
    """
    candidates_color = [sequence_path, os.path.join(sequence_path, "rgb")]
    candidates_depth = [sequence_path, os.path.join(sequence_path, "depth")]

    color_root = sequence_path
    color_count = -1
    for root in candidates_color:
        count = len(_collect_indexed_files(root, "color"))
        if count > color_count:
            color_count = count
            color_root = root

    depth_root = sequence_path
    depth_count = -1
    for root in candidates_depth:
        count = len(_collect_indexed_files(root, "depth"))
        if count > depth_count:
            depth_count = count
            depth_root = root

    return color_root, depth_root


def _resolve_c3vd_folder_rel(data_path: str, folder_token: str) -> str:
    """
    Resolve split folder token to an on-disk folder relative to data_path.
    Accepts either:
      - explicit subsets (training/..., validation/..., testing/...)
      - sequence-only folder names (e.g. cecum_t1_a)
    """
    token = folder_token.strip().replace("\\", "/").strip("/")

    # Accept MonoLoT-style tokens, e.g.:
    #   cecum_t1_a_under_review
    #   cecum_t1_a_under_review/c1v1
    token_heads = [token]
    if "/" in token:
        token_heads.append(token.split("/", 1)[0])

    normalized_tokens = []
    for t in token_heads:
        if t and t not in normalized_tokens:
            normalized_tokens.append(t)
        if t.endswith("_under_review"):
            t2 = t[: -len("_under_review")]
            if t2 and t2 not in normalized_tokens:
                normalized_tokens.append(t2)

    candidate_rels = []
    for t in normalized_tokens:
        candidate_rels.extend(
            [
                t,
                os.path.join("training", t),
                os.path.join("validation", t),
                os.path.join("testing", t),
            ]
        )

    for rel in candidate_rels:
        if os.path.isdir(os.path.join(data_path, rel)):
            return _as_posix(rel)

    # Best-effort fallback: return the first normalized token.
    if len(normalized_tokens) > 0:
        return _as_posix(normalized_tokens[0])
    return _as_posix(token)


def _scan_sequence_indices(sequence_path: str) -> List[int]:
    color_root, _ = _discover_c3vd_roots(sequence_path)
    color_map = _collect_indexed_files(color_root, "color")
    return sorted(color_map.keys())


def _discover_sequence_dirs(root: str) -> List[str]:
    seqs = []
    for name in sorted(_safe_listdir(root)):
        abs_path = os.path.join(root, name)
        if not os.path.isdir(abs_path):
            continue
        if len(_scan_sequence_indices(abs_path)) > 0:
            seqs.append(name)
    return seqs


def _build_lines_for_subset(data_path: str, subset_rel: str) -> List[str]:
    subset_abs = os.path.join(data_path, subset_rel)
    lines: List[str] = []
    for seq_name in _discover_sequence_dirs(subset_abs):
        seq_rel = _as_posix(os.path.join(subset_rel, seq_name))
        seq_abs = os.path.join(data_path, seq_rel)
        for idx in _scan_sequence_indices(seq_abs):
            lines.append(f"{seq_rel} {idx} l")
    return lines


def build_c3vd_default_filelists(
    data_path: str,
    write_to_splits_dir: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Build C3VD train/val/test filelists from dataset layout.

    Preferred layout:
      data_path/
        training/<sequence>/*
        validation/<sequence>/*
        testing/<sequence>/*

    Fallback layout (no subset folders):
      data_path/<sequence>/*
      -> split by sequence (80/10/10, deterministic by sorted sequence names).
    """
    data_path = os.path.abspath(os.path.expanduser(data_path))
    has_train = os.path.isdir(os.path.join(data_path, "training"))
    has_val = os.path.isdir(os.path.join(data_path, "validation"))
    has_test = os.path.isdir(os.path.join(data_path, "testing"))

    if has_train or has_val or has_test:
        train_lines = _build_lines_for_subset(data_path, "training") if has_train else []
        val_lines = _build_lines_for_subset(data_path, "validation") if has_val else []
        test_lines = _build_lines_for_subset(data_path, "testing") if has_test else []
    else:
        # Fallback for flat layout
        seq_names = _discover_sequence_dirs(data_path)
        n = len(seq_names)
        if n == 0:
            train_lines, val_lines, test_lines = [], [], []
        else:
            n_train = max(1, int(round(0.8 * n)))
            n_val = max(1, int(round(0.1 * n))) if n >= 3 else 0
            n_test = n - n_train - n_val
            if n_test <= 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                elif n_val > 0:
                    n_val -= 1

            train_seqs = seq_names[:n_train]
            val_seqs = seq_names[n_train:n_train + n_val]
            test_seqs = seq_names[n_train + n_val:]

            if len(val_seqs) == 0:
                val_seqs = list(test_seqs)

            def _lines_for_seqs(seq_list: List[str]) -> List[str]:
                lines: List[str] = []
                for seq in seq_list:
                    seq_rel = _as_posix(seq)
                    seq_abs = os.path.join(data_path, seq_rel)
                    for idx in _scan_sequence_indices(seq_abs):
                        lines.append(f"{seq_rel} {idx} l")
                return lines

            train_lines = _lines_for_seqs(train_seqs)
            val_lines = _lines_for_seqs(val_seqs)
            test_lines = _lines_for_seqs(test_seqs)

    if len(val_lines) == 0:
        val_lines = list(test_lines)

    filelists = {
        "train": train_lines,
        "val": val_lines,
        "test": test_lines,
    }

    if write_to_splits_dir is not None:
        split_dir = os.path.abspath(os.path.expanduser(write_to_splits_dir))
        _write_lines(os.path.join(split_dir, "train_files.txt"), filelists["train"])
        _write_lines(os.path.join(split_dir, "val_files.txt"), filelists["val"])
        _write_lines(os.path.join(split_dir, "test_files.txt"), filelists["test"])

    return filelists


def _numbers_from_text(text: str) -> List[float]:
    vals = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    out: List[float] = []
    for v in vals:
        try:
            out.append(float(v))
        except Exception:
            continue
    return out


def _k3_from_values(vals: List[float]) -> np.ndarray:
    n = len(vals)
    if n == 4:
        fx, fy, cx, cy = vals
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32
        )
    if n == 9:
        return np.array(vals, dtype=np.float32).reshape(3, 3)
    if n == 12:
        return np.array(vals, dtype=np.float32).reshape(3, 4)[:, :3]
    if n == 16:
        return np.array(vals, dtype=np.float32).reshape(4, 4)[:3, :3]
    raise ValueError(f"Unsupported intrinsics length: {n}")


def _normalize_k3(
    k3: np.ndarray,
    width_hint: float = DEFAULT_C3VD_NATIVE_WIDTH,
    height_hint: float = DEFAULT_C3VD_NATIVE_HEIGHT,
) -> np.ndarray:
    k3 = k3.astype(np.float32).copy()
    vmax = max(
        abs(float(k3[0, 0])),
        abs(float(k3[1, 1])),
        abs(float(k3[0, 2])),
        abs(float(k3[1, 2])),
    )
    # If values look like pixels, normalize.
    if vmax > 5.0:
        k3[0, 0] /= float(width_hint)
        k3[0, 2] /= float(width_hint)
        k3[1, 1] /= float(height_hint)
        k3[1, 2] /= float(height_hint)
    k3[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return k3


def _k4_from_normalized_k3(k3: np.ndarray) -> np.ndarray:
    k4 = np.eye(4, dtype=np.float32)
    k4[:3, :3] = k3.astype(np.float32)
    return k4


def _parse_intrinsics_file(path: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    width_hint = DEFAULT_C3VD_NATIVE_WIDTH
    height_hint = DEFAULT_C3VD_NATIVE_HEIGHT

    try:
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                width_hint = float(data.get("width", width_hint))
                height_hint = float(data.get("height", height_hint))

                if "K" in data:
                    arr = np.array(data["K"], dtype=np.float32)
                    if arr.shape == (4, 4):
                        k3 = arr[:3, :3]
                    elif arr.shape == (3, 3):
                        k3 = arr
                    else:
                        k3 = _k3_from_values(arr.reshape(-1).tolist())
                    k3 = _normalize_k3(k3, width_hint, height_hint)
                    return _k4_from_normalized_k3(k3)

                needed = ["fx", "fy", "cx", "cy"]
                if all(k in data for k in needed):
                    vals = [
                        float(data["fx"]),
                        float(data["fy"]),
                        float(data["cx"]),
                        float(data["cy"]),
                    ]
                    k3 = _k3_from_values(vals)
                    k3 = _normalize_k3(k3, width_hint, height_hint)
                    return _k4_from_normalized_k3(k3)

            return None

        if ext == ".npy":
            arr = np.load(path)
            arr = np.array(arr, dtype=np.float32)
            if arr.shape == (4, 4):
                k3 = arr[:3, :3]
            elif arr.shape == (3, 3):
                k3 = arr
            else:
                k3 = _k3_from_values(arr.reshape(-1).tolist())
            k3 = _normalize_k3(k3, width_hint, height_hint)
            return _k4_from_normalized_k3(k3)

        if ext == ".npz":
            npz = np.load(path, allow_pickle=True)
            for key in ["K", "k", "intrinsics", "camera_matrix"]:
                if key in npz:
                    arr = np.array(npz[key], dtype=np.float32)
                    if arr.shape == (4, 4):
                        k3 = arr[:3, :3]
                    elif arr.shape == (3, 3):
                        k3 = arr
                    else:
                        k3 = _k3_from_values(arr.reshape(-1).tolist())
                    k3 = _normalize_k3(k3, width_hint, height_hint)
                    return _k4_from_normalized_k3(k3)
            return None

        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
        nums = _numbers_from_text(txt)
        if len(nums) == 0:
            return None

        # Try to read width/height hints if included in the text
        m_w = re.search(r"width\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", txt, flags=re.IGNORECASE)
        m_h = re.search(r"height\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", txt, flags=re.IGNORECASE)
        if m_w:
            width_hint = float(m_w.group(1))
        if m_h:
            height_hint = float(m_h.group(1))

        if len(nums) in (4, 9, 12, 16):
            k3 = _k3_from_values(nums)
            k3 = _normalize_k3(k3, width_hint, height_hint)
            return _k4_from_normalized_k3(k3)

    except Exception:
        return None

    return None


def _default_normalized_k4() -> np.ndarray:
    # Default to the fixed normalized C3VD intrinsics used by MonoLoT.
    # Reason: this is a stronger and more reproducible baseline for C3VD than
    # the generic centered K=[0.5, 0.5], and it better matches common
    # Monodepth2-style training practice on this dataset.
    return np.array(
        [
            [0.56959306, 0.0, 0.5, 0.0],
            [0.0, 0.71185083, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def resolve_c3vd_depth_path(
    data_path: str,
    folder_token: str,
    frame_index: int,
    folder_cache: Optional[Dict[str, dict]] = None,
) -> str:
    """
    Resolve absolute depth path for a C3VD frame.
    Raises FileNotFoundError if exact frame depth is unavailable.
    """
    cache = folder_cache if folder_cache is not None else {}
    token = folder_token.strip()

    if token not in cache:
        folder_rel = _resolve_c3vd_folder_rel(data_path, token)
        seq_abs = os.path.join(data_path, folder_rel)
        color_root, depth_root = _discover_c3vd_roots(seq_abs)
        cache[token] = {
            "folder_rel": folder_rel,
            "color_root": color_root,
            "depth_root": depth_root,
            "depth_map": _collect_indexed_files(depth_root, "depth"),
        }

    info = cache[token]
    depth_map = info["depth_map"]
    if frame_index not in depth_map:
        raise FileNotFoundError(
            f"Missing C3VD depth frame {frame_index} in '{token}' "
            f"(resolved folder: {info['folder_rel']})"
        )

    return os.path.join(info["depth_root"], depth_map[frame_index])


class C3VDDataset(MonoDataset):
    """
    C3VD dataloader compatible with EndoDAC MonoDataset interface.

    Expected split line format:
      <folder> <frame_index> <side>
    where side is typically 'l' (kept for compatibility with existing pipeline).
    """

    def __init__(
        self,
        data_path,
        filenames,
        height,
        width,
        frame_idxs,
        num_scales,
        is_train=False,
        img_ext=".png",
        depth_scale=DEFAULT_C3VD_DEPTH_SCALE,
        intrinsics_path=None,
        use_intrinsics_file=True,
        allow_nearest_when_missing=True,
        **kwargs,
    ):
        self.depth_scale = float(depth_scale)
        self._intrinsics_path_arg = intrinsics_path
        self.use_intrinsics_file = bool(use_intrinsics_file)
        self.allow_nearest_when_missing = bool(allow_nearest_when_missing)

        # Built after base init
        self.folder_info: Dict[str, dict] = {}

        super(C3VDDataset, self).__init__(
            data_path,
            filenames,
            height,
            width,
            frame_idxs,
            num_scales,
            is_train=is_train,
            img_ext=img_ext,
        )

        self._build_folder_cache()
        self.K = self._load_intrinsics_k4()

        # Recompute after folder cache is available (super() computes load_depth too early).
        self.load_depth = self.check_depth()

    def _build_folder_cache(self) -> None:
        unique_folders = set()
        for line in self.filenames:
            parts = line.strip().split()
            if len(parts) > 0:
                unique_folders.add(parts[0])

        for folder_token in sorted(unique_folders):
            folder_rel = _resolve_c3vd_folder_rel(self.data_path, folder_token)
            seq_abs = os.path.join(self.data_path, folder_rel)

            color_root, depth_root = _discover_c3vd_roots(seq_abs)
            color_map = _collect_indexed_files(color_root, "color")
            depth_map = _collect_indexed_files(depth_root, "depth")

            self.folder_info[folder_token] = {
                "folder_rel": folder_rel,
                "sequence_abs": seq_abs,
                "color_root": color_root,
                "depth_root": depth_root,
                "color_map": color_map,
                "depth_map": depth_map,
                "sorted_indices": sorted(color_map.keys()),
            }

    def _auto_intrinsics_file(self) -> Optional[str]:
        if self._intrinsics_path_arg:
            p = os.path.abspath(os.path.expanduser(self._intrinsics_path_arg))
            if os.path.isfile(p):
                return p
            return None

        candidates = [
            os.path.join(self.data_path, "intrinsics.txt"),
            os.path.join(self.data_path, "camera_intrinsics.txt"),
            os.path.join(self.data_path, "K.txt"),
            os.path.join(self.data_path, "calibration", "intrinsics.txt"),
            os.path.join(self.data_path, "calibration", "camera_intrinsics.txt"),
            os.path.join(self.data_path, "calibration", "K.txt"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _load_intrinsics_k4(self) -> np.ndarray:
        if not self.use_intrinsics_file:
            return _default_normalized_k4()

        path = self._auto_intrinsics_file()
        if path is None:
            return _default_normalized_k4()

        k4 = _parse_intrinsics_file(path)
        if k4 is None:
            return _default_normalized_k4()

        return k4.astype(np.float32)

    def _get_nearest_idx(self, sorted_indices: List[int], target: int) -> int:
        if len(sorted_indices) == 0:
            return target
        return min(sorted_indices, key=lambda i: abs(i - target))

    def _resolve_color_name(self, folder: str, frame_index: int) -> str:
        if folder not in self.folder_info:
            raise FileNotFoundError(f"Unknown C3VD folder token '{folder}'")

        info = self.folder_info[folder]
        cmap = info["color_map"]
        if frame_index in cmap:
            return cmap[frame_index]

        if self.allow_nearest_when_missing:
            nearest = self._get_nearest_idx(info["sorted_indices"], frame_index)
            if nearest in cmap:
                return cmap[nearest]

        raise FileNotFoundError(
            f"Missing C3VD color frame {frame_index} in folder '{folder}' "
            f"(resolved path: {info['folder_rel']})"
        )

    def _resolve_depth_name(self, folder: str, frame_index: int) -> Optional[str]:
        if folder not in self.folder_info:
            return None
        dmap = self.folder_info[folder]["depth_map"]
        return dmap.get(frame_index, None)

    def check_depth(self):
        if not hasattr(self, "folder_info"):
            return False
        if len(self.filenames) == 0:
            return False

        try:
            line = self.filenames[0].strip().split()
            if len(line) < 2:
                return False
            folder = line[0]
            if len(line) >= 4:
                frame_index = int(line[2])
            else:
                frame_index = int(line[1])
        except Exception:
            return False

        return self._resolve_depth_name(folder, frame_index) is not None

    def get_color(self, folder, frame_index, side, do_flip):
        _ = side  # kept for interface compatibility
        info = self.folder_info.get(folder)
        if info is None:
            # Resolve lazily if split token wasn't in the initial map.
            folder_rel = _resolve_c3vd_folder_rel(self.data_path, folder)
            seq_abs = os.path.join(self.data_path, folder_rel)
            color_root, depth_root = _discover_c3vd_roots(seq_abs)
            color_map = _collect_indexed_files(color_root, "color")
            self.folder_info[folder] = {
                "folder_rel": folder_rel,
                "sequence_abs": seq_abs,
                "color_root": color_root,
                "depth_root": depth_root,
                "color_map": color_map,
                "depth_map": _collect_indexed_files(depth_root, "depth"),
                "sorted_indices": sorted(color_map.keys()),
            }
            info = self.folder_info[folder]

        color_name = self._resolve_color_name(folder, frame_index)
        color_path = os.path.join(info["color_root"], color_name)
        color = self.loader(color_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, folder, frame_index, side, do_flip):
        _ = side  # kept for interface compatibility
        info = self.folder_info.get(folder)
        if info is None:
            raise FileNotFoundError(f"Unknown C3VD folder token '{folder}'")

        depth_name = self._resolve_depth_name(folder, frame_index)
        if depth_name is None:
            raise FileNotFoundError(
                f"Missing C3VD depth frame {frame_index} in folder '{folder}' "
                f"(resolved path: {info['folder_rel']})"
            )

        depth_path = os.path.join(info["depth_root"], depth_name)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"cv2.imread failed for depth file: {depth_path}")

        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32) * self.depth_scale

        if do_flip:
            depth = np.fliplr(depth)

        return depth

    def get_pose(self, folder, frame_index):
        _ = folder
        _ = frame_index
        raise NotImplementedError("Pose loading is not implemented for C3VD in this pipeline.")
