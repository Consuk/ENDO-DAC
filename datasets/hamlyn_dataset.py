from __future__ import absolute_import, division, print_function

import os
import random
import re
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms

# Prevent PIL from failing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path: str) -> Image.Image:
    """Load an image file as an RGB PIL Image."""
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class HamlynDataset(data.Dataset):
    """
    Dataset for the Hamlyn endoscopic dataset using explicit file lists.

    Each line in `filenames` is:
        <folder> <frame_index> <side>

    where:
      - folder: e.g. rectified01  (or sometimes rectified01/rectified01 in some splits)
      - frame_index: integer index (Hamlyn typically uses 10-digit zero padded filenames)
      - side: 'l' or 'r'

    Main fix:
      - exact frame matching by default
      - side-aware frame indexing (image01 / image02 tracked independently)
      - optional nearest-neighbor fallback remains available if ever needed
    """

    def __init__(
        self,
        data_path: str,
        filenames: list,
        height: int,
        width: int,
        frame_idxs: list,
        num_scales: int,
        is_train: bool = False,
        img_ext: str = ".jpg",
        use_intrinsics_file: bool = True,
        intrinsics_filename: str = "intrinsics.txt",
        exact_match: bool = True,
        allow_nearest_when_missing: bool = True,
        debug_missing_limit: int = 20,
    ):
        super().__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = int(height)
        self.width = int(width)
        self.frame_idxs = frame_idxs
        self.num_scales = int(num_scales)
        self.is_train = bool(is_train)
        self.img_ext = img_ext
        self._allowed_img_exts = {self.img_ext.lower(), ".jpg", ".jpeg", ".png"}

        self.use_intrinsics_file = bool(use_intrinsics_file)
        self.intrinsics_filename = str(intrinsics_filename)

        self.exact_match = bool(exact_match)
        self.allow_nearest_when_missing = bool(allow_nearest_when_missing)
        self.debug_missing_limit = int(debug_missing_limit)
        self._missing_debug_count = 0

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # Colour jitter parameters (torchvision version compatibility)
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # Resize transforms for each pyramid scale
        self.interp = Image.LANCZOS
        self.resize: Dict[int, transforms.Resize] = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp
            )

        # Fallback "dummy" normalized intrinsics
        self.K_fallback_norm = np.array(
            [[0.5, 0.0, 0.5],
             [0.0, 0.5, 0.5],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # Map each folder token from split -> actual on-disk folder
        self.actual_folder_map: Dict[str, str] = {}

        # Side-aware filename maps:
        # self.index_map[folder]["l"][idx] -> filename in image01
        # self.index_map[folder]["r"][idx] -> filename in image02
        self.index_map: Dict[str, Dict[str, Dict[int, str]]] = {}
        self.sorted_indices: Dict[str, Dict[str, list]] = {}

        unique_folders = set()
        for line in self.filenames:
            parts = line.strip().split()
            if parts:
                unique_folders.add(parts[0])

        for folder in unique_folders:
            candidate_paths = [folder, os.path.join(folder, folder)]
            actual_folder = None

            side_maps = {
                "l": {},
                "r": {},
            }

            for cand in candidate_paths:
                left_path = os.path.join(self.data_path, cand, "image01")
                right_path = os.path.join(self.data_path, cand, "image02")

                if os.path.isdir(left_path) or os.path.isdir(right_path):
                    actual_folder = cand

                    if os.path.isdir(left_path):
                        for fname in os.listdir(left_path):
                            if fname.lower().endswith(tuple(self._allowed_img_exts)):
                                stem = os.path.splitext(fname)[0]
                                try:
                                    idx = int(stem)
                                except ValueError:
                                    continue
                                side_maps["l"][idx] = fname

                    if os.path.isdir(right_path):
                        for fname in os.listdir(right_path):
                            if fname.lower().endswith(tuple(self._allowed_img_exts)):
                                stem = os.path.splitext(fname)[0]
                                try:
                                    idx = int(stem)
                                except ValueError:
                                    continue
                                side_maps["r"][idx] = fname
                    break

            if actual_folder is None:
                actual_folder = folder

            self.actual_folder_map[folder] = actual_folder
            self.index_map[folder] = side_maps
            self.sorted_indices[folder] = {
                "l": sorted(side_maps["l"].keys()),
                "r": sorted(side_maps["r"].keys()),
            }

        # Intrinsics cache: folder -> {"l": K_3x3, "r": K_3x3}
        self._intrinsics_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Depth maps exist, but we keep them off for training to avoid variable-size collation
        self.load_depth = False

    def __len__(self) -> int:
        return len(self.filenames)

    # -------------------------- preprocessing --------------------------

    def preprocess(self, inputs: dict, color_aug) -> None:
        """Resize and augment colour images for each scale (in-place)."""
        for k in list(inputs.keys()):
            if "color" in k:
                _, frame_id, scale_id = k
                if scale_id == -1:
                    for i in range(self.num_scales):
                        inputs[("color", frame_id, i)] = self.resize[i](inputs[k])

        for k in list(inputs.keys()):
            if "color" in k:
                _, frame_id, scale_id = k
                f = inputs[k]
                inputs[("color", frame_id, scale_id)] = self.to_tensor(f)
                inputs[("color_aug", frame_id, scale_id)] = self.to_tensor(color_aug(f))

    # -------------------------- helpers --------------------------

    def get_nearest_index(self, folder: str, target_idx: int, side: str) -> int:
        """Return the nearest available frame index for a given folder and side."""
        mapping = self.index_map.get(folder, {}).get(side, {})
        if not mapping:
            return target_idx
        if target_idx in mapping:
            return target_idx
        candidates = self.sorted_indices.get(folder, {}).get(side, [])
        if not candidates:
            return target_idx
        return min(candidates, key=lambda k: abs(k - target_idx))

    def _candidate_filenames_exact(self, frame_index: int) -> list:
        stem = f"{frame_index:010d}"
        return [
            stem + self.img_ext,
            stem + ".jpg",
            stem + ".jpeg",
            stem + ".png",
        ]

    def _resolve_filename(self, folder: str, frame_index: int, side: str) -> str:
        """
        Resolve the filename for a requested frame.

        By default:
          - exact match is required
        Optional fallback:
          - nearest frame on the same side only
        """
        actual_folder = self.actual_folder_map.get(folder, folder)
        side_dir = "image01" if side == "l" else "image02"

        # 1) Exact match first
        for fname in self._candidate_filenames_exact(frame_index):
            img_path = os.path.join(self.data_path, actual_folder, side_dir, fname)
            if os.path.isfile(img_path):
                return fname

        # 2) Optional nearest fallback
        if self.allow_nearest_when_missing:
            idx = self.get_nearest_index(folder, frame_index, side)
            fname = self.index_map.get(folder, {}).get(side, {}).get(idx)
            if fname is not None:
                return fname

        raise FileNotFoundError(
            f"Could not resolve exact frame for folder={folder}, side={side}, "
            f"frame_index={frame_index} under {os.path.join(self.data_path, actual_folder, side_dir)}"
        )

    def get_color(self, folder: str, frame_index: int, side: str, do_flip: bool) -> Image.Image:
        """Load a colour image from disk."""
        side_dir = "image01" if side == "l" else "image02"
        actual_folder = self.actual_folder_map.get(folder, folder)
        fname = self._resolve_filename(folder, frame_index, side)
        img_path = os.path.join(self.data_path, actual_folder, side_dir, fname)

        img = self.loader(img_path)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def get_depth(self, folder: str, frame_index: int, side: str, do_flip: bool) -> np.ndarray:
        """Load a depth map from disk (used only if self.load_depth is enabled)."""
        depth_dir = "depth01" if side == "l" else "depth02"
        actual_folder = self.actual_folder_map.get(folder, folder)

        base = os.path.splitext(self._resolve_filename(folder, frame_index, side))[0]
        depth_fname = base + ".png"
        depth_path = os.path.join(self.data_path, actual_folder, depth_dir, depth_fname)

        depth = np.array(Image.open(depth_path))
        if do_flip:
            depth = np.fliplr(depth)
        return depth

    def check_depth(self) -> bool:
        """Always return True for Hamlyn (depth files exist)."""
        return True

    @staticmethod
    def _numbers_from_text(text: str) -> list:
        return [float(x) for x in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)]

    @staticmethod
    def _k_from_flat(vals: list) -> np.ndarray:
        """Create a 3x3 K from a flat list of floats."""
        n = len(vals)
        if n == 4:
            fx, fy, cx, cy = vals
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            return K
        if n == 9:
            return np.array(vals, dtype=np.float32).reshape(3, 3)
        if n == 12:
            return np.array(vals, dtype=np.float32).reshape(3, 4)[:, :3]
        if n == 16:
            return np.array(vals, dtype=np.float32).reshape(4, 4)[:3, :3]
        raise ValueError(f"Unsupported intrinsics length: {n}")

    def _load_intrinsics_for_folder(self, folder: str) -> Dict[str, np.ndarray]:
        """
        Load and cache per-folder intrinsics.

        Returns:
            dict with keys {"l","r"} mapping to 3x3 K matrices (float32).
        """
        if folder in self._intrinsics_cache:
            return self._intrinsics_cache[folder]

        actual_folder = self.actual_folder_map.get(folder, folder)
        intr_path = os.path.join(self.data_path, actual_folder, self.intrinsics_filename)

        Ks = {"l": self.K_fallback_norm.copy(), "r": self.K_fallback_norm.copy()}

        if not self.use_intrinsics_file:
            self._intrinsics_cache[folder] = Ks
            return Ks

        try:
            with open(intr_path, "r") as f:
                txt = f.read()
            nums = self._numbers_from_text(txt)

            if len(nums) in (4, 9, 12, 16):
                K = self._k_from_flat(nums)
                Ks = {"l": K, "r": K}
            elif len(nums) in (8, 18, 24, 32):
                half = len(nums) // 2
                K_l = self._k_from_flat(nums[:half])
                K_r = self._k_from_flat(nums[half:])
                Ks = {"l": K_l, "r": K_r}
            else:
                raise ValueError(
                    f"Unexpected number of values in {intr_path}: {len(nums)}"
                )
        except Exception:
            Ks = {"l": self.K_fallback_norm.copy(), "r": self.K_fallback_norm.copy()}

        for s in ("l", "r"):
            K = Ks[s].astype(np.float32)
            if K.shape != (3, 3):
                K = K.reshape(3, 3).astype(np.float32)
            K[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            Ks[s] = K

        self._intrinsics_cache[folder] = Ks
        return Ks

    @staticmethod
    def _to_normalized_K(K: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """
        Convert K to a normalized K (relative to original image size), if needed.
        """
        K = K.astype(np.float32).copy()
        v = max(
            abs(float(K[0, 0])),
            abs(float(K[1, 1])),
            abs(float(K[0, 2])),
            abs(float(K[1, 2])),
        )
        if v > 5.0:
            K[0, 0] /= float(orig_w)
            K[0, 2] /= float(orig_w)
            K[1, 1] /= float(orig_h)
            K[1, 2] /= float(orig_h)

        K[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return K

    def _make_K_pyramid(
        self,
        folder: str,
        side: str,
        orig_w: int,
        orig_h: int,
        do_flip: bool,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], int]:
        """
        Build K/inv_K for all scales.

        Returns:
            (Ks_by_scale, invKs_by_scale, intrinsics_from_file_flag)
        """
        Ks_file = self._load_intrinsics_for_folder(folder)
        K_raw = Ks_file.get(side, self.K_fallback_norm)

        used_file = int(self.use_intrinsics_file and not np.allclose(K_raw, self.K_fallback_norm))
        K_norm = self._to_normalized_K(K_raw, orig_w, orig_h)

        Ks: Dict[int, np.ndarray] = {}
        invKs: Dict[int, np.ndarray] = {}

        for scale in range(self.num_scales):
            w_s = self.width // (2 ** scale)
            h_s = self.height // (2 ** scale)

            K = K_norm.copy()
            K[0, :] *= float(w_s)
            K[1, :] *= float(h_s)

            if do_flip:
                K[0, 2] = float(w_s - 1) - float(K[0, 2])

            K4 = np.eye(4, dtype=np.float32)
            K4[:3, :3] = K.astype(np.float32)

            inv_K4 = np.linalg.pinv(K4)

            Ks[scale] = K4.astype(np.float32)
            invKs[scale] = inv_K4.astype(np.float32)

        return Ks, invKs, used_file

    # -------------------------- main access --------------------------

    def __getitem__(self, index: int) -> dict:
        """Construct a training sample from the provided file list."""
        inputs: Dict = {}

        line = self.filenames[index].strip().split()
        if len(line) == 0:
            raise ValueError(f"Empty filename entry at index {index}")

        folder = line[0]
        frame_index = int(line[1]) if len(line) > 1 else 0
        side = line[2] if len(line) > 2 else "l"
        if side not in ("l", "r"):
            side = "l"

        seq_digits = "".join([c for c in folder if c.isdigit()])
        if len(seq_digits) >= 2:
            sequence = int(seq_digits[-2:])
        elif len(seq_digits) == 1:
            sequence = int(seq_digits)
        else:
            sequence = 0

        inputs["sequence"] = torch.from_numpy(np.array(sequence, dtype=np.int64))
        inputs["frame_id"] = torch.from_numpy(np.array(frame_index, dtype=np.int64))

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # Load frames (raw PIL at scale -1)
        for i in self.frame_idxs:
            if i == "s":
                other_side = "l" if side == "r" else "r"
                img = self.get_color(folder, frame_index, other_side, do_flip)
                inputs[("color", i, -1)] = img
            else:
                img = self.get_color(folder, frame_index + i, side, do_flip)
                inputs[("color", i, -1)] = img

        ref_img: Image.Image = inputs[("color", 0, -1)]
        orig_w, orig_h = ref_img.size

        Ks, invKs, used_file = self._make_K_pyramid(folder, side, orig_w, orig_h, do_flip)

        for scale in range(self.num_scales):
            inputs[("K", scale)] = torch.from_numpy(Ks[scale]).float()
            inputs[("inv_K", scale)] = torch.from_numpy(invKs[scale]).float()

        inputs["intrinsics_from_file"] = torch.tensor(used_file, dtype=torch.int64)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = lambda x: x

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            inputs.pop(("color", i, -1))
            inputs.pop(("color_aug", i, -1))

        if self.load_depth and not self.is_train:
            depth = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = torch.from_numpy(np.expand_dims(depth, 0).astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs