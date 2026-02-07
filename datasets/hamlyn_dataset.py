"""
Hamlyn Dataset loader (Endo-DAC) with:
- predefined train/val/test file lists, and
- monocular self-supervised training support (temporal neighbours + optional stereo token).

Key change vs the previous version:
This loader can now read **per-sequence intrinsics** from:
    <data_path>/<sequence_dir>/intrinsics.txt

Example structure (as you described):
    Hamlyn/
      rectified01/
        rectified01/
          intrinsics.txt
          image01/0000000001.jpg
          image02/0000000001.jpg
          depth01/0000000001.png
          depth02/0000000001.png

The intrinsics are converted into the (K, inv_K) pyramid expected by the training code.
K is produced in pixel units for each pyramid scale, consistent with how Monodepth-style
pipelines use K in backproject/project layers.

Notes:
- If intrinsics.txt cannot be parsed / is missing, we fall back to a reasonable
  "dummy" normalized K (fx=fy=0.5, cx=cy=0.5). This keeps runs from crashing, but
  for Hamlyn you should ensure intrinsics.txt exists per sequence.
- When random horizontal flip is enabled, we also flip the principal point:
      cx' = (W - 1) - cx
  at the corresponding scale. This matters when using real intrinsics.
"""

from __future__ import absolute_import, division, print_function

import os
import random
import re
from typing import Dict, Optional, Tuple

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

        # Fallback "dummy" normalized intrinsics (only used if intrinsics file is missing/broken)
        self.K_fallback_norm = np.array(
            [[0.5, 0.0, 0.5],
             [0.0, 0.5, 0.5],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # Map each folder name from split -> actual folder containing image01
        self.index_map: Dict[str, Dict[int, str]] = {}
        self.sorted_indices: Dict[str, list] = {}
        self.actual_folder_map: Dict[str, str] = {}

        unique_folders = set()
        for line in self.filenames:
            parts = line.strip().split()
            if parts:
                unique_folders.add(parts[0])

        for folder in unique_folders:
            candidate_paths = [folder, os.path.join(folder, folder)]
            actual_folder = None
            index_dict: Dict[int, str] = {}

            for cand in candidate_paths:
                folder_path = os.path.join(self.data_path, cand, "image01")
                if os.path.isdir(folder_path):
                    actual_folder = cand
                    for fname in os.listdir(folder_path):
                        if fname.lower().endswith(tuple(self._allowed_img_exts)):
                            stem = os.path.splitext(fname)[0]
                            try:
                                idx = int(stem)
                            except ValueError:
                                continue
                            index_dict[idx] = fname
                    break

            if actual_folder is None:
                actual_folder = folder

            self.actual_folder_map[folder] = actual_folder
            self.index_map[folder] = index_dict
            self.sorted_indices[folder] = sorted(index_dict.keys())

        # Intrinsics cache: folder -> {"l": K_3x3, "r": K_3x3}
        self._intrinsics_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # Depth maps exist, but we keep them off for training to avoid variable-size collation
        self.load_depth = False

    def __len__(self) -> int:
        return len(self.filenames)

    # -------------------------- preprocessing --------------------------

    def preprocess(self, inputs: dict, color_aug) -> None:
        """Resize and augment colour images for each scale (in-place)."""
        # First resize raw images to each scale
        for k in list(inputs.keys()):
            if "color" in k:
                _, frame_id, scale_id = k
                if scale_id == -1:
                    for i in range(self.num_scales):
                        inputs[("color", frame_id, i)] = self.resize[i](inputs[k])

        # Then convert to tensors and apply colour augmentation
        for k in list(inputs.keys()):
            if "color" in k:
                _, frame_id, scale_id = k
                f = inputs[k]
                inputs[("color", frame_id, scale_id)] = self.to_tensor(f)
                inputs[("color_aug", frame_id, scale_id)] = self.to_tensor(color_aug(f))

    # -------------------------- helpers --------------------------

    def get_nearest_index(self, folder: str, target_idx: int) -> int:
        """Return the nearest available frame index for a given folder."""
        mapping = self.index_map.get(folder, {})
        if not mapping:
            return target_idx
        if target_idx in mapping:
            return target_idx
        candidates = self.sorted_indices.get(folder, [])
        if not candidates:
            return target_idx
        return min(candidates, key=lambda k: abs(k - target_idx))

    def get_color(self, folder: str, frame_index: int, side: str, do_flip: bool) -> Image.Image:
        """Load a colour image from disk."""
        side_dir = "image01" if side == "l" else "image02"
        idx = self.get_nearest_index(folder, frame_index)
        fname = self.index_map.get(folder, {}).get(idx)
        if fname is None:
            # Hamlyn commonly uses 10-digit zero padding
            fname = f"{frame_index:010d}{self.img_ext}"

        actual_folder = self.actual_folder_map.get(folder, folder)
        img_path = os.path.join(self.data_path, actual_folder, side_dir, fname)

        # If the chosen extension does not exist, try common alternatives
        if not os.path.isfile(img_path):
            stem = os.path.splitext(fname)[0]
            for ext in (".jpg", ".jpeg", ".png"):
                alt = os.path.join(self.data_path, actual_folder, side_dir, stem + ext)
                if os.path.isfile(alt):
                    img_path = alt
                    break

        img = self.loader(img_path)
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def get_depth(self, folder: str, frame_index: int, side: str, do_flip: bool) -> np.ndarray:
        """Load a depth map from disk (used only if self.load_depth is enabled)."""
        depth_dir = "depth01" if side == "l" else "depth02"
        idx = self.get_nearest_index(folder, frame_index)
        fname = self.index_map.get(folder, {}).get(idx)
        if fname is None:
            fname = f"{frame_index:010d}{self.img_ext}"
        base = os.path.splitext(fname)[0]
        depth_fname = base + ".png"

        actual_folder = self.actual_folder_map.get(folder, folder)
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
        # Supports ints + floats, optional signs.
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

            # Support 1 or 2 matrices in the file
            #  - 1 matrix: 4 / 9 / 12 / 16 numbers
            #  - 2 matrices: 8 / 18 / 24 / 32 numbers (left then right)
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
        except Exception as e:
            # Keep fallback, but don't crash training
            # (If you want strict behaviour, change this to 'raise')
            Ks = {"l": self.K_fallback_norm.copy(), "r": self.K_fallback_norm.copy()}

        # Ensure dtype/shape
        for s in ("l", "r"):
            K = Ks[s].astype(np.float32)
            if K.shape != (3, 3):
                K = K.reshape(3, 3).astype(np.float32)
            # enforce standard form
            K[2, :] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            Ks[s] = K

        self._intrinsics_cache[folder] = Ks
        return Ks

    @staticmethod
    def _to_normalized_K(K: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        """
        Convert K to a normalized K (relative to original image size), if needed.

        Heuristic:
          - If focal/principal values are larger than ~5, assume pixel units -> normalize.
          - Otherwise assume already normalized.
        """
        K = K.astype(np.float32).copy()
        v = max(abs(float(K[0, 0])), abs(float(K[1, 1])), abs(float(K[0, 2])), abs(float(K[1, 2])))
        if v > 5.0:
            K[0, 0] /= float(orig_w)
            K[0, 2] /= float(orig_w)
            K[1, 1] /= float(orig_h)
            K[1, 2] /= float(orig_h)
        # Make sure bottom row is [0,0,1]
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

        # Determine if we really used file intrinsics or fell back
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
                # Horizontal flip about the image center
                K[0, 2] = float(w_s - 1) - float(K[0, 2])

            # Use pinv for numerical stability
            inv_K = np.linalg.pinv(K)

            Ks[scale] = K.astype(np.float32)
            invKs[scale] = inv_K.astype(np.float32)

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

        # sequence id from folder digits (last two digits)
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

        # Determine original size from the reference image (after flip size is unchanged)
        ref_img: Image.Image = inputs[("color", 0, -1)]
        orig_w, orig_h = ref_img.size  # PIL uses (W,H)

        # Build K pyramid (per-sequence intrinsics when available)
        Ks, invKs, used_file = self._make_K_pyramid(folder, side, orig_w, orig_h, do_flip)

        for scale in range(self.num_scales):
            inputs[("K", scale)] = torch.from_numpy(Ks[scale]).float()
            inputs[("inv_K", scale)] = torch.from_numpy(invKs[scale]).float()

        # For W&B / debugging
        inputs["intrinsics_from_file"] = torch.tensor(used_file, dtype=torch.int64)

        # Apply same color augmentation to all images
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = lambda x: x

        # Resize + to_tensor + aug
        self.preprocess(inputs, color_aug)

        # Remove raw images
        for i in self.frame_idxs:
            inputs.pop(("color", i, -1))
            inputs.pop(("color_aug", i, -1))

        # Optional depth (disabled by default)
        if self.load_depth and not self.is_train:
            depth = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = torch.from_numpy(np.expand_dims(depth, 0).astype(np.float32))

        # Optional stereo baseline transform (not used for pure monocular runs)
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1  # nominal baseline
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs
