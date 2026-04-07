from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        # ---- robust sequence / keyframe parsing (works for SCARED + Hamlyn) ----
        digits = [int(ch) for ch in folder if ch.isdigit()]

        if len(digits) >= 2:
            sequence = digits[0]
            keyframe = digits[1]
        elif len(digits) == 1:
            sequence = digits[0]
            keyframe = 0
        else:
            # fallback if no digits found (shouldn't really happen)
            sequence = 0
            keyframe = 0

        inputs["sequence"] = torch.from_numpy(np.array(sequence, dtype=np.int64))
        inputs["keyframe"] = torch.from_numpy(np.array(keyframe, dtype=np.int64))
        # ------------------------------------------------------------------------

        
        explicit_frame_map = None
        frame_index = 0
        side = None

        # Default format:
        #   <folder> <frame_idx> <side>
        #
        # Also support explicit triplet format (used by MonoLoT C3VD splits):
        #   <folder> <prev_idx> <center_idx> <next_idx> [side]
        if len(line) >= 4:
            try:
                prev_idx = int(line[1])
                center_idx = int(line[2])
                next_idx = int(line[3])
                explicit_frame_map = {-1: prev_idx, 0: center_idx, 1: next_idx}
                frame_index = center_idx
                side = line[4] if len(line) >= 5 else "l"
            except Exception:
                # Fallback to standard format if not all tokens are frame indices.
                if len(line) >= 2:
                    frame_index = int(line[1])
                side = line[2] if len(line) >= 3 else None
        else:
            if len(line) >= 2:
                frame_index = int(line[1])
            side = line[2] if len(line) >= 3 else None

        # Optional per-sample intrinsics hook (used by C3VD).
        sample_K = self.K
        intrinsics_from_file = None
        if hasattr(self, "get_intrinsics_for_sample"):
            try:
                intr_ret = self.get_intrinsics_for_sample(folder, frame_index, side, do_flip)
                if isinstance(intr_ret, tuple):
                    sample_K = intr_ret[0]
                    if len(intr_ret) > 1:
                        intrinsics_from_file = intr_ret[1]
                elif intr_ret is not None:
                    sample_K = intr_ret
            except Exception:
                sample_K = self.K

        # frame_id = "{:06d}".format(int(frame_index))
        inputs["frame_id"] = torch.from_numpy(np.array(frame_index))
        
        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                if explicit_frame_map is not None and i in explicit_frame_map:
                    source_frame_idx = explicit_frame_map[i]
                else:
                    source_frame_idx = frame_index + i
                inputs[("color", i, -1)] = self.get_color(folder, source_frame_idx, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = sample_K.copy()
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)


            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if intrinsics_from_file is not None:
            inputs["intrinsics_from_file"] = torch.tensor(
                int(bool(intrinsics_from_file)), dtype=torch.int64
            )

        if do_color_aug:
            color_aug = transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        # Optional per-sample valid-region loss mask hook.
        if hasattr(self, "get_loss_mask"):
            try:
                mask = self.get_loss_mask(folder, frame_index, side, do_flip)
            except Exception:
                mask = None

            if mask is not None:
                if isinstance(mask, np.ndarray):
                    mask_t = torch.from_numpy(mask.astype(np.float32))
                elif torch.is_tensor(mask):
                    mask_t = mask.detach().float()
                else:
                    mask_t = torch.tensor(mask, dtype=torch.float32)

                if mask_t.ndim == 2:
                    mask_t = mask_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                elif mask_t.ndim == 3:
                    if mask_t.shape[0] == 1:
                        mask_t = mask_t.unsqueeze(0)  # [1,1,H,W]
                    elif mask_t.shape[-1] == 1:
                        mask_t = mask_t.permute(2, 0, 1).unsqueeze(0)
                    else:
                        mask_t = mask_t[:1].unsqueeze(0)
                elif mask_t.ndim == 4:
                    mask_t = mask_t[:1, :1]
                else:
                    mask_t = mask_t.reshape(1, 1, self.height, self.width)

                mask_t = (mask_t > 0.5).float()

                for scale in range(self.num_scales):
                    h_s = self.height // (2 ** scale)
                    w_s = self.width // (2 ** scale)
                    m_s = F.interpolate(mask_t, size=(h_s, w_s), mode="nearest")
                    inputs[("loss_mask", scale)] = m_s.squeeze(0)

                inputs["loss_mask"] = inputs[("loss_mask", 0)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
            
        # pose_gt = self.get_pose(folder, frame_index)
        # inputs["pose_gt"] = torch.from_numpy(pose_gt)

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
    
    def get_pose(self, folder, frame_index):
        raise NotImplementedError
