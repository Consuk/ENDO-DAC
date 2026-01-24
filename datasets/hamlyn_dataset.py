"""
Hamlyn Dataset loader with support for predefined train/test file lists
and monocular training.

This dataset follows a similar interface to the MonoDataset provided in
the original codebase. It accepts a list of lines describing which
frames to load for training or evaluation. Each line should be formatted
as:

    <folder> <frame_index> <side>

Where:
    * ``folder`` is the name of the sequence folder, e.g. ``rectified01``.
    * ``frame_index`` is the index of the frame to load as an integer.
    * ``side`` is either ``l`` or ``r`` indicating the left or right view.

The dataset will automatically locate the corresponding image and depth
files inside the ``image01``/``image02`` and ``depth01``/``depth02``
subfolders of each sequence folder. It will also provide dummy camera
intrinsics for each pyramid scale so that the training code can run
without learning intrinsics from the dataset itself. Intrinsics are
normalised such that at full resolution fx = 0.5 * width and fy =
0.5 * height with principal point at the centre of the image. These
values are unlikely to perfectly match the real camera but are sufficient
for training when the network learns intrinsics via its own decoder.

Note: This implementation intentionally omits any cropping or filtering
logic present in the previous Hamlyn dataset. The goal is to provide
a minimal, deterministic loader which should not modify the data beyond
optional random horizontal flipping and colour jittering.
"""

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
from PIL import Image
from PIL import ImageFile

import torch
import torch.utils.data as data
from torchvision import transforms

# Prevent PIL from failing on truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


def pil_loader(path: str) -> Image.Image:
    """Load an image file as an RGB PIL Image.

    Args:
        path: Absolute path to the image file.

    Returns:
        A PIL Image in RGB mode.
    """
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class HamlynDataset(data.Dataset):
    """Dataset for the Hamlyn endoscopic dataset using explicit file lists.

    This class is designed to be drop-in compatible with the existing
    training pipeline. It behaves similarly to the ``MonoDataset`` but
    instead of inferring file names from a generic naming scheme, it
    constructs a mapping from the provided list of examples to the
    corresponding image and depth files on disk. If a requested frame
    index is missing from the sequence, the nearest available frame is
    used instead. Stereo pairs are supported via the 's' entry in
    ``frame_idxs``; however, by default this dataset is intended for
    monocular training with ``frame_idxs`` such as [0, -1, 1].

    Parameters
    ----------
    data_path : str
        Root directory where all sequence folders reside. Each folder
        should contain ``image01``, ``image02``, ``depth01`` and
        ``depth02`` subfolders.
    filenames : list[str]
        List of strings describing which frames to load. Each entry
        should be whitespace separated as ``<folder> <frame_index> <side>``.
    height : int
        Output image height after resizing.
    width : int
        Output image width after resizing.
    frame_idxs : list
        Frame indices relative to the current frame to load. For
        example [0, -1, 1] loads the current, previous and next frames.
        Use 's' to indicate the opposite stereo camera.
    num_scales : int
        Number of image pyramid scales used in training.
    is_train : bool, optional
        If True, enables random colour jitter and horizontal flips.
    img_ext : str, optional
        File extension for image files (e.g. '.jpg' or '.png'). This is
        used when constructing a fallback filename if a frame index is
        missing from the mapping. Defaults to '.jpg'.
    """

    def __init__(self,
                 data_path: str,
                 filenames: list,
                 height: int,
                 width: int,
                 frame_idxs: list,
                 num_scales: int,
                 is_train: bool = False,
                 img_ext: str = ".jpg"):
        super(HamlynDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.num_scales = num_scales
        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # Colour jitter parameters. To match the behaviour of the original
        # datasets, we attempt to use tuple ranges first. If this fails
        # (older torchvision versions), fall back to scalar ranges.
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            # sanity check instantiation of ColourJitter
            transforms.transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # Define resize transforms for each pyramid scale
        self.interp = Image.LANCZOS
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp
            )

        # Normalised camera intrinsics. These will be scaled by the
        # requested output width and height in __getitem__. The choice of
        # 0.5 for focal lengths and principal point centres places the
        # optical centre in the middle of the image with a field of view
        # of approximately 90 degrees. These values are not dataset
        # specific but serve as a reasonable default when intrinsics are
        # learned during training.
        self.K = np.array(
            [
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        # Build an index mapping for each sequence folder. This allows
        # efficient lookup of filenames given a numeric frame index. If a
        # frame index is not present in the mapping, the nearest
        # available index is used. The mapping only considers files in
        # ``image01`` because all other modalities share the same base
        # filename with a different extension or parent directory.
        #
        # In the Hamlyn dataset, some splits provide folder names
        # without the inner repetition (e.g. "rectified01" instead of
        # "rectified01/rectified01"). To support both conventions, we
        # resolve each folder name to an actual directory that contains
        # the ``image01`` subdirectory and record that mapping. If
        # neither convention yields a valid directory, the folder will
        # map to itself and missing frames will raise a FileNotFoundError
        # during loading.
        self.index_map = {}
        self.sorted_indices = {}
        self.actual_folder_map = {}
        unique_folders = set()
        for line in self.filenames:
            parts = line.strip().split()
            if not parts:
                continue
            folder = parts[0]
            unique_folders.add(folder)
        for folder in unique_folders:
            # Determine the actual folder path that contains the data.
            # Start by assuming the folder provided is correct.
            candidate_paths = []
            # Candidate 1: data_path/folder
            candidate_paths.append(folder)
            # Candidate 2: data_path/folder/folder (handles splits that omit the repeated folder)
            candidate_paths.append(os.path.join(folder, folder))
            actual_folder = None
            index_dict = {}
            for cand in candidate_paths:
                # Check if image01 exists under this candidate
                folder_path = os.path.join(self.data_path, cand, "image01")
                if os.path.isdir(folder_path):
                    actual_folder = cand
                    # Build mapping for this folder
                    for fname in os.listdir(folder_path):
                        # only consider files matching the configured image extension
                        if fname.lower().endswith(self.img_ext.lower()):
                            name_no_ext = os.path.splitext(fname)[0]
                            try:
                                idx = int(name_no_ext)
                            except ValueError:
                                # Skip non-numeric filenames
                                continue
                            index_dict[idx] = fname
                    break
            # Record the actual folder (fall back to the original folder if no valid path found)
            if actual_folder is None:
                # no mapping; we still record an empty index_dict and actual folder same as provided
                actual_folder = folder
            self.actual_folder_map[folder] = actual_folder
            self.index_map[folder] = index_dict
            # Pre-sort indices for quick nearest neighbour lookup
            self.sorted_indices[folder] = sorted(index_dict.keys())

        # Depth is always available for Hamlyn sequences.  However, we only load
        # depth maps during evaluation (not during training) to avoid
        # collate errors when images have different native resolutions.
        # self.load_depth indicates whether depth files exist for this dataset.
        self.load_depth = True

    def __len__(self) -> int:
        return len(self.filenames)

    def preprocess(self, inputs: dict, color_aug) -> None:
        """Resize and augment colour images for each scale.

        This function operates in-place on the ``inputs`` dictionary. It
        creates pyramid scaled versions of each colour image and their
        augmented counterparts. The augmented images use the same random
        transform for all images in the sample to maintain consistency
        between frames.

        Args:
            inputs: Dictionary containing raw PIL Images under keys
                of the form ``("color", frame_id, -1)``. These will be
                replaced by resized tensors at all scales.
            color_aug: A callable that applies colour jitter to an image.
        """
        # First resize raw images to each scale
        for k in list(inputs.keys()):
            if "color" in k:
                _, frame_id, scale_id = k
                if scale_id == -1:
                    for i in range(self.num_scales):
                        inputs[(k[0], frame_id, i)] = self.resize[i](inputs[k])

        # Then convert to tensors and apply colour augmentation
        for k in list(inputs.keys()):
            f = inputs[k]
            if "color" in k:
                _, frame_id, scale_id = k
                inputs[(k[0], frame_id, scale_id)] = self.to_tensor(f)
                inputs[(k[0] + "_aug", frame_id, scale_id)] = self.to_tensor(
                    color_aug(f)
                )

    def get_nearest_index(self, folder: str, target_idx: int) -> int:
        """Return the nearest available frame index for a given folder.

        If the exact ``target_idx`` exists, it is returned. Otherwise the
        closest existing index is chosen.

        Args:
            folder: Name of the sequence folder.
            target_idx: Desired numeric frame index.

        Returns:
            The nearest available frame index.
        """
        mapping = self.index_map.get(folder, {})
        if not mapping:
            return target_idx
        if target_idx in mapping:
            return target_idx
        # Choose the index with minimal absolute difference
        candidates = self.sorted_indices.get(folder, [])
        if not candidates:
            return target_idx
        return min(candidates, key=lambda k: abs(k - target_idx))

    def get_color(self, folder: str, frame_index: int, side: str, do_flip: bool) -> Image.Image:
        """Load a colour image from disk.

        Args:
            folder: Sequence folder name.
            frame_index: Numeric index of the frame to load. If the
                exact index is not present in the sequence, the nearest
                available index is used.
            side: Either 'l' or 'r' indicating which stereo side to load.
            do_flip: If True, perform a horizontal flip of the image.

        Returns:
            A PIL Image in RGB mode.
        """
        # Determine which directory to load from based on the side
        side_dir = "image01" if side == "l" else "image02"
        # Resolve the nearest available index if needed
        idx = self.get_nearest_index(folder, frame_index)
        fname = self.index_map.get(folder, {}).get(idx)
        if fname is None:
            # Fallback to zero-padded naming convention. Hamlyn filenames are
            # typically 10-digit zero padded (e.g. 0000000980.jpg). Use 10 digits
            # here to avoid mismatches when the index_map is empty.
            fname = f"{frame_index:010d}{self.img_ext}"
        # Use the resolved actual folder to construct the image path
        actual_folder = self.actual_folder_map.get(folder, folder)
        img_path = os.path.join(self.data_path, actual_folder, side_dir, fname)
        color = self.loader(img_path)
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)
        return color

    def get_depth(self, folder: str, frame_index: int, side: str, do_flip: bool) -> np.ndarray:
        """Load a depth map from disk.

        Args:
            folder: Sequence folder name.
            frame_index: Numeric index of the frame to load.
            side: Either 'l' or 'r' indicating which stereo side to load.
            do_flip: If True, perform a horizontal flip of the depth map.

        Returns:
            A 2D numpy array containing the depth values.
        """
        depth_dir = "depth01" if side == "l" else "depth02"
        idx = self.get_nearest_index(folder, frame_index)
        fname = self.index_map.get(folder, {}).get(idx)
        if fname is None:
            fname = f"{frame_index:010d}{self.img_ext}"
        # Replace the image extension with .png for depth maps
        base = os.path.splitext(fname)[0]
        depth_fname = base + ".png"
        # Use the resolved actual folder to construct the depth path
        actual_folder = self.actual_folder_map.get(folder, folder)
        depth_path = os.path.join(self.data_path, actual_folder, depth_dir, depth_fname)
        depth = np.array(Image.open(depth_path))
        if do_flip:
            depth = np.fliplr(depth)
        return depth

    def check_depth(self) -> bool:
        """Always return True for Hamlyn since depth maps are provided."""
        return True

    def __getitem__(self, index: int) -> dict:
        """Construct a training sample from the provided file list.

        This method follows the logic of the standard ``MonoDataset`` to
        assemble a dictionary of images and associated metadata. It
        handles temporal neighbours defined in ``self.frame_idxs`` and
        stereo pairing if requested via the 's' token.

        Args:
            index: Index into the ``filenames`` list.

        Returns:
            A dictionary containing tensors for each image scale and
            additional information such as camera intrinsics, depth
            ground truth and stereo extrinsics if applicable.
        """
        inputs = {}

        # Parse the file line describing the current sample
        line = self.filenames[index].strip().split()
        if len(line) == 0:
            raise ValueError(f"Empty filename entry at index {index}")
        folder = line[0]
        frame_index = int(line[1]) if len(line) > 1 else 0
        side = line[2] if len(line) > 2 else "l"

        # Compute sequence id from the folder name (last two digits)
        seq_digits = ''.join([c for c in folder if c.isdigit()])
        if len(seq_digits) >= 2:
            sequence = int(seq_digits[-2:])
        elif len(seq_digits) == 1:
            sequence = int(seq_digits)
        else:
            sequence = 0
        inputs["sequence"] = torch.from_numpy(np.array(sequence, dtype=np.int64))
        inputs["frame_id"] = torch.from_numpy(np.array(frame_index, dtype=np.int64))

        # Determine whether to apply colour augmentation and horizontal flip
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        # Load the requested frames for each index in frame_idxs
        for i in self.frame_idxs:
            if i == "s":
                # Opposite stereo side
                other_side = "l" if side == "r" else "r"
                img = self.get_color(folder, frame_index, other_side, do_flip)
                inputs[("color", i, -1)] = img
            else:
                # Temporal neighbour or current frame
                img = self.get_color(folder, frame_index + i, side, do_flip)
                inputs[("color", i, -1)] = img

        # Prepare dummy intrinsics for each scale. These are scaled
        # versions of the normalised K defined in __init__.
        for scale in range(self.num_scales):
            K = self.K.copy()
            # Scale focal lengths and principal point to the current
            # resolution. Width and height reduction per scale follows
            # powers of two.
            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # Apply the same colour augmentation to all images
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            color_aug = lambda x: x

        # Resize and convert images to tensors; create augmented copies
        self.preprocess(inputs, color_aug)
        # Remove the raw image entries at scale -1
        for i in self.frame_idxs:
            inputs.pop(("color", i, -1))
            inputs.pop(("color_aug", i, -1))

        # Load ground truth depth map only during evaluation (not training).
        # If loaded during training, depth maps of varying native resolution
        # would be stacked by the DataLoader causing a runtime resize error.
        if self.load_depth and not self.is_train:
            depth = self.get_depth(folder, frame_index, side, do_flip)
            # Expand dims to [1, H, W] for consistency
            inputs["depth_gt"] = torch.from_numpy(
                np.expand_dims(depth, 0).astype(np.float32)
            )

        # If stereo is requested, provide the baseline transform
        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            # A nominal baseline of 0.1 metres is assumed
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        return inputs