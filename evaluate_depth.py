from __future__ import absolute_import, division, print_function

import os
import cv2
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

import scipy.stats as st

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from options import MonodepthOptions

from datasets.scared_dataset import SCAREDRAWDataset
# NOTE: We intentionally do NOT use ENDO-DAC's HamlynDataset for Hamlyn split anymore,
# because it scans the whole dataset and ignores test_files.txt.

import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac


cv2.setNumThreads(0)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")


# -------------------------
# Hamlyn Split Dataset (A-logic)
# -------------------------
class HamlynSplitDataset(Dataset):
    """
    A minimal dataset that:
    - reads frames from splits/hamlyn/test_files.txt
    - loads ONLY left image (image01) as ("color",0,0)
    - resizes to (height,width)
    This ensures:
    - dataset length matches your split txt (e.g., ~1701)
    - batching works without collate issues
    """
    def __init__(self, data_path, filenames, height, width, img_ext=".jpg"):
        self.data_path = data_path
        self.filenames = [l.strip() for l in filenames if len(l.strip()) > 0]
        self.height = height
        self.width = width
        self.img_ext = img_ext

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def _resolve_left_image_path(self, line):
        """
        Supports multiple possible formats in test_files.txt.

        Common possible formats:
        1) "<relative_path_to_image>" (already points to an image file)
        2) "<sequence_or_folder> <frame_id>" (Monodepth style)
           We'll try to build something like:
           - data_path/<seq>/<frame>.jpg
           - data_path/<seq>/image01/<frame>.jpg
           - data_path/<seq>/<seq>/image01/<frame>.jpg  (Hamlyn rectifiedXX/rectifiedXX/image01)
        3) "<seq>/<seq> <frame_id>"
        """
        parts = line.split()

        # Case 1: single token
        if len(parts) == 1:
            rel = parts[0]
            # If it already ends with image extension, use directly
            if rel.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                p = os.path.join(self.data_path, rel)
                if os.path.exists(p):
                    return p

            # Otherwise attempt to interpret as folder/frame without id -> not possible
            # Fallthrough: treat as folder and try frame=0000000000
            seq = rel
            frame = "0000000000"
        else:
            seq, frame = parts[0], parts[1]

        # Ensure frame has extension
        if not frame.lower().endswith((".jpg", ".jpeg", ".png")):
            frame_file = frame + self.img_ext
        else:
            frame_file = frame

        # Build candidates (try multiple)
        candidates = [
            os.path.join(self.data_path, seq, frame_file),
            os.path.join(self.data_path, seq, "image01", frame_file),
            os.path.join(self.data_path, seq, seq, "image01", frame_file),
            os.path.join(self.data_path, seq, seq, "image_1", frame_file),
            os.path.join(self.data_path, seq, "rectified01", frame_file),
        ]

        for p in candidates:
            if os.path.exists(p):
                return p

        # If nothing found, raise a clear error
        raise FileNotFoundError(
            f"[HamlynSplitDataset] Could not resolve image path from line='{line}'. "
            f"Tried: {candidates}"
        )

    def __getitem__(self, idx):
        line = self.filenames[idx]
        img_path = self._resolve_left_image_path(line)

        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height), resample=Image.BILINEAR)
        img_t = self.to_tensor(img)  # (3,H,W), float [0,1]

        # Return monodepth-style key
        return {
            ("color", 0, 0): img_t,
        }


def load_gt_depths_npz(eval_split):
    gt_path = os.path.join(splits_dir, eval_split, "gt_depths.npz")
    data_npz = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)
    gt_depths = data_npz["data"]
    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)
    return gt_depths


def evaluate(opt):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Choose mono or stereo with --eval_mono or --eval_stereo"

    # ----------------------------
    # Model loading (ENDO-DAC / B)
    # ----------------------------
    if opt.ext_disp_to_eval is None:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), \
            f"Cannot find folder: {opt.load_weights_folder}"

        print(f"-> Loading weights from {opt.load_weights_folder}")

        if opt.model_type == 'endodac':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path, map_location="cpu")

            depther = endodac.endodac(
                backbone_size="base",
                r=opt.lora_rank, lora_type=opt.lora_type,
                image_shape=(224, 280),
                pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token
            )
            model_dict = depther.state_dict()
            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict}, strict=False)
            depther.cuda().eval()

        elif opt.model_type == 'afsfm':
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path, map_location="cpu")

            encoder = encoders.ResnetEncoder(opt.num_layers, False)
            depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict}, strict=False)
            depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

            encoder.cuda().eval()
            depth_decoder.cuda().eval()

            def depther(image):
                return depth_decoder(encoder(image))

        else:
            raise ValueError("You must set --model_type endodac or --model_type afsfm")

    else:
        print(f"-> Loading predictions from {opt.ext_disp_to_eval}")
        pred_disps = np.load(opt.ext_disp_to_eval)

    # ----------------------------
    # Dataset (A-logic for Hamlyn)
    # ----------------------------
    img_ext = ".png" if opt.png else ".jpg"

    if opt.eval_split == "hamlyn":
        split_file = os.path.join(splits_dir, "hamlyn", "test_files.txt")
        assert os.path.exists(split_file), f"Missing split file: {split_file}"
        filenames = readlines(split_file)

        dataset = HamlynSplitDataset(
            data_path=opt.data_path,
            filenames=filenames,
            height=opt.height,
            width=opt.width,
            img_ext=img_ext
        )

        # Safe batching (dataset returns only fixed-size tensors)
        batch_size = getattr(opt, "eval_batch_size", 16)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False
        )

        # Load A-style GT
        gt_depths = load_gt_depths_npz("hamlyn")

    elif opt.eval_split == "endovis":
        # keep B behavior (dataset provides ordering)
        filenames = readlines(os.path.join(splits_dir, "endovis", "test_files.txt"))
        dataset = SCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width, [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)

        gt_depths = load_gt_depths_npz("endovis")

    else:
        raise ValueError("This script currently supports --eval_split hamlyn or endovis (A-logic version).")

    # ----------------------------
    # Predictions (A-style: predict first)
    # ----------------------------
    inference_times = []
    if opt.ext_disp_to_eval is None:
        print(f"-> Computing predictions with size {opt.width}x{opt.height}")
        pred_disps_list = []

        with torch.no_grad():
            for step_i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                t0 = time.time()
                output = depther(input_color)
                inference_times.append(time.time() - t0)

                if not isinstance(output, dict) or ("disp", 0) not in output:
                    raise RuntimeError("Model output does not contain ('disp',0).")

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()  # (B,H,W)

                # A-style: do NOT apply batch_post_process_disparity even if post_process True (matches your A)
                pred_disps_list.append(pred_disp)

        pred_disps = np.concatenate(pred_disps_list, axis=0)

    # ----------------------------
    # Sanity: pred count vs GT count
    # ----------------------------
    print(f"-> num_pred: {pred_disps.shape[0]} | num_gt: {len(gt_depths)}")
    assert pred_disps.shape[0] == len(gt_depths), \
        f"Mismatch: {pred_disps.shape[0]} predictions vs {len(gt_depths)} gt depth maps"

    # ----------------------------
    # Scaling mode
    # ----------------------------
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling")
        opt.disable_median_scaling = True
    else:
        print("   Mono evaluation - using median scaling")

    # ----------------------------
    # Evaluate (A-style: eval after)
    # ----------------------------
    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_valid = gt_depth[mask]

        # IMPORTANT:
        # A-logic: do NOT apply pred_depth_scale_factor here.
        # (If you want B behavior, uncomment next line)
        # pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_valid) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        err = compute_errors(gt_valid, pred_depth)
        errors.append(err)

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    errors = np.array(errors)
    mean_errors = np.mean(errors, axis=0)

    # Confidence intervals (keep B feature)
    cls = []
    for k in range(len(mean_errors)):
        cl = st.t.interval(alpha=0.95, df=len(errors) - 1,
                           loc=mean_errors[k], scale=st.sem(errors[:, k]))
        cls.append(cl[0])
        cls.append(cl[1])
    cls = np.array(cls)

    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
    if len(inference_times) > 0:
        print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times)) * 1000))
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()

    # Sensible default if your options don’t define model_type
    if not hasattr(opt, "model_type") or opt.model_type is None:
        opt.model_type = "endodac"  # change if needed

    evaluate(opt)
