from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import scipy.stats as st

from utils.layers import disp_to_depth
from utils.utils import readlines, compute_errors
from options import MonodepthOptions
from datasets.scared_dataset import SCAREDRAWDataset
from datasets.hamlyn_dataset import HamlynDataset
import datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def render_depth(disp):
    disp = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp = disp.astype(np.uint8)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_color


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def evaluate(opt):
    """
    Evaluation B file, but using Evaluation A logic:
    - build dataset from splits/<split>/test_files.txt
    - load gt_depths from splits/<split>/gt_depths.npz (object array supported)
    - run inference -> collect pred_disps -> evaluate after
    """
    import wandb
    import matplotlib.pyplot as plt

    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    # ----------------------------
    # A-style: init wandb (optional)
    # ----------------------------
    if getattr(opt, "wandb", False):
        wandb.init(project=getattr(opt, "wandb_project", "iilDepth-Testing"))

    _DEPTH_COLORMAP = plt.get_cmap('plasma', 256)

    def colormap(inputs, normalize=True):
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()
        vis = inputs
        if normalize:
            ma = float(vis.max())
            mi = float(vis.min())
            d = ma - mi if ma != mi else 1e5
            vis = (vis - mi) / d

        if vis.ndim == 2:
            vis = _DEPTH_COLORMAP(vis)[..., :3]  # H,W,3
        return vis

    # ----------------------------
    # A-style: load weights / set depther
    # ----------------------------
    if opt.ext_disp_to_eval is None:
        if not opt.model_type == 'depthanything':
            opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
            assert os.path.isdir(opt.load_weights_folder), \
                "Cannot find a folder at {}".format(opt.load_weights_folder)
            print("-> Loading weights from {}".format(opt.load_weights_folder))
        else:
            print("Evaluating Depth Anything model")

        if opt.model_type == 'endodac':
            depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
            depther_dict = torch.load(depther_path)
        elif opt.model_type == 'afsfm':
            encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
            encoder_dict = torch.load(encoder_path)

        # ----------------------------
        # A-style: filenames from test_files.txt
        # ----------------------------
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        # A-style dataset build (Hamlyn requires filenames list)
        if opt.eval_split == 'endovis':
            dataset = SCAREDRAWDataset(
                opt.data_path, filenames,
                opt.height, opt.width,
                [0], 4, is_train=False
            )
        elif opt.eval_split == 'hamlyn':
            dataset = HamlynDataset(
                opt.data_path,
                opt.height, opt.width,
                [0], 4,
                is_train=False
            )

        elif opt.eval_split == 'c3vd':
            dataset = C3VDDataset(
                opt.data_path, filenames,
                opt.height, opt.width,
                [0], 4, is_train=False
            )
            MAX_DEPTH = 100
        else:
            raise ValueError(f"Unknown eval_split: {opt.eval_split}")

        # A-style batching (faster than B’s batch=1)
        dataloader = DataLoader(
            dataset,
            batch_size=getattr(opt, "eval_batch_size", 16),
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=False
        )

        if opt.model_type == 'endodac':
            depther = endodac.endodac(
                backbone_size="base",
                r=opt.lora_rank, lora_type=opt.lora_type,
                image_shape=(224, 280),
                pretrained_path=opt.pretrained_path,
                residual_block_indexes=opt.residual_block_indexes,
                include_cls_token=opt.include_cls_token
            )
            model_dict = depther.state_dict()
            depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
            depther.cuda().eval()

        elif opt.model_type == 'afsfm':
            encoder = encoders.ResnetEncoder(opt.num_layers, False)
            depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path))
            encoder.cuda().eval()
            depth_decoder.cuda().eval()

            def depther(image):
                return depth_decoder(encoder(image))

    else:
        # A-style: load predicted disparities directly
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        # Still need dataloader only for image ordering sanity (optional)
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        if opt.eval_split == 'endovis':
            dataset = SCAREDRAWDataset(opt.data_path, filenames, opt.height, opt.width, [0], 4, is_train=False)
        elif opt.eval_split == 'hamlyn':
            dataset = HamlynDataset(opt.data_path, filenames, opt.height, opt.width, [0], 4, is_train=False)
        elif opt.eval_split == 'c3vd':
            dataset = C3VDDataset(opt.data_path, filenames, opt.height, opt.width, [0], 4, is_train=False)
            MAX_DEPTH = 100
        else:
            raise ValueError(f"Unknown eval_split: {opt.eval_split}")

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=False)

    # ----------------------------
    # A-style: compute predictions first (unless ext_disp_to_eval)
    # ----------------------------
    if opt.ext_disp_to_eval is None:
        pred_disps_list = []
        print("-> Computing predictions with size {}x{}".format(opt.width, opt.height))

        with torch.no_grad():
            for step_i, data in enumerate(dataloader):
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = depther(input_color)

                # A/B compatibility: pick correct key
                if isinstance(output, dict) and ("disp", 0) in output:
                    output_disp = output[("disp", 0)]
                elif isinstance(output, dict) and ("disp", 0) in output:
                    output_disp = output[("disp", 0)]
                else:
                    # If your model returns something different, adjust here
                    output_disp = output[("disp", 0)]

                pred_disp, _ = disp_to_depth(output_disp, opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()  # (B,H,W)

                # NOTE: A had post_process code commented out; keep same behavior:
                # (do nothing even if post_process True)

                pred_disps_list.append(pred_disp)

        pred_disps = np.concatenate(pred_disps_list, axis=0)  # (N,H,W)

    # ----------------------------
    # A-style: load gt_depths from NPZ (CRITICAL for Hamlyn A-logic)
    # ----------------------------
    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    data_npz = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)
    gt_depths = data_npz["data"]

    # Hamlyn commonly uses object arrays (variable shapes)
    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)

    print("-> Loaded gt_depths. num_gt =", len(gt_depths), "num_pred =", pred_disps.shape[0])
    assert pred_disps.shape[0] == len(gt_depths), \
        f"Mismatch: {pred_disps.shape[0]} predictions vs {len(gt_depths)} gt depth maps"

    # ----------------------------
    # A-style scaling rules
    # ----------------------------
    if opt.eval_stereo:
        print("   Stereo evaluation - disabling median scaling")
        opt.disable_median_scaling = True
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    # ----------------------------
    # A-style: evaluate after predictions
    # ----------------------------
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]

        # A-style: optional W&B disparity visualization
        if getattr(opt, "wandb", False):
            disp_vis = colormap(pred_disp, normalize=True)  # H,W,3 float
            wandb.log({"disp_testing": wandb.Image((disp_vis * 255).astype(np.uint8))}, step=i)

        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1.0 / pred_disp

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth_valid = gt_depth[mask]

        # NOTE: A did NOT apply pred_depth_scale_factor here.
        # If you want it identical to A, leave it out.
        # If you want B behavior, uncomment next line:
        # pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth_valid) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth_valid, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)
    print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())