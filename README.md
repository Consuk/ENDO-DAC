# [MICCAI'2024] EndoDAC: Efficient Adapting Foundation Model for Self-Supervised Depth Estimation from Any Endoscopic Camera

![Image](https://github.com/BeileiCui/EndoDAC/blob/main/assets/main.jpg)

### [__[arxiv]__](http://arxiv.org/abs/2405.08672)

* 2024-05-14 Our paper has been early accepted (top 11%) by MICCAI 2024!
* 2024-05-15 arxiv version is online.

## Abstract
Depth estimation plays a crucial role in various tasks within endoscopic surgery, including navigation, surface reconstruction, and augmented reality visualization. Despite the significant achievements of foundation models in vision tasks, including depth estimation, their direct application to the medical domain often results in suboptimal performance. This highlights the need for efficient adaptation methods to adapt these models to endoscopic depth estimation. We propose Endoscopic Depth Any Camera (EndoDAC) which is an efficient self-supervised depth estimation framework that adapts foundation models to endoscopic scenes. Specifically, we develop the Dynamic Vector-Based Low-Rank Adaptation (DV-LoRA) and employ Convolutional Neck blocks to tailor the foundational model to the surgical domain, utilizing remarkably few trainable parameters. Given that camera information is not always accessible, we also introduce a self-supervised adaptation strategy that estimates camera intrinsics using the pose encoder. Our framework is capable of being trained solely on monocular surgical videos from any camera, ensuring minimal training costs. Experiments demonstrate that our approach obtains superior performance even with fewer training epochs and unaware of the ground truth camera intrinsic.

## Results

| Method | Year | Abs Rel | Sq Rel | RMSE | RMSE log | &delta; | Checkpoint| 
|  :----:  | :----:  | :----:   |  :----:  | :----:  | :----:  | :----:  | :----:  |
| Fang et al. | 2020 | 0.078 |	0.794 |	6.794 |	0.109 |	0.946 |- |
| Endo-SfM | 2021 | 0.062 |	0.606 |	5.726 |	0.093 |	0.957 |- |
| AF-SfMLeaner | 2022 | 0.059 |	0.435 |	4.925 |	0.082 |	0.974 |- |
| Yang et al. | 2024 | 0.062 |	0.558 |	5.585 |	0.090 |	0.962 |- |
|__EndoDAC (Ours)__ | | __0.052__ |	__0.362__ |	__4.464__ |	__0.073__ |	__0.979__ | [google_drive](https://drive.google.com/file/d/1qzAYBtwYJDN7hEi6pApqBOOz6pUhyY70/view?usp=drive_link) |

## Initialization

Create an environment with conda:
```
conda env create -f conda.yaml
conda activate endodac
```

Install required dependencies with pip:
```
pip install -r requirements.txt
```

Download pretrained model from: [depth_anything_vitb14](https://drive.google.com/file/d/163ILZcnz_-IUoIgy1UF_r7PAQBqgDbll/view?usp=sharing). Create a folder named ```pretrained_model``` in this repo and place the downloaded model in it.

## Dataset
### SCARED
Please follow [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) to prepare the SCARED dataset.

### C3VD
This repo now supports C3VD directly for training/evaluation.

Expected structure (your screenshots already match this):
```
<C3VD_ROOT>/
  training/<sequence>/*_color.png
  validation/<sequence>/*_color.png
  testing/<sequence>/*_color.png, *_depth.tiff
```

Notes:
- C3VD depth is decoded as `depth_mm = raw_uint16 * (100/65535)`.
- C3VD depth GT is along camera z-axis and clamped to `0-100 mm` by the dataset.
- C3VD uses an omnidirectional camera model; for EndoDAC training we recommend `--learn_intrinsics True`.
- If split files under `splits/c3vd/` are missing, they are auto-generated from the folder structure above.
- Optional fixed intrinsics can be passed with `--c3vd_intrinsics_path` and used when `--learn_intrinsics False`.

Generate split files explicitly:
```
python generate_c3vd_splits.py --data_path <C3VD_ROOT>
```

## Utilization

### Training
```
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py --data_path <your_data_path> --log_dir './logs'
```

Example for C3VD:
```
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py \
  --dataset c3vd --split c3vd --eval_split c3vd \
  --data_path <C3VD_ROOT> --log_dir ./logs --learn_intrinsics True
```

### Evaluation

Export ground truth depth and pose before evaluation:
```
CUDA_VISIBLE_DEVICES=0 python export_gt_depth.py --data_path <your_data_path> --split endovis
python export_gt_pose.py --data_path <your_data_path> --split endovis --sequence sequence2
python export_gt_pose.py --data_path <your_data_path> --split endovis --sequence sequence1
```

Export C3VD depth GT:
```
python export_gt_depth.py --data_path <C3VD_ROOT> --split c3vd --useage eval
```

Assume to evaluate the epoch 19 weights of a __depth estimation model__ named ```endodac```:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> \
--load_weights_folder './logs/endodac/models/weights_19' --eval_mono
```

Evaluate on C3VD:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
  --dataset c3vd --eval_split c3vd --data_path <C3VD_ROOT> \
  --load_weights_folder './logs/endodac/models/weights_19' --eval_mono \
  --min_depth 0.1 --max_depth 150.0 \
  --c3vd_eval_min_depth 0.001 --c3vd_eval_max_depth 100.0
```

Assume to evaluate the epoch 19 weights of a __pose and intrinsic estimation model__ named ```endodac```:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_pose.py --data_path <your_data_path> \
--load_weights_folder './logs/endodac/models/weights_19' --eval_mono
```

## Acknowledgment
Our code is based on the implementation of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner), [Depth-Anything](https://github.com/LiheYoung/Depth-Anything). We thank their excellent works.
