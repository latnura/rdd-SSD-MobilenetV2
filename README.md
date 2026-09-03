# Road Damage Detection - SSD Lite MobileNetV2

A real-time road damage detection system (linear cracks, alligator cracks, and potholes) built on the **SSD Lite MobileNetV2** architecture. This repo covers the full pipeline: dataset conversion, training, evaluation, video testing, and a real-time field detection system integrated with GPS and damage area estimation in square meters.

## Table of Contents

- [About the Project](#about-the-project)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Detected Damage Classes](#detected-damage-classes)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Testing on Video](#testing-on-video)
- [Real-time Detection System with GPS](#real-time-detection-system-with-gps)
- [Acknowledgement](#acknowledgement)
- [License](#license)

## About the Project

This project builds an object detection model for road surface damage based on **SSD Lite MobileNetV2**, trained on a merged Road Damage Dataset (VOC format). Besides standard training and evaluation scripts, the repo also includes a field detection system (`deteksiluaswithROI.py`) designed to run on edge devices (e.g. Jetson) that performs:

- Damage detection within a defined *Region of Interest* (ROI) in the video frame
- Damage area estimation (m²) based on camera calibration parameters
- GPS coordinate logging and automatic road segmentation (STA)
- Sending detection results (photo, damage type, area, location) to an external API

## Key Features

- **Detection of 3 road damage types** using SSD Lite MobileNetV2 (lightweight, suitable for edge devices)
- **Real-time damage area estimation** based on camera height, tilt angle, and vertical FOV
- **Simple centroid-based object tracking** to avoid double-counting the same object
- **GPS integration (NMEA over serial/USB)** for location logging and automatic road segment checkpoints
- **Automatic upload** of detection results (photo + metadata) to a server/API
- **Full training pipeline**: choice of optimizer (Adam/SGD/RMSprop), scheduler (plateau, multi-step, cosine), Weights & Biases (W&B) logging, and mAP/Precision/Recall/F1 tracking at every validation step
- **Full model evaluation** including mAP@0.5, mAP@0.5:0.95, GFLOPs, and parameter count
- **Offline video testing** with FPS and inference time measurement, results automatically saved to CSV

## Repository Structure

```
rdd-SSD-MobilenetV2/
├── vision/                        # SSD architecture, dataset loaders, loss, and utilities
├── models_scheduler_plateau/      # Default folder for storing training checkpoints
├── convert_rdd_to_voc.py          # Converts the raw dataset into Pascal VOC structure
├── train_ssd.py                   # Trains the SSD Lite MobileNetV2 model
├── eval_ssd.py                    # Evaluates the model (mAP, Precision, Recall, F1, GFLOPs)
├── ssd_test_video.py               # Tests the model on a video file + FPS measurement
├── deteksiluaswithROI.py          # Real-time detection system + GPS + area estimation + API upload
├── labels.txt                     # Class list (single line, comma-separated)
├── labels4.txt                    # Class list (one class per line, including BACKGROUND)
├── mb2-ssd-lite-mp-0_686.pth      # Pretrained base net weights for transfer learning
└── requirements.txt                # Python dependency list
```

## Detected Damage Classes

The model detects 3 types of road damage plus a background class. The dataset class codes are used internally, while more readable names are used in the real-time detection script:

| Index | Dataset Code | Damage Name |
|---|---|---|
| 0 | BACKGROUND | Not damage |
| 1 | L00 | Potholes |
| 2 | R02 | Linear Crack |
| 3 | R03 | Alligator Crack |

> Note: re-check this mapping if the class order in your trained model differs from `labels4.txt`.

## Installation

1. Clone this repo:

```bash
git clone https://github.com/latnura/rdd-SSD-MobilenetV2.git
cd rdd-SSD-MobilenetV2
```

2. Create a virtual environment (Python 3.8-3.10 recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

3. Install the dependencies. Since `requirements.txt` targets CUDA 12.1 builds, use PyTorch's extra index URL:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

If you are running on CPU only or a different CUDA version, adjust the `torch`, `torchvision`, and `nvidia-*` package versions in `requirements.txt` first.

## Dataset Preparation

The `convert_rdd_to_voc.py` script converts a raw dataset (mixed format with paired `.jpg` and `.xml` files, resulting from a merge) into the Pascal VOC structure required by `train_ssd.py`.

1. Place the raw dataset in the following folder (with `train`, `valid`, and `test` subfolders):

```
./DATASET_RAW/Merge-Harmoni-clean-with-Existing-clean.v1-merge-existing-harmoni-clean.voc/
├── train/
├── valid/
└── test/
```

2. Run the conversion:

```bash
python convert_rdd_to_voc.py --dataset_name VOCdevkit
```

The converted dataset will be saved to `./Dataset_testval/VOCdevkit/` with the following structure:

```
Dataset_testval/VOCdevkit/
├── VOC2007/            # Training data (train + valid)
└── test/VOC2007/       # Test data (test + valid)
```

## Training the Model

Basic training example (this exact command is printed automatically after the dataset conversion finishes):

```bash
python train_ssd.py \
  --datasets ./Dataset_testval/VOCdevkit/VOC2007 \
  --validation_dataset ./Dataset_testval/VOCdevkit/test/VOC2007 \
  --net mb2-ssd-lite \
  --base_net mb2-ssd-lite-mp-0_686.pth \
  --batch_size 16 \
  --num_epochs 120 \
  --optimizer adam \
  --lr 1e-4 \
  --scheduler plateau \
  --lr_patience 3 \
  --lr_factor 0.5 \
  --checkpoint_folder models_scheduler_plateau/
```

Key arguments:

| Argument | Description |
|---|---|
| `--net` | Backbone architecture (`mb2-ssd-lite`, `mb1-ssd`, `vgg16-ssd`, etc.) |
| `--base_net` / `--pretrained_ssd` / `--resume` | Source of initial weights (ImageNet pretrained, SSD pretrained, or resuming training) |
| `--scheduler` | `none`, `plateau`, `multi-step`, or `cosine` |
| `--optimizer` | `adam`, `momentum`, `rmsprop`, or `sgd` |
| `--validation_epochs` | Validation frequency (mAP, Precision, Recall, F1 are computed every N epochs) |
| `--results_file` | CSV filename for logging results at each validation step |

Training is integrated with **Weights & Biases**. Make sure you are logged in (`wandb login`) before running the script, or set `WANDB_MODE=offline` if you don't want to send logs to the cloud.

## Evaluating the Model

```bash
python eval_ssd.py \
  --net mb2-ssd-lite \
  --trained_model models_scheduler_plateau/mb2-ssd-lite-Epoch-XXX-Loss-X.XXXX.pth \
  --dataset ./Dataset_testval/VOCdevkit/test/VOC2007 \
  --label_file models_scheduler_plateau/voc-model-labels.txt \
  --eval_dir eval_results
```

This script produces:

- mAP@0.5 and mAP@0.5:0.95 (VOC2007 metric)
- Precision, Recall, and F1-Score per class at IoU 0.5
- Model complexity (GFLOPs and parameter count, computed with `thop` on a 512x512 input)

## Testing on Video

`ssd_test_video.py` runs the model on a video file, displays the detection results live, and saves the output video along with a performance summary (FPS, average inference time) to `inference_summary.csv`.

Before running, adjust the following variables inside the script:

```python
model_path = "./models_scheduler_plateau/<path_to_checkpoint>.pth"
label_path = "labels4.txt"
video_path = "<path_to_input_video>"
```

```bash
python ssd_test_video.py
```

Press `q` or `Esc` to stop early. The output video will be saved in the `videos/` folder.

## Real-time Detection System with GPS

`deteksiluaswithROI.py` is a field system that combines damage detection, area estimation, GPS logging, and API upload into a single program. It is designed to run on an edge device connected to a camera and a GPS module (NMEA-compatible, e.g. a u-blox module via serial/USB).

### Configuration

All parameters are set in the `CONFIG` dictionary at the top of the script:

- **`app`**: input source (live camera or video file), frame resolution, cooldown between captures
- **`roi`**: upper and lower bounds of the detection zone (in pixels)
- **`camera_params`**: camera height (`H_meter`), tilt angle (`theta_deg`), vertical FOV (`alpha_vfov_deg`), and focal calibration factor (`f_calib`) used for area estimation in m²
- **`model`**: model path, class name list, video/camera input path, and confidence threshold
- **`api`**: endpoint URL and API key for sending detection results
- **`gps`**: serial port and baud rate for the GPS module
- **`checkpoint`**: distance (meters) used to split the road into segments (STA)

Before running, adjust at least the following:

```python
CONFIG["model"]["path"] = "./models/mb2-ssd-lite.pth"
CONFIG["api"]["url"] = "https://your-domain.com/upload.php"
CONFIG["api"]["key"] = "YOUR_SECRET_API_KEY"
CONFIG["gps"]["port"] = "/dev/ttyACM0"   # adjust to your GPS port
```

### Running

```bash
python deteksiluaswithROI.py
```

The program will prompt for the road name and survey KM, then start reading GPS data, running detection on every frame, calculating the area of damage found within the ROI, and sending data (photo + metadata) to the API whenever a new piece of damage is detected. GPS coordinate logs are saved automatically to `deteksi_kerusakan/log_gps_aktif.csv`.

## Acknowledgement

The SSD/SSD-Lite architecture in the `vision/` folder is adapted from the [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) implementation by qfgaohao (MIT licensed), with the following modifications:

- **Input resolution changed from the default 300x300 to 512x512** in the `vision/` configuration, with prior boxes and layer sizes adjusted accordingly.
- Additional scheduler support (plateau, multi-step, cosine), Weights & Biases logging, and mAP/Precision/Recall/F1 computation added to the training loop.
- Addition of a field detection script integrated with GPS and damage area estimation.

## License

This project is licensed under the **MIT License**, see the [LICENSE](LICENSE) file for details. Portions of the code (the `vision/` folder, and parts of `train_ssd.py` and `eval_ssd.py`) are adapted from [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) by Hao Gao, also MIT licensed. The original copyright notice is retained as required by that license.
