# Real-Time Road Damage Detection & Spatial Measurement System

This repository contains a comprehensive pipeline for a real-time road damage detection system using the SSD (Single Shot MultiBox Detector) architecture with a MobileNetV2 backbone. 

Designed for both high accuracy and edge-device efficiency, this system not only detects road damages (e.g., potholes, linear cracks, alligator cracks) but also integrates GPS tracking and spatial transformation algorithms to measure the physical area of the damage in real-world dimensions (square meters).

## 🌟 Key Features
- **High-Resolution Inference:** Optimized for 640x640 input resolution to capture fine-grained road crack details.
- **Spatial Transformation:** Converts 2D pixel bounding boxes into real-world physical area measurements using camera intrinsic and extrinsic parameters (Height, Tilt Angle, Field of View).
- **Automated Experiment Tracking:** Integrated with Weights & Biases (W&B) and features Bayesian Optimization for hyperparameter tuning.
- **Hardware Integration:** Real-time GPS coordinate logging via serial communication (NMEA parsing) synced with video inference.
- **Smart Object Tracking:** Centroid-based distance tracking within a defined Region of Interest (ROI) to prevent duplicate counting across continuous frames.

## 🛠️ Environment Setup
This project is developed and tested on an **Ubuntu Linux** environment with CUDA support.

1. Clone this repository:
   ```bash
   git clone [https://github.com/latnura/rdd-SSD-MobilenetV2.git](https://github.com/latnura/rdd-SSD-MobilenetV2.git)
   cd rdd-SSD-MobilenetV2

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
