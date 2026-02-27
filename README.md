# 🚗 Real-Time Object Detection for Autonomous Vehicles

> A real-time, edge-optimized deep learning perception module for autonomous driving systems.
> Developed under the Digital Egypt Pioneers Initiative (DEPI) – AI & Machine Learning Track (2025).

## 📌 Project Overview

Autonomous vehicles require **low-latency, high-accuracy environmental perception** to ensure safe navigation.

This project presents a **YOLOv8-based real-time object detection system** optimized for edge deployment using **ONNX and TensorRT**, achieving:

* ⚡ **78.5 ms inference latency**
* 🎯 **71.1% mAP@50**
* 🚘 Multi-class detection for driving-critical objects
* 🧠 Full MLOps monitoring pipeline

## 🧠 Architecture
<img width="2471" height="164" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/72ab2aa1-d928-48d3-a072-df76d71d0076" />

## 🎯 Problem Statement

Autonomous systems often struggle with:

* Low-light and night driving
* Fog, rain, and weather interference
* Small object detection (traffic lights, distant pedestrians)
* Real-time latency constraints

We built a perception module that balances:

* ✅ Accuracy
* ✅ Speed
* ✅ Edge deployability
* ✅ Safety-critical robustness

## 🧠 Model Architecture

We selected **YOLOv8** due to:

* Anchor-free detection head
* C2f backbone blocks
* Efficient gradient flow
* Native ONNX & TensorRT export
* Strong small-object detection performance

### Detection Pipeline

```
Live Camera Feed
        ↓
Frame Preprocessing (Resize 640x640)
        ↓
YOLOv8 Inference
        ↓
NMS & Post-processing
        ↓
Structured Detection Output
```

## 📊 Dataset Strategy

### 📦 Datasets Used

* KITTI (Autonomous Driving Focused)
* COCO (Transfer Learning Base)
* OpenImages (Edge Cases & Diversity)

### 🔄 Data Engineering

* Multi-dataset fusion
* Class balancing
* YOLO format conversion
* Scene-leakage prevention
* Annotation normalization

### 🌧️ Custom Augmentation Pipeline

* Night simulation
* Fog & rain injection
* Motion blur
* Mosaic augmentation
* Brightness/contrast shifts

## 🏋️ Training Configuration

| Parameter       | Value   |
| --------------- | ------- |
| Model           | YOLOv8  |
| Image Size      | 512     |
| Batch Size      | 16      |
| Optimizer       | AdamW   |
| LR              | 0.0032  |
| Epochs          | 80      |
| Mixed Precision | Enabled |

### 🔬 Experiments Conducted

* Baseline
* Augmentation testing
* Hyperparameter tuning (Optuna)
* Image size benchmarking
* Fine-tuning strategy

Tracked with **MLflow**.

## 📈 Performance Results

| Metric               | Result                             |
| -------------------- | ---------------------------------- |
| Inference Latency    | **78.5 ms**                        |
| mAP@50               | **71.1%**                          |
| FPS (Edge Optimized) | 35–45 FPS                          |
| Target Latency       | < 200 ms (Exceeded by 2.5× faster) |

### Key Observations

* +12.7% accuracy gain after extended tuning
* Strong generalization in urban scenarios
* Remaining challenge: small traffic light detection

## ⚙️ Deployment & Optimization

### Model Export

* PyTorch (.pt)
* ONNX (.onnx)
* TensorRT Engine (.engine)

### Edge Optimization

* FP16 quantization
* Threaded inference pipeline
* Hardware-aware tuning
* Jetson validation

## 🔄 MLOps Integration

Built full monitoring pipeline:

* Model versioning (MLflow)
* Dataset versioning (DVC)
* Drift detection
* FPS & latency monitoring
* Automated retraining triggers
* Confidence distribution tracking

## 🏗 Project Structure

```
├── data/
├── notebooks/
├── src/
│   ├── training/
│   ├── inference/
│   ├── deployment/
│   └── mlops/
├── models/
├── experiments/
├── exports/
│   ├── onnx/
│   ├── tensorrt/
├── demo/
└── README.md
```

## 🎥 Live Demo

📺 YouTube Demo: https://youtu.be/UQaHwTuYU6g 


## 🚀 Future Work

* Upgrade to YOLOv8-Medium
* Improve small-object detection
* INT8 Quantization
* Multi-sensor fusion (LiDAR + Camera)
* Real vehicle hardware integration

## 🧪 Tech Stack

* Python
* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* ONNX
* TensorRT
* MLflow
* DVC
* Optuna

## 👨‍💻 Team

Developed as part of Digital Egyption Pioneers Initiative AI Cohort 2025:

* Elsayed Ali Elsayed Aboulila
* Nizar Hossam Hussein
* Abd El-Rahman Ahmed
* Ahmed Ashraf Abbas
* Mohamed Ashraf Mohamed

Supervised by:
Dr. Sherif Salem

## 🏁 Conclusion

This project goes beyond model training - it delivers a **deployable, monitored, edge-ready perception module** designed for safety-critical autonomous systems.

It demonstrates full-stack ML engineering:
From data engineering → model optimization → deployment → monitoring.
