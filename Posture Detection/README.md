# Posture Detection System

A comprehensive computer vision system for detecting and classifying student postures in classroom videos using YOLOv8 pose estimation and a custom PyTorch classifier.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Visualization](#visualization)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Pose Estimation**: Utilizes YOLOv8n-pose for accurate keypoint detection from video frames
- **Feature Engineering**: Extracts 14-dimensional feature vectors from pose keypoints including joint positions and angles
- **Posture Classification**: Multi-class classification for attentive, distracted, hand-raising, and writing postures
- **Temporal Smoothing**: Applies majority voting over a sliding window to reduce prediction noise
- **Real-time Processing**: Efficient inference with configurable frame sampling
- **Visualization**: Generates training curves, confusion matrices, and annotated output videos
- **Modular Design**: Clean separation of data extraction, training, and inference pipelines

## Project Structure

```
Posture Detection/
├── config.py                    # Configuration file with paths and hyperparameters
├── yolov8n-pose.pt             # YOLOv8 pose estimation model weights
├── data/
│   ├── datasets/
│   │   ├── X_train.npy         # Training feature vectors
│   │   └── y_train.npy         # Training labels
│   ├── keypoints/
│   │   ├── keypoints_data.json # Extracted pose keypoints
│   │   └── labels.json         # Manual posture labels
│   └── raw_videos/             # Input video files
├── models/
│   ├── posture_classifier.pth  # Trained PyTorch classifier
│   └── training_history.json   # Training metrics and history
├── output/
│   └── graphs/                 # Generated visualization graphs
└── src/
    ├── __init__.py
    ├── Dataset.py              # Dataset construction from keypoints and labels
    ├── FeatureExtraction.py    # Feature engineering from pose keypoints
    ├── Graphs.py               # Training visualization generation
    ├── Inference.py            # Real-time posture detection pipeline
    ├── Keypoints.py            # Keypoint extraction from video
    ├── Label.py                # Manual video annotation tool
    └── Train.py                # Model training script
```

## Requirements

### Hardware
- CPU or GPU with CUDA support (recommended for faster processing)
- Minimum 8GB RAM
- Storage: ~2GB for models and data

### Software
- Python 3.8+
- Windows/Linux/macOS
- FFmpeg (for video processing, usually pre-installed)

## Installation

### Step 1: Clone or Download the Project
Place the project folder in your desired location. For this example, we'll assume it's at:
```
C:\Users\[YourUsername]\Desktop\Posture-Detection\Posture Detection\
```

### Step 2: Install Python
Ensure Python 3.8 or higher is installed. Download from [python.org](https://python.org) if needed.

### Step 3: Set Up Virtual Environment (Recommended)
```bash
# Navigate to the project directory
cd "C:\Users\[YourUsername]\Desktop\Posture-Detection\Posture Detection"

# Create virtual environment
python -m venv posture_env

# Activate virtual environment
# On Windows:
posture_env\Scripts\activate
# On Linux/macOS:
# source posture_env/bin/activate
```

### Step 4: Install PyTorch
Install PyTorch with CUDA support if you have a GPU:
```bash
# For CUDA 11.8 (check your CUDA version with nvcc --version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only version:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install Core Dependencies
```bash
pip install ultralytics opencv-python numpy matplotlib seaborn
```

### Step 6: Verify Installation
```bash
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics installed')"
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

### Step 7: Download YOLO Model (if not present)
The project includes `yolov8n-pose.pt`, but if you need to download it manually:
```bash
# This will be handled automatically by the scripts, but you can pre-download:
# The model is included in the project files
```

## Usage

### Data Preparation Pipeline

#### Step 1: Prepare Input Video
Place your classroom video file at:
```
data/raw_videos/classroom_video.mp4
```

#### Step 2: Extract Keypoints
Run keypoint extraction from the video:
```bash
python src/Keypoints.py
```
This will:
- Load the YOLO pose model
- Process video frames (sampling every 3 frames by default)
- Save keypoints to `data/keypoints/keypoints_data.json`

#### Step 3: Label the Data
Manually annotate posture segments:
```bash
python src/Label.py
```
Controls:
- `A`: Start/Stop "attentive" segment
- `D`: Start/Stop "distracted" segment  
- `H`: Start/Stop "hand_raising" segment
- `W`: Start/Stop "writing" segment
- `SPACE`: Pause/Resume video
- `S`: Save current segments
- `Q`: Quit

Labels are saved to `data/keypoints/labels.json`

#### Step 4: Build Dataset
Create training data from keypoints and labels:
```bash
python src/Dataset.py
```
This generates:
- `data/datasets/X_train.npy`: Feature vectors
- `data/datasets/y_train.npy`: Labels

## Training

Train the posture classifier:
```bash
python src/Train.py
```
This will:
- Load the dataset
- Train for 50 epochs with validation
- Save the best model to `models/posture_classifier.pth`
- Save training history to `models/training_history.json`

## Inference

Run posture detection on a new video:
```bash
python src/Inference.py
```
Make sure to update `config.py` with the correct input video path. The output will be saved to `output/output_annotated.mp4`.

## Visualization

Generate training graphs and metrics:
```bash
python src/Graphs.py
```
This creates various plots in `output/graphs/` including:
- Training loss curves
- Validation accuracy
- Confusion matrices
- Per-class accuracy charts

## Configuration

All settings are centralized in `config.py`. Key parameters:

### Paths
- `INPUT_VIDEO`: Input video path
- `OUTPUT_VIDEO`: Annotated output path
- `KEYPOINTS_JSON`: Keypoints data file
- `LABELS_JSON`: Label annotations
- `CLASSIFIER_MODEL`: Trained model path

### Model Parameters
- `NUM_CLASSES`: 4 (attentive, distracted, hand-raising, writing)
- `FEATURE_SIZE`: 14 (extracted features per person)
- `EPOCHS`: 50
- `BATCH_SIZE`: 32
- `LEARNING_RATE`: 0.001

### Processing Settings
- `EXTRACT_EVERY_N_FRAMES`: 3 (keypoint extraction sampling)
- `INFERENCE_EVERY_N_FRAMES`: 2 (inference sampling)
- `SMOOTH_WINDOW`: 10 (temporal smoothing window)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `BATCH_SIZE` in `config.py`
2. **Video not found**: Ensure video is at `data/raw_videos/classroom_video.mp4`
3. **Model not found**: Run training first or check `models/posture_classifier.pth`
4. **Import errors**: Ensure all dependencies are installed in the virtual environment

### Performance Tips

- Use GPU for faster processing
- Adjust frame sampling rates for speed vs accuracy trade-off
- Increase `SMOOTH_WINDOW` for more stable predictions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For questions or support, please open an issue in the repository.