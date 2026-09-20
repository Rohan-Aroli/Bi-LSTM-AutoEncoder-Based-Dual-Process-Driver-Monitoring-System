# Bi-LSTM Dual-Process Driver Monitoring System

A real-time driver monitoring system that combines computer vision, facial landmark analysis, temporal feature processing, and a Bi-LSTM autoencoder to detect anomalous driver behavior associated with drowsiness.

The system is designed around a dual-process architecture:

1. A Vision Process captures and processes live camera frames.
2. An Inference Process performs temporal sequence reconstruction using a Bi-LSTM autoencoder.

The two processes communicate through ZeroMQ, allowing the vision and inference pipelines to operate independently.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Processing Pipeline](#processing-pipeline)
- [Machine Learning Model](#machine-learning-model)
- [Feature Representation](#feature-representation)
- [Calibration](#calibration)
- [Project Structure](#project-structure)
- [File Description](#file-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Weights](#model-weights)
- [Running the System](#running-the-system)
- [Runtime Behavior](#runtime-behavior)
- [Configuration](#configuration)
- [Generated Files](#generated-files)
- [Controls](#controls)
- [Technical Details](#technical-details)
- [Design Decisions](#design-decisions)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)
- [Git Workflow](#git-workflow)
- [Project Status](#project-status)
- [License](#license)

---

## Overview

The system monitors a driver's face through a webcam and converts facial observations into a temporal feature representation.

The vision pipeline extracts:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head pose information

The head rotation matrix is converted into a continuous 6D representation. Together with EAR and MAR, this produces an 8-dimensional feature vector for each frame.

The system maintains a rolling sequence of 60 frames:

```text
60 frames x 8 features
```

These sequences are sent to a Bi-LSTM autoencoder for reconstruction.

The reconstruction error is used as an anomaly score:

```text
Low reconstruction error
        |
        v
      NORMAL

High reconstruction error
        |
        v
      DROWSY
```

The system also performs user-specific calibration before live inference.

---

## System Architecture

The application uses two independent processes.

```text
                    +----------------------+
                    |       Webcam         |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |    Vision Process    |
                    |----------------------|
                    | OpenCV               |
                    | MediaPipe FaceMesh   |
                    | Feature Extraction   |
                    | Calibration          |
                    | Frame Buffering      |
                    +----------+-----------+
                               |
                         ZeroMQ PUSH
                               |
                               v
                    +----------------------+
                    |   Inference Process  |
                    |----------------------|
                    | PyTorch              |
                    | Bi-LSTM Autoencoder  |
                    | MSE Calculation      |
                    | Anomaly Detection    |
                    +----------+-----------+
                               |
                         ZeroMQ PUB
                               |
                               v
                    +----------------------+
                    |    Vision Process    |
                    |----------------------|
                    | Display State        |
                    | NORMAL / DROWSY      |
                    +----------------------+
```

The main process starts the inference process before starting the vision process so that the inference communication endpoint is ready before feature data is transmitted.

The inference process uses CUDA when a compatible GPU is available and otherwise falls back to CPU execution.

---

## Key Features

### Real-Time Face Tracking

MediaPipe Face Mesh is used to detect facial landmarks from the webcam stream.

The system processes one face at a time.

### Facial Feature Extraction

The vision pipeline extracts:

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head rotation

These features are converted into an 8-dimensional representation.

### Head Pose Representation

Six canonical facial points are used with OpenCV's pose-estimation pipeline to estimate head rotation.

The selected facial landmarks include:

- Nose
- Chin
- Left eye
- Right eye
- Left mouth
- Right mouth

The resulting rotation matrix is converted into a continuous 6D representation.

### User-Specific Calibration

The system establishes a baseline for the current driver before performing live inference.

The baseline contains:

- EAR baseline
- MAR baseline
- Head rotation baseline

The calibration data is stored locally in:

```text
calibration_anchor.npz
```

### Temporal Sequence Processing

Rather than evaluating individual frames independently, the system maintains a rolling 60-frame buffer.

```text
Frame 1   -> 8 features
Frame 2   -> 8 features
...
Frame 60  -> 8 features
```

This produces:

```text
(60, 8)
```

which is passed to the neural network.

### Bi-LSTM Autoencoder

The inference model contains:

```text
Bi-LSTM Encoder
        |
        v
Latent Temporal Representation
        |
        v
Bi-LSTM Decoder
        |
        v
8-D Reconstruction
```

The model uses:

- Input dimension: 8
- Hidden dimension: 32
- LSTM layers: 2
- Bidirectional encoder
- Bidirectional decoder

### MSE-Based Anomaly Detection

The model reconstructs the input sequence.

The reconstruction error is calculated using Mean Squared Error (MSE).

The resulting error is compared with an anomaly threshold.

```text
reconstruction_error <= threshold
            |
            v
          NORMAL


reconstruction_error > threshold
            |
            v
          DROWSY
```

### Missing Frame Handling

If the driver's face temporarily disappears, the system attempts to preserve temporal continuity.

The buffer manager handles missing frames using:

- Linear interpolation for EAR and MAR
- Spherical Linear Interpolation (SLERP) for head rotation

If the driver remains missing beyond the configured limit, the buffer is cleared and the system enters a hold state.

---

## Processing Pipeline

The complete runtime pipeline is:

```text
Webcam
  |
  v
OpenCV Frame Capture
  |
  v
MediaPipe Face Mesh
  |
  v
Facial Landmark Extraction
  |
  +--------------------+
  |                    |
  v                    v
EAR / MAR          Head Pose
  |                    |
  +---------+----------+
            |
            v
      8D Feature Vector
            |
            v
      Driver Calibration
            |
            v
      Rolling 60-Frame Buffer
            |
            v
        ZeroMQ PUSH
            |
            v
    Bi-LSTM Autoencoder
            |
            v
      Reconstruction
            |
            v
       MSE Calculation
            |
            v
      Anomaly Threshold
            |
       +----+----+
       |         |
       v         v
    NORMAL     DROWSY
       |         |
       +----+----+
            |
            v
       ZeroMQ PUB
            |
            v
      Vision Display
```

---

## Machine Learning Model

The neural network is implemented in:

```text
BiLSTM/Model.py
```

The model is a sequence-to-sequence Bi-LSTM autoencoder.

### Encoder

The encoder receives a temporal feature sequence:

```text
(batch_size, sequence_length, input_features)
```

For live inference:

```text
(1, 60, 8)
```

The encoder consists of a two-layer bidirectional LSTM.

```text
Input
  |
  v
Bi-LSTM Encoder
  |
  v
Temporal Representation
```

### Decoder

The decoder receives the encoded temporal representation and attempts to reconstruct the original sequence.

```text
Encoded Representation
        |
        v
    Bi-LSTM Decoder
        |
        v
      Linear
        |
        v
     8 Features
```

The final linear layer maps the decoder output back to the original eight-dimensional feature space.

---

## Feature Representation

Each frame is represented using eight values:

```text
[6D Head Rotation, EAR, MAR]
```

The six head-pose values are obtained from the relative rotation matrix.

The resulting feature vector contains:

```text
[Rx1, Rx2, Rx3,
 Ry1, Ry2, Ry3,
 EAR,
 MAR]
```

The same feature ordering is maintained throughout the buffering and inference pipeline.

---

## Calibration

Calibration occurs in multiple stages.

### Stage 1: Driver Baseline Calibration

The system collects valid facial observations during the initial calibration period.

The baseline contains:

```text
EAR baseline
MAR baseline
Rotation baseline
```

The baseline is saved as:

```text
calibration_anchor.npz
```

If this file already exists, the system can load the existing calibration rather than performing the initial calibration again.

### Stage 2: MSE Threshold Calibration

After the driver baseline has been established, the system collects normal driving feature vectors.

The inference process divides the collected sequence into overlapping 60-frame windows.

For each sequence, reconstruction error is calculated.

The threshold is derived statistically from the observed reconstruction errors.

The resulting threshold information is stored in:

```text
mse_threshold.json
```

This allows anomaly detection to use calibration-derived reconstruction-error statistics.

---

## Project Structure

```text
Bi-Lstm dual process architecture/
│
├── README.md
├── .gitignore
├── config.py
├── main.py
│
├── BiLSTM/
│   ├── Model.py
│   ├── inference_node.py
│   └── mse_calibrator.py
│
├── model/
│   └── .gitkeep
│
└── vision/
    ├── buffer_manager.py
    ├── calibration.py
    ├── feature_extractor.py
    ├── mse_calib.py
    └── vision_node.py
```

Runtime-generated files and trained model weights are intentionally excluded from version control.

---

## File Description

### `main.py`

Application entry point.

Responsible for:

- Starting the inference process
- Configuring multiprocessing
- Starting the vision process
- Handling shutdown
- Terminating the inference process when the application exits

The inference process is started before the vision pipeline so that ZeroMQ communication is ready before the vision process begins transmitting data.

---

### `config.py`

Central configuration file.

It contains configuration for:

- Camera
- FPS
- ZeroMQ ports
- Calibration
- Buffer length
- Missing-frame handling
- Facial landmark indices

Default communication ports:

```text
PUSH: 5555
SUB:  5556
```

---

### `BiLSTM/Model.py`

Defines the `BiLSTMAutoencoder`.

Responsibilities:

- Bi-LSTM encoder
- Bi-LSTM decoder
- Reconstruction layer
- Forward pass

---

### `BiLSTM/inference_node.py`

Runs the inference process.

Responsibilities:

- Initialize ZeroMQ
- Load the trained model
- Select CPU or CUDA
- Receive feature sequences
- Perform model inference
- Calculate reconstruction error
- Apply the anomaly threshold
- Publish the resulting state

The current inference states are:

```text
NORMAL
DROWSY
```

---

### `BiLSTM/mse_calibrator.py`

Calculates the reconstruction-error threshold.

Responsibilities:

- Receive calibration sequences
- Generate overlapping 60-frame windows
- Run batch inference
- Calculate reconstruction errors
- Calculate the statistical threshold
- Save threshold information to JSON

---

### `vision/vision_node.py`

Main vision process.

Responsibilities:

- Initialize the webcam
- Initialize MediaPipe Face Mesh
- Extract facial landmarks
- Run calibration
- Maintain the live feature pipeline
- Send feature buffers to the inference process
- Receive inference states
- Render the result on the camera feed

---

### `vision/feature_extractor.py`

Contains the feature extraction functions.

Responsibilities:

- Extract canonical facial landmarks
- Estimate head rotation
- Calculate EAR
- Calculate MAR
- Construct the raw facial feature representation

---

### `vision/calibration.py`

Handles driver-specific calibration.

Responsibilities:

- Load existing calibration
- Collect calibration observations
- Calculate baseline values
- Save calibration data
- Convert raw observations into the calibrated feature representation

---

### `vision/buffer_manager.py`

Maintains the rolling temporal feature buffer.

Responsibilities:

- Maintain the 60-frame buffer
- Detect missing frames
- Interpolate missing EAR/MAR values
- Interpolate head rotation using SLERP
- Return complete sequences to the inference process

---

### `vision/mse_calib.py`

Collects the normal feature sequence used during MSE threshold calibration.

The collector accumulates valid 8D feature vectors until the required number of frames is available.

---

## Requirements

The current implementation uses the following Python packages:

```text
numpy
opencv-python
mediapipe
pyzmq
torch
scipy
```

Python standard-library modules are also used, including:

```text
multiprocessing
time
sys
os
json
```

A CUDA-capable PyTorch installation can be used when a compatible NVIDIA GPU and CUDA environment are available.

---

## Installation

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Bi-LSTM-Dual-Process-Driver-Monitoring
```

### 2. Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy opencv-python mediapipe pyzmq scipy
```

Install PyTorch according to the target machine and CUDA configuration.

### 4. Verify the Environment

```bash
python -c "import numpy, cv2, mediapipe, zmq, torch, scipy; print('Environment OK')"
```

---

## Model Weights

The trained model weights are not included in the repository.

Place the trained model inside:

```text
model/
```

For example:

```text
model/
└── bilstm_autoencoder_Cleaned.pth
```

Model weight files are excluded from Git using `.gitignore`.

Before running the system on another machine, ensure that the model-loading path in:

```text
BiLSTM/inference_node.py
```

points to the local model file.

The current implementation contains a machine-specific model path. This should be changed to a project-relative or configuration-based path before distributing the project to other machines.

Recommended structure:

```text
project/
│
├── model/
│   └── bilstm_autoencoder_Cleaned.pth
│
└── BiLSTM/
    └── inference_node.py
```

---

## Running the System

After activating the virtual environment and placing the model weights in the appropriate location:

```bash
python main.py
```

The application starts the inference process first and then launches the vision pipeline.

Expected startup sequence:

```text
[Main] Booting Driver Monitoring System...
        |
        v
Inference Node starts
        |
        v
Vision Node starts
        |
        v
Camera initialization
        |
        v
Driver calibration
        |
        v
MSE threshold calibration
        |
        v
Live inference
```

---

## Runtime Behavior

### Initial State

The vision process initializes:

- Webcam
- MediaPipe Face Mesh
- ZeroMQ sockets
- Calibration manager
- Feature buffer

### Driver Calibration

If no calibration file exists, the system collects facial observations and creates:

```text
calibration_anchor.npz
```

### MSE Calibration

The system collects normal feature sequences and sends them to the inference process.

The inference process calculates reconstruction-error statistics and creates:

```text
mse_threshold.json
```

### Live Inference

Once calibration is complete:

```text
Camera
   |
   v
Face landmarks
   |
   v
8D feature vector
   |
   v
60-frame rolling buffer
   |
   v
Bi-LSTM autoencoder
   |
   v
MSE
   |
   v
NORMAL / DROWSY
```

---

## Configuration

Runtime settings are centralized in:

```text
config.py
```

Typical configuration includes:

```python
CAMERA_ID = 0
FPS = 30

ZMQ_PUSH_PORT = 5555
ZMQ_SUB_PORT = 5556

CALIBRATION_FILE = "calibration_anchor.npz"

MAX_BUFFER_LEN = 60
MAX_MISSING_FRAMES = 10
```

The camera ID can be changed if another camera device should be used.

For example:

```python
CAMERA_ID = 1
```

---

## Generated Files

The following files are generated during runtime.

### `calibration_anchor.npz`

Contains the driver-specific calibration baseline.

This file is ignored by Git because it is generated from a specific calibration session.

### `mse_threshold.json`

Contains the reconstruction-error statistics used for anomaly detection.

This file is also ignored by Git because it is generated during runtime.

---

## Controls

The camera interface currently supports:

```text
q
```

Exit the application.

```text
r
```

Force recalibration.

When `r` is pressed, the existing calibration state is reset and the system can perform calibration again.

---

## Technical Details

### ZeroMQ Communication

The system uses ZeroMQ for inter-process communication.

#### Vision -> Inference

The vision process uses a ZeroMQ `PUSH` socket.

Default endpoint:

```text
tcp://127.0.0.1:5555
```

The vision process sends:

```text
60-frame feature sequences
```

or calibration batches.

#### Inference -> Vision

The inference process uses a ZeroMQ `PUB` socket.

Default endpoint:

```text
tcp://127.0.0.1:5556
```

The inference process publishes the current state.

Example:

```json
{
    "state": "NORMAL"
}
```

or:

```json
{
    "state": "DROWSY"
}
```

This allows the vision process to receive inference results without directly performing the neural-network computation.

---

### Temporal Buffering

The system maintains a maximum buffer length of:

```text
60 frames
```

At approximately:

```text
30 FPS
```

this represents approximately two seconds of temporal information.

When the buffer becomes full, the sequence is sent to the inference process.

---

### Missing Frame Handling

When the driver's face is temporarily unavailable, the buffer manager attempts to preserve temporal continuity.

#### EAR and MAR

Linear interpolation is used for missing scalar feature values.

#### Head Rotation

Head rotation is represented using rotation matrices and interpolated using SLERP.

#### Extended Missing Frames

If the number of missing frames exceeds the configured limit, the buffer is cleared and the system enters a hold state.

This prevents excessively stale observations from being passed into the neural network.

---

## Design Decisions

### Why Two Processes?

The vision and neural-network pipelines have different computational responsibilities.

The vision process handles:

```text
Camera
MediaPipe
Feature extraction
Buffering
Display
```

The inference process handles:

```text
PyTorch
Model loading
GPU computation
MSE calculation
Anomaly classification
```

Separating these responsibilities prevents heavy inference computation from directly blocking the camera-processing pipeline.

### Why ZeroMQ?

ZeroMQ provides lightweight inter-process communication without requiring a separate message broker.

The current architecture uses:

```text
PUSH / PULL
```

for sending feature data to the inference process and:

```text
PUB / SUB
```

for sending inference results back to the vision process.

---

## Limitations

The current implementation has several limitations.

### Model Dependency

The system requires trained Bi-LSTM autoencoder weights.

The model weights are intentionally not included in the repository.

### Hardware Dependency

Real-time operation depends on the performance of the system running the application.

CUDA acceleration is used when available; otherwise inference falls back to CPU execution.

### Camera Dependency

The current implementation requires a camera accessible through OpenCV.

### Calibration Dependency

The anomaly threshold depends on the calibration process and the driver's observed normal behavior.

### Local Runtime Artifacts

Calibration data is generated locally and is not intended to be shared between different users or machines.

---

## Future Improvements

Potential improvements include:

- Replace the machine-specific model path with a project-relative configuration
- Add command-line configuration
- Add a formal configuration file
- Add automated unit and integration tests
- Add structured logging
- Add model versioning
- Add performance benchmarking
- Add inference latency measurements
- Add model evaluation metrics
- Add reconstruction-error visualization
- Add configurable alert thresholds
- Add persistent event logging
- Add deployment support
- Containerize the application
- Integrate hardware-based driver alerts
- Add a dashboard for monitoring driver state and system events

---

## Troubleshooting

### Camera Does Not Open

Check the configured camera ID:

```python
CAMERA_ID = 0
```

Try another camera index if multiple cameras are connected.

---

### Model Weights Cannot Be Loaded

Check that the trained model exists in:

```text
model/
└── bilstm_autoencoder_Cleaned.pth
```

Also check the model path configured in:

```text
BiLSTM/inference_node.py
```

---

### Inference Node Fails to Start

Check:

- PyTorch installation
- CUDA configuration
- Model weights
- ZeroMQ installation
- Python environment

The main process checks whether the inference process is running before starting the vision pipeline.

---

### Calibration Does Not Complete

Make sure:

- The camera is working.
- The driver's face is visible.
- MediaPipe is detecting facial landmarks.
- The driver remains within the camera's field of view.

If necessary, press:

```text
r
```

to force recalibration.

---

### ZeroMQ Communication Problems

Verify that the configured ports are available:

```text
5555
5556
```

The application uses localhost communication:

```text
127.0.0.1
```

---

## Git Workflow

The repository uses Git for version control.

A recommended development workflow is:

```text
main
 |
 +---- feature/calibration
 |
 +---- feature/inference
 |
 +---- feature/vision
 |
 +---- feature/dashboard
```

### Create a Feature Branch

```bash
git switch -c feature/<feature-name>
```

### Check Repository Status

```bash
git status
```

### Review Changes

```bash
git diff
```

### Stage Changes

```bash
git add .
```

### Commit Changes

```bash
git commit -m "Add <feature>"
```

### Push the Branch

```bash
git push -u origin feature/<feature-name>
```

After testing, merge completed features into `main`.

---

## Repository and Runtime Files

The repository intentionally excludes machine-specific and generated files such as:

```text
*.npz
*.npy
*.pth
*.pt
*.onnx

venv/
.venv/

__pycache__/

*.log
*.tmp

calibration_anchor.npz
mse_threshold.json
```

The `model/` directory is retained in the repository using `.gitkeep` so that users know where to place trained model weights.

---

## Project Status

Current implementation includes:

```text
[OK] Dual-process architecture
[OK] OpenCV camera pipeline
[OK] MediaPipe Face Mesh
[OK] EAR extraction
[OK] MAR extraction
[OK] Head-pose estimation
[OK] 8D feature representation
[OK] Driver calibration
[OK] 60-frame rolling buffer
[OK] Missing-frame interpolation
[OK] Bi-LSTM autoencoder
[OK] ZeroMQ inter-process communication
[OK] MSE-based anomaly detection
[OK] Runtime threshold calibration
[OK] NORMAL / DROWSY state output
```

