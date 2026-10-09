import os
from pathlib import Path
import numpy as np

# Universal Base Directory (relative to this file)
BASE_DIR = Path(__file__).resolve().parent

# File Paths (Universal & Cross-Platform)
MODEL_PATH = str(BASE_DIR / "model" / "bilstm_autoencoder_Cleaned.pth")
CALIBRATION_FILE = str(BASE_DIR / "calibration_anchor.npz")
MSE_THRESHOLD_FILE = str(BASE_DIR / "mse_threshold.json")

# System Settings
CAMERA_PORTS = {"left": 1, "center": 0, "right": 2}
ACTIVE_CAMERA = "center"
YAW_THRESHOLDS = {"left": -30.0, "right": 30.0}
YAW_TREND_THRESHOLD = 20.0
YAW_RETURN_THRESHOLD = 20.0

FPS = 30
ZMQ_PUSH_PORT = 5555
ZMQ_SUB_PORT = 5556
MSE_CALIBRATION_SECONDS = 10  # Duration for MSE calibration (in seconds)

# Buffer & Interpolation Limits
MAX_BUFFER_LEN = 60
MAX_MISSING_FRAMES = 10
CALIBRATION_FRAMES = 900 # 30 seconds at 30 FPS

# Canonical 3D Facial Points (in OpenCV coordinate space)
# Order: Nose tip, Chin, Left Eye, Right Eye, Left Mouth, Right Mouth
CANONICAL_FACE_POINTS = np.array([
    [0.0, 0.0, 0.0],          # Nose tip
    [0.0, -330.0, -65.0],     # Chin
    [-225.0, 170.0, -135.0],  # Left eye left corner
    [225.0, 170.0, -135.0],   # Right eye right corner
    [-150.0, -150.0, -125.0], # Left Mouth corner
    [150.0, -150.0, -125.0]   # Right mouth corner
], dtype=np.float64)