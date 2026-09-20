import cv2
import zmq
import numpy as np
import time
import mediapipe as mp

import config
from vision.feature_extractor import get_2d_landmarks, get_rotation_matrix, calculate_aspect_ratios
from vision.calibration import CalibrationManager
from vision.buffer_manager import BufferManager
from vision.mse_calib import MSEDataCollector

mse_collector = MSEDataCollector(fps=config.FPS, seconds=config.MSE_CALIBRATION_SECONDS)

def run_vision_node():
    print("[Vision Node] Initializing ZeroMQ sockets...")
    context = zmq.Context()
    
    # Forward Channel: Push 60-frame buffers to the GPU
    push_sock = context.socket(zmq.PUSH)
    push_sock.connect(f"tcp://127.0.0.1:{config.ZMQ_PUSH_PORT}")
    
    # Reverse Channel: Subscribe to model inference updates
    sub_sock = context.socket(zmq.SUB)
    sub_sock.connect(f"tcp://127.0.0.1:{config.ZMQ_SUB_PORT}")
    sub_sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sub_sock.setsockopt(zmq.CONFLATE, 1) # Drop stale states, keep only the newest

    # Initialize State Managers
    calibrator = CalibrationManager()
    buffer_mgr = BufferManager()

    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(config.CAMERA_ID)
    
    # UI State Variables
    display_text = "INITIALIZING..."
    display_color = (0, 255, 255) # Yellow

    print("[Vision Node] Starting camera loop...")

    MSE_State=False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[Vision Node] Camera feed dropped.")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        raw_r, raw_ear, raw_mar = None, None, None

        # 1. Feature Extraction (if face is found)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            pts_2d = get_2d_landmarks(landmarks, w, h)
            
            raw_r = get_rotation_matrix(pts_2d, w, h)
            raw_ear, raw_mar = calculate_aspect_ratios(landmarks, w, h)

        # 2. Pipeline Routing
        if not calibrator.is_calibrated():
            # STATE: CALIBRATION (Ignition Window)
            display_text = f"CALIBRATING... ({len(calibrator.calibration_buffer)}/{calibrator.required_frames})"
            display_color = (0, 255, 255)
            
            # Pass raw features (will ignore if raw_r is None)
            calibrator.update_calibration_buffer(raw_ear, raw_mar, raw_r)
        
        elif calibrator.is_calibrated() and not MSE_State:
            # STATE 2: MSE THRESHOLD GATHERING (10 Seconds)
            display_text = f"GATHERING MSE... ({len(mse_collector.buffer)}/{mse_collector.required_frames})"
            display_color = (255, 165, 0) # Orange
            
            if raw_r is not None:
                vector_8d = calibrator.transform_to_8d(raw_ear, raw_mar, raw_r)
                is_ready = mse_collector.add_frame(vector_8d)
                
                if is_ready:
                    # Fire the 300-frame batch down the pipe with a specific instruction
                    payload = {
                        "type": "CALIBRATE_MSE", 
                        "buffer": mse_collector.get_batch()
                    }
                    push_sock.send_pyobj(payload)
                    MSE_State = True
                    display_text = "CALCULATING THRESHOLD..."
                    print("[Vision Node] Sent 300 frames for MSE calibration.")


        else:
            # STATE: LIVE INFERENCE
            vector_8d = None
            if raw_r is not None:
                # Transform to relative 8D vector using the baseline anchor
                vector_8d = calibrator.transform_to_8d(raw_ear, raw_mar, raw_r)

            # Update rolling buffer (handles None for missing frames)
            status, payload = buffer_mgr.update(vector_8d)

            if status == "HOLD":
                display_text = "DRIVER MISSING / HOLD"
                display_color = (0, 0, 255) # Red
            elif status == "READY":
                # Fire the buffer down the ZeroMQ pipe to the GPU
                try:
                    push_sock.send_pyobj({"buffer": payload}, flags=zmq.NOBLOCK)
                except zmq.Again:
                    # Failsafe if the pipe gets jammed
                    print("[WARNING]: Pipeline is choking, Dropped the buffer!")
                    pass

        # 3. Asynchronous UI Update
        try:
            # Check for a new prediction from the GPU without blocking the camera
            latest_result = sub_sock.recv_pyobj(flags=zmq.NOBLOCK)
            
            # Update UI based on GPU output
            if latest_result["state"] == "NORMAL":
                display_text = "AWAKE"
                display_color = (0, 255, 0) # Green
            else:
                display_text = "DROWSY ANOMALY DETECTED"
                display_color = (0, 0, 255) # Red
                
        except zmq.Again:
            # No new message from GPU. Keep rendering the existing display_text.
            pass

        # 4. Rendering
        cv2.putText(frame, display_text, (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 3)
        cv2.imshow('Bi-LSTM Driver Monitoring System', frame)

        # Press 'q' to exit or 'r' to force recalibration
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[Vision Node] Forcing recalibration...")
            import os
            if os.path.exists(config.CALIBRATION_FILE):
                os.remove(config.CALIBRATION_FILE)
            calibrator = CalibrationManager() # Reinitialize
            buffer_mgr = BufferManager()

    cap.release()
    cv2.destroyAllWindows()
    push_sock.close()
    sub_sock.close()
    context.term()

if __name__ == '__main__':
    run_vision_node()