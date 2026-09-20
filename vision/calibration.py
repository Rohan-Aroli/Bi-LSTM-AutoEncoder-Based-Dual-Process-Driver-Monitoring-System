import os
import numpy as np
import config

class CalibrationManager:
    def __init__(self):
        self.filepath = config.CALIBRATION_FILE
        self.baselines = self.load_calibration()
        self.calibration_buffer = []
        self.required_frames = 900 # 3-second ignition window

    def is_calibrated(self):
        """Returns True if baselines are loaded and ready."""
        return self.baselines is not None

    def load_calibration(self):
        """Loads the anchor file if it exists, bypassing the ignition window."""
        if os.path.exists(self.filepath):
            data = np.load(self.filepath)
            print(f"[Calibration] Loaded existing baseline from {self.filepath}")
            return {
                "EAR": float(data['EAR']),
                "MAR": float(data['MAR']),
                "R": data['R']
            }
        return None

    def save_calibration(self):
        """Saves the calculated baselines to a compressed NumPy file."""
        np.savez(
            self.filepath, 
            EAR=self.baselines["EAR"], 
            MAR=self.baselines["MAR"], 
            R=self.baselines["R"]
        )
        print(f"[Calibration] Saved new baseline to {self.filepath}")

    def update_calibration_buffer(self, raw_ear, raw_mar, raw_r):
        """
        Collects strictly VALID frames until the 90-frame limit is reached.
        Ignores missing frames to ensure a pure baseline.
        """
        # Do not interpolate. Only append if we have a valid rotation matrix.
        if raw_r is None:
            return False

        self.calibration_buffer.append({
            "EAR": raw_ear,
            "MAR": raw_mar,
            "R": raw_r
        })

        if len(self.calibration_buffer) >= self.required_frames:
            self._compute_baselines()
            self.save_calibration()
            # self.calibration_buffer = [] # Clear memory
            return True
            
        return False

    def _orthogonalize_matrix(self, M):
        """
        Snaps a distorted matrix back into a mathematically pure 
        3x3 rotation matrix using SVD.
        """
        U, _, Vt = np.linalg.svd(M)
        R = np.dot(U, Vt)
        
        # Prevent reflection artifact
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(U, Vt)
            
        return R

    def _compute_baselines(self):
        """Calculates element-wise medians and applies SVD orthogonalization."""
        ears = [frame["EAR"] for frame in self.calibration_buffer]
        mars = [frame["MAR"] for frame in self.calibration_buffer]
        matrices = [frame["R"] for frame in self.calibration_buffer]

        ear_median = np.median(ears)
        mar_median = np.median(mars)
        
        # 1. Element-wise median across the time axis
        median_matrix = np.median(matrices, axis=0)
        
        # 2. SVD Cleanup
        r_base = self._orthogonalize_matrix(median_matrix)

        self.baselines = {
            "EAR": ear_median,
            "MAR": mar_median,
            "R": r_base
        }
        print(f"[Calibration] Complete. EAR_base: {ear_median:.3f}, MAR_base: {mar_median:.3f}")

    def transform_to_8d(self, raw_ear, raw_mar, raw_r):
        """
        Applies relative math and formats into the specific dataset column order:
        [R11, R21, R31, R12, R22, R32, EAR, MAR]
        """
        if not self.is_calibrated():
            raise ValueError("System is not calibrated yet.")

        # 1. EAR and MAR Relative Calibration
        ear_calib = raw_ear / self.baselines["EAR"]
        mar_calib = raw_mar - self.baselines["MAR"]

        # 2. Rotation Relative Calibration (R_curr * R_base^-1)
        # Note: Inverse of an orthogonal rotation matrix is its transpose.
        r_base_inv = self.baselines["R"].T 
        r_rel = r_base_inv @ raw_r 

        # 3. Extract the first two COLUMNS 
        c1 = r_rel[:, 0]
        c2 = r_rel[:, 1]

        # 4. Construct the final 8D Vector
        vector_8d = np.zeros(8, dtype=np.float32)
        vector_8d[0:3] = c1
        vector_8d[3:6] = c2
        vector_8d[6] = ear_calib
        vector_8d[7] = mar_calib

        return vector_8d

    def get_calibrated_sequences(self, sequence_length=60):
        """
        Transforms the raw calibration buffer into relative 8D vectors
        and chops them into non-overlapping sequences for batch GPU processing.
        
        Returns:
            np.ndarray: Shape (batch_size, sequence_length, 8) -> e.g., (15, 60, 8)
        """
        # Ensure we actually have the baseline medians calculated first
        if not self.baselines:
            print("[Calibrator] Error: Baselines not calculated yet.")
            return None

        # 1. Transform all raw frames into normalized 8D vectors
        normalized_batch = []
        for frame in self.calibration_buffer:
            vec_8d = self.transform_to_8d(frame["EAR"], frame["MAR"], frame["R"])
            normalized_batch.append(vec_8d)
            
        # Convert to a 2D numpy array -> Shape: (900, 8)
        normalized_batch = np.array(normalized_batch, dtype=np.float32)
        
        # 2. Chop into non-overlapping sequences
        num_frames = normalized_batch.shape[0]
        
        # Calculate how many full 60-frame chunks we can make (900 // 60 = 15)
        num_sequences = num_frames // sequence_length
        
        # Truncate any trailing frames if the buffer isn't perfectly divisible
        normalized_batch = normalized_batch[:num_sequences * sequence_length]
        
        # 3. Reshape array into PyTorch's native batch format -> Shape: (15, 60, 8)
        sequenced_batch = normalized_batch.reshape(num_sequences, sequence_length, 8)
        
        return sequenced_batch