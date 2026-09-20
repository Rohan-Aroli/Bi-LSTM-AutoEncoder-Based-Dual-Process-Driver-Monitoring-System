import numpy as np

class MSEDataCollector:
    def __init__(self, fps=30, seconds=10):
        self.required_frames = fps * seconds  # e.g., 300 frames
        self.buffer = []

    def is_ready(self):
        return len(self.buffer) >= self.required_frames

    # def add_frame(self, vector_8d):
    #     """Appends a valid 8D frame. Ignores Nones (dropped frames)."""
    #     if not self.is_ready() and vector_8d is not None:
    #         self.buffer.append(vector_8d)
    #     return self.is_ready()

    # vision/mse_calib.py

    def add_frame(self, vector_8d):
        """
        Appends a valid 8D frame. 
        If a frame is missing (None), the timeline is broken. 
        Clears the buffer to ensure a strictly continuous baseline.
        """
        # 1. The Strict Reset Rule
        if vector_8d is None:
            if len(self.buffer) > 0:
                print(f"[MSE Calibrator] Frame dropped at {len(self.buffer)}/{self.required_frames}. Restarting baseline collection.")
                self.buffer.clear()
            return False
            
        # 2. Normal Collection
        if not self.is_ready():
            self.buffer.append(vector_8d)
            
        return self.is_ready()

    def get_batch(self):
        """Returns the continuous sequence as a PyTorch-ready NumPy array."""
        return np.array(self.buffer, dtype=np.float32)