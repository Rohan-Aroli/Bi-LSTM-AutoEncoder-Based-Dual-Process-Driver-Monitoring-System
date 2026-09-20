import numpy as np

class MSEDataCollector:
    def __init__(self, fps=30, seconds=10):
        self.required_frames = fps * seconds  # e.g., 300 frames
        self.buffer = []

    def is_ready(self):
        return len(self.buffer) >= self.required_frames

    def add_frame(self, vector_8d):
        """Appends a valid 8D frame. Ignores Nones (dropped frames)."""
        if not self.is_ready() and vector_8d is not None:
            self.buffer.append(vector_8d)
        return self.is_ready()

    def get_batch(self):
        """Returns the continuous sequence as a PyTorch-ready NumPy array."""
        return np.array(self.buffer, dtype=np.float32)