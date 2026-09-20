import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path

try:
    import config
    DEFAULT_MSE_FILE = config.MSE_THRESHOLD_FILE
except ImportError:
    DEFAULT_MSE_FILE = str(Path(__file__).resolve().parent.parent / "mse_threshold.json")

class MSEBaselineManager:
    def __init__(self, filepath=None):
        self.filepath = filepath if filepath is not None else DEFAULT_MSE_FILE
        self.criterion = nn.MSELoss(reduction='none') # Don't average immediately

    def is_calibrated(self):
        """Returns True if the JSON threshold file exists."""
        return os.path.exists(self.filepath)

    def load_threshold(self):
        """Loads the saved threshold limit."""
        if self.is_calibrated():
            with open(self.filepath, 'r') as f:
                return json.load(f)["threshold_limit"]
        return None

    def compute_and_save(self, model, device, normalized_batch):
        """
        Takes the (900, 8) array, slices it into 60-frame sequences, 
        and calculates the statistical MSE threshold.
        """
        seq_len = 60
        num_frames = normalized_batch.shape[0]
        
        print(f"[MSE Calibrator] Slicing {num_frames} frames into sliding windows...")
        
        # 1. Create overlapping 60-frame windows
        # A 900-frame sequence yields 841 overlapping 60-frame windows
        sequences = []
        for i in range(num_frames - seq_len + 1):
            sequences.append(normalized_batch[i : i + seq_len])
            
        # 2. Convert to Tensor -> Shape: (841, 60, 8)
        batch_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)
        
        # 3. Massive GPU Batch Inference
        print(f"[MSE Calibrator] Running {len(sequences)} sequences through the Autoencoder...")
        with torch.no_grad():
            reconstructed = model(batch_tensor)
            
            # 4. Tail MSE Calculation: Compare only the 60th frame of each sequence
            # reconstructed[:, -1, :] compares against batch_tensor[:, -1, :]
            errors = self.criterion(reconstructed[:, -1, :], batch_tensor[:, -1, :])
            
            # Average across the 8 features to get a single MSE score per window
            # Resulting array shape: (841,)
            tail_mses = errors.mean(dim=1).cpu().numpy()
            
        # 5. Robust Statistics (Using Median to ignore single-frame blinks)
        median_mse = np.median(tail_mses)
        std_mse = np.std(tail_mses)
        
        # Calculate final threshold using Median + 3 Standard Deviations
        threshold = median_mse + (3 * std_mse)
        
        # 6. Save to Disk
        save_data = {
            "median_mse": float(median_mse),
            "std_dev": float(std_mse),
            "threshold_limit": float(threshold)
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
            
        print(f"[MSE Calibrator] Complete! Median: {median_mse:.6f} | Limit: {threshold:.6f}")
        return threshold