# import torch
# import torch.nn as nn
# import numpy as np
# import json
# import os

# class MSEBaselineManager:
#     def __init__(self, filepath="mse_threshold.json"):
#         self.filepath = filepath
#         self.criterion = nn.MSELoss(reduction='none') # Don't average immediately

#     def is_calibrated(self):
#         """Returns True if the JSON threshold file exists."""
#         return os.path.exists(self.filepath)

#     def load_threshold(self):
#         """Loads the saved threshold limit."""
#         if self.is_calibrated():
#             with open(self.filepath, 'r') as f:
#                 return json.load(f)["threshold_limit"]
#         return None

#     def compute_and_save(self, model, device, normalized_batch):
#         """
#         Takes the (900, 8) array, slices it into 60-frame sequences, 
#         and calculates the statistical MSE threshold.
#         """
#         seq_len = 60
#         num_frames = normalized_batch.shape[0]
        
#         print(f"[MSE Calibrator] Slicing {num_frames} frames into non-overlapping windows...")
        
#         # 1. Create strictly non-overlapping 60-frame windows
#         sequences = []
#         for i in range(0, num_frames - seq_len + 1, seq_len):
#             sequences.append(normalized_batch[i : i + seq_len])
            
#         # 2. Convert to Tensor -> Shape: (N, 60, 8)
#         batch_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)
        
#         # 3. Massive GPU Batch Inference
#         print(f"[MSE Calibrator] Running {len(sequences)} sequences through the Autoencoder...")
#         with torch.no_grad():
#             reconstructed = model(batch_tensor)
            
#             # 4. Tail MSE Calculation: Compare all 60 frames of each sequence

#             errors = self.criterion(reconstructed, batch_tensor)
#             # 4. Sequence MSE Calculation: Evaluate the FULL 60-frame sequence
#             raw_errors = self.criterion(reconstructed, batch_tensor)
            
#             # Average across the 8 features to get a single MSE score per window
#             # Resulting array shape: (N,)
#             tail_mses = errors.mean(dim=1).cpu().numpy()
#             # Average across both time dimension (dim=1) and feature dimension (dim=2)
#             errors = raw_errors.mean(dim=(1, 2)).cpu().numpy()
            
#         # 5. Robust Statistics (Using Median to ignore single-frame blinks)
#         median_mse = np.median(tail_mses)
#         std_mse = np.std(tail_mses)
#         # 5. Dynamic Data-Driven Threshold (Max normal error + 20% margin)
#         max_normal_error = np.max(errors)
#         # ANOMALY_THRESHOLD = max_normal_error * 1.20
#         ANOMALY_THRESHOLD = max_normal_error 
    

#         # Calculate final threshold using Median + 3 Standard Deviations
#         threshold = median_mse + (3 * std_mse)
        
#         # 6. Save to Disk
#         save_data = {
#             "median_mse": float(median_mse),
#             "std_dev": float(std_mse),
#             "threshold_limit": float(threshold),
#             "max_normal_error": float(max_normal_error),
#             "threshold_limit": float(ANOMALY_THRESHOLD)
#         }
        
#         with open(self.filepath, 'w') as f:
#             json.dump(save_data, f, indent=4)
            
#         print(f"[MSE Calibrator] Complete! Median: {median_mse:.6f} | Limit: {threshold:.6f}")
#         return threshold
#         print(f"[MSE Calibrator] Complete! Max Normal Error: {max_normal_error:.6f} | Limit: {ANOMALY_THRESHOLD:.6f}")
#         return ANOMALY_THRESHOLD

import torch
import torch.nn as nn
import numpy as np
import json
import os

class MSEBaselineManager:
    def __init__(self, filepath="mse_threshold.json"):
        self.filepath = filepath
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
        Takes the calibration array, slices it into 60-frame sequences, 
        and calculates the statistical MSE threshold.
        """
        seq_len = 60
        num_frames = normalized_batch.shape[0]
        
        print(f"[MSE Calibrator] Slicing {num_frames} frames into non-overlapping windows...")
        
        # 1. Create strictly non-overlapping 60-frame windows
        sequences = []
        for i in range(0, num_frames - seq_len + 1, seq_len):
            sequences.append(normalized_batch[i : i + seq_len])
            
        # 2. Convert to Tensor -> Shape: (N, 60, 8)
        batch_tensor = torch.tensor(np.array(sequences), dtype=torch.float32).to(device)
        
        # 3. Massive GPU Batch Inference
        print(f"[MSE Calibrator] Running {len(sequences)} sequences through the Autoencoder...")
        with torch.no_grad():
            reconstructed = model(batch_tensor)
            
            # 4. Sequence MSE Calculation: Evaluate the FULL 60-frame sequence
            raw_errors = self.criterion(reconstructed, batch_tensor)
            
            # Average across both time dimension (dim=1) and feature dimension (dim=2)
            # Resulting array shape: (N,) - one MSE score per sequence
            errors = raw_errors.mean(dim=(1, 2)).cpu().numpy()
            
        # 5. Dynamic Data-Driven Threshold
        # Using a 35% margin over the max normal error to account for edge hardware noise
        max_normal_error = np.max(errors)
        margin_multiplier = 1.1
        ANOMALY_THRESHOLD = max_normal_error * margin_multiplier
        
        # 6. Save to Disk
        save_data = {
            "max_normal_error": float(max_normal_error),
            "margin_multiplier": float(margin_multiplier),
            "threshold_limit": float(ANOMALY_THRESHOLD)
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
            
        print(f"[MSE Calibrator] Complete! Max Normal Error: {max_normal_error:.6f} | Limit: {ANOMALY_THRESHOLD:.6f}")
        return ANOMALY_THRESHOLD