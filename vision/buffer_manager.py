import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation, Slerp
import config

class BufferManager:
    def __init__(self):
        self.buffer = deque(maxlen=config.MAX_BUFFER_LEN)
        self.max_missing = config.MAX_MISSING_FRAMES
        
        self.missing_count = 0
        self.last_valid_vector = None

    def _6d_to_mat(self, v6):
        """
        Converts a continuous 6D rotation vector back into a 3x3 orthogonal matrix 
        using Gram-Schmidt on the first two COLUMNS.
        """
        c1 = v6[0:3]
        c2 = v6[3:6]

        # Normalize first column
        c1 = c1 / max(np.linalg.norm(c1), 1e-8)
        
        # Calculate third column as cross product of c1 and c2, then normalize
        c3 = np.cross(c1, c2)
        c3 = c3 / max(np.linalg.norm(c3), 1e-8)
        
        # Calculate orthogonal second column
        c2 = np.cross(c3, c1)

        return np.column_stack((c1, c2, c3))

    def _mat_to_6d(self, mat):
        """
        Flattens the first two COLUMNS of a 3x3 rotation matrix into a 6D vector.
        """
        return np.concatenate((mat[:, 0], mat[:, 1]))

    def update(self, current_8d_vector):
        """
        Takes the current frame's 8D vector (or None if face was lost).
        Returns a tuple: (status_string, payload_array)
        - "HOLD": Driver missing for too long. UI should show warnings. Payload is None.
        - "BUFFERING": Collecting frames, buffer not yet 60. Payload is None.
        - "READY": Buffer is 60 frames. Payload is (60, 8) numpy array.
        """
        # Case 1: Face is missing in this frame
        if current_8d_vector is None:
            self.missing_count += 1
            
            # The Guillotine: drop the buffer if missing for too long
            if self.missing_count > self.max_missing:
                self.buffer.clear()
                self.last_valid_vector = None
                return "HOLD", None
                
            return "BUFFERING", None

        # Case 2: Face is found, but we missed some frames previously
        if self.missing_count > 0 and self.last_valid_vector is not None:
            missing = self.missing_count
            start_vec = self.last_valid_vector
            end_vec = current_8d_vector
            
            # 1. Linear Interpolation for EAR (idx 0) and MAR (idx 1)
            ear_interp = np.linspace(start_vec[0], end_vec[0], missing + 2)[1:-1]
            mar_interp = np.linspace(start_vec[1], end_vec[1], missing + 2)[1:-1]
            
            # 2. SLERP for 6D Rotation (idx 2:8)
            mat_start = self._6d_to_mat(start_vec[2:8])
            mat_end = self._6d_to_mat(end_vec[2:8])
            
            # Ensure matrices are valid rotations before SLERP
            rotations = Rotation.from_matrix([mat_start, mat_end])
            slerp = Slerp([0, 1], rotations)
            times = np.linspace(0, 1, missing + 2)[1:-1]
            interp_rots = slerp(times).as_matrix()
            
            # 3. Reconstruct and append interpolated frames
            for i in range(missing):
                interp_6d = self._mat_to_6d(interp_rots[i])
                interp_vec = np.zeros(8)
                
                # Match dataset column order exactly
                interp_vec[0:6] = interp_6d
                interp_vec[6] = ear_interp[i]
                interp_vec[7] = mar_interp[i]
                
                self.buffer.append(interp_vec)
                
        # Case 3: Process the current valid frame
        self.missing_count = 0
        self.buffer.append(current_8d_vector)
        self.last_valid_vector = current_8d_vector

        # Check if we have a full rolling window to send to the PyTorch model
        if len(self.buffer) == config.MAX_BUFFER_LEN:
            return "READY", np.array(self.buffer, dtype=np.float32)
            
        return "BUFFERING", None