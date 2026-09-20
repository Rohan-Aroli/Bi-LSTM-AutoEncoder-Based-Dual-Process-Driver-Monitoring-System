import cv2
import numpy as np
from config import CANONICAL_FACE_POINTS

def get_2d_landmarks(face_landmarks, img_w, img_h):
    """
    Extracts the 6 canonical facial points required for cv2.solvePnP.
    Indices: Nose tip (1), Chin (152), Left Eye Outer (33), 
             Right Eye Outer (263), Left Mouth Outer (61), Right Mouth Outer (291)
    """
    indices = [1, 152, 33, 263, 61, 291]
    points = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        x, y = int(landmark.x * img_w), int(landmark.y * img_h)
        points.append([x, y])
    return np.array(points, dtype=np.float64)

def get_rotation_matrix(image_points, img_w, img_h):
    """
    Computes the 3x3 rotation matrix using solvePnP and Rodrigues' rotation formula.
    """
    focal_length = img_w
    cam_matrix = np.array([[focal_length, 0, img_w / 2],
                           [0, focal_length, img_h / 2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    
    success, rotation_vec, translation_vec = cv2.solvePnP(
        CANONICAL_FACE_POINTS, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return None
        
    # Convert the 3D rotation vector into a 3x3 orthogonal rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    return rotation_mat

def _euclidean_dist(p1, p2):
    """Helper function to calculate L2 norm between two 2D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_aspect_ratios(face_landmarks, img_w, img_h):
    """
    Calculates EAR and MAR using strict Euclidean distances.
    Multiplying by img_w and img_h converts normalized MediaPipe output to raw pixel scales.
    """
    def get_pt(idx):
        pt = face_landmarks.landmark[idx]
        return [pt.x * img_w, pt.y * img_h]

    # ---------------------------------------------------------
    # 1. EAR (Eye Aspect Ratio) Calculation
    # ---------------------------------------------------------
    # Left Eye MediaPipe Indices
    l_outer = get_pt(33)
    l_inner = get_pt(133)
    l_top1, l_bot1 = get_pt(160), get_pt(144)
    l_top2, l_bot2 = get_pt(158), get_pt(153)
    
    # EAR formula: (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    l_ear = (_euclidean_dist(l_top1, l_bot1) + _euclidean_dist(l_top2, l_bot2)) / \
            (2.0 * _euclidean_dist(l_outer, l_inner))

    # Right Eye MediaPipe Indices
    r_inner = get_pt(362)
    r_outer = get_pt(263)
    r_top1, r_bot1 = get_pt(385), get_pt(380)
    r_top2, r_bot2 = get_pt(387), get_pt(373)
    
    r_ear = (_euclidean_dist(r_top1, r_bot1) + _euclidean_dist(r_top2, r_bot2)) / \
            (2.0 * _euclidean_dist(r_inner, r_outer))

    raw_ear = (l_ear + r_ear) / 2.0

    # ---------------------------------------------------------
    # 2. MAR (Mouth Aspect Ratio) Calculation
    # ---------------------------------------------------------
    # Mouth MediaPipe Indices (using 3 vertical measurements for robust detection)
    m_left = get_pt(78)
    m_right = get_pt(308)
    
    m_top1, m_bot1 = get_pt(82), get_pt(87)   # Left side of lips
    m_top2, m_bot2 = get_pt(13), get_pt(14)   # Center of lips
    m_top3, m_bot3 = get_pt(312), get_pt(317) # Right side of lips
    
    mar_v1 = _euclidean_dist(m_top1, m_bot1)
    mar_v2 = _euclidean_dist(m_top2, m_bot2)
    mar_v3 = _euclidean_dist(m_top3, m_bot3)
    mar_h = _euclidean_dist(m_left, m_right)
    
    # MAR formula: (V1 + V2 + V3) / (2 * H)
    raw_mar = (mar_v1 + mar_v2 + mar_v3) / (2.0 * mar_h)

    return raw_ear, raw_mar