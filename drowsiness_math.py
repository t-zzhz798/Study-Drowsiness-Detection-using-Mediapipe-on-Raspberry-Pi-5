import cv2
import numpy as np

#RIGHT AND LEFT EYE LANDMARKS
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

#MOUTH LANDMARKS
MOUTH = [
    61, 291,   # 0, 1 - left/right corners
    81, 311,   # 2, 3 - outer vertical pair 1
    78, 308,   # 4, 5 - outer vertical pair 2
    13, 14     # 6, 7 - inner vertical pair
]

#FACE LANDMARKS FOR HEAD POSE ESTIMATION
FACE_3D_INDICES = [1, 152, 263, 33, 287, 57]

# 3D MODEL POINTS FOR HEAD POSE ESTIMATION
FACE_3D_MODEL = np.array([
    [0.0,    0.0,    0.0  ],   # Nose tip
    [0.0,   -330.0, -65.0 ],   # Chin
    [-225.0, 170.0, -135.0],   # Left eye corner
    [225.0,  170.0, -135.0],   # Right eye corner
    [-150.0, -150.0, -125.0],  # Left mouth
    [150.0, -150.0, -125.0],   # Right mouth
], dtype=np.float64)

# RETURNS LOCATION OF EYE LANDMARKS AS NUMPY ARRAY
def get_eye_landmarks(landmarks, indices, w, h):
    return np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in indices
    ], dtype=np.float64)

# RETURNS LOCATION OF MOUTH LANDMARKS AS NUMPY ARRAY
def get_mouth_landmarks(landmarks, w, h):
    return np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in MOUTH
    ], dtype=np.float64)

#EAR CALCULATE
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)

#MAR CALCULATE
def mouth_aspect_ratio(mouth):
    D = np.linalg.norm(mouth[0] - mouth[1])
    N1 = np.linalg.norm(mouth[2] - mouth[3])
    N2 = np.linalg.norm(mouth[4] - mouth[5])
    N3 = np.linalg.norm(mouth[6] - mouth[7])

    if D == 0:
        return 0.0

    return (N1 + N2 + N3) / (3.0 * D)

#YAW, ROLL, PITCH CALCULATE
def get_head_pose(landmarks, w, h):
    face_2d = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in FACE_3D_INDICES
    ], dtype=np.float64)

    focal_length = w
    cam_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL,
        face_2d,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_matrix, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll