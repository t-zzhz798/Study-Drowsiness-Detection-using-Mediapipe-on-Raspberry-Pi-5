import cv2
import numpy as np


# Mediapipe landmark numbers for each eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# Mouth points used for the mouth aspect ratio
# first two are the mouth corners, the rest are vertical distances
MOUTH = [
    61, 291,   # left and right mouth corners
    81, 311,   # outer mouth height
    78, 308,   # another outer height pair
    13, 14     # inner lip height
]


# points used for estimating where the head is facing
# nose, chin, eyes, mouth corners
FACE_3D_INDICES = [1, 152, 263, 33, 287, 57]


# rough 3D face model points used by solvePnP
FACE_3D_MODEL = np.array([
    [0.0,     0.0,    0.0],     # nose tip
    [0.0,   -330.0,  -65.0],    # chin
    [-225.0, 170.0, -135.0],    # left eye corner
    [225.0,  170.0, -135.0],    # right eye corner
    [-150.0, -150.0, -125.0],   # left mouth corner
    [150.0,  -150.0, -125.0],   # right mouth corner
], dtype=np.float64)


def get_eye_landmarks(landmarks, indices, w, h):
    # convert normalized Mediapipe coordinates into actual pixel positions
    return np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in indices
    ], dtype=np.float64)


def get_mouth_landmarks(landmarks, w, h):
    # convert normalized Mediapipe coordinates into actual pixel positions
    return np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in MOUTH
    ], dtype=np.float64)


def eye_aspect_ratio(eye):
    # vertical eye distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # horizontal eye distance
    C = np.linalg.norm(eye[0] - eye[3])

    # avoid dividing by zero
    if C == 0:
        return 0.0

    # smaller value means the eye is more closed
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    # mouth width
    D = np.linalg.norm(mouth[0] - mouth[1])

    N1 = np.linalg.norm(mouth[2] - mouth[3])
    N2 = np.linalg.norm(mouth[4] - mouth[5])
    N3 = np.linalg.norm(mouth[6] - mouth[7])

    if D == 0:
        return 0.0

    # bigger value means mouth is more open
    return (N1 + N2 + N3) / (3.0 * D)


def get_head_pose(landmarks, w, h):
    # get the 2D face points from Mediapipe
    face_2d = np.array([
        [landmarks[i].x * w, landmarks[i].y * h]
        for i in FACE_3D_INDICES
    ], dtype=np.float64)

    # basic camera matrix
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

    # if pose detection fails, just return neutral values
    if not success:
        return 0.0, 0.0, 0.0

    # convert rotation vector into angles
    rot_matrix, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_matrix)

    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]

    return pitch, yaw, roll