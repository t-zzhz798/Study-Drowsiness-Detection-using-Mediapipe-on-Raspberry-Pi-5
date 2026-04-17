import collections
import numpy as np
from drowsiness_math import (
    eye_aspect_ratio,
    get_eye_landmarks,
    mouth_aspect_ratio,
    get_mouth_landmarks,
    get_head_pose,
    LEFT_EYE,
    RIGHT_EYE
)

# DROWSINESS DETECTION PARAMETERS
EAR_CONSEC_FRAMES = 38
EAR_DROWSY = 0.75

PERCLOS_WINDOW = 500
PERCLOS_THRESHOLD = 0.28

MAR_YAWN_MIN_FRAMES = 15
MAR_YAWN_MAX_FRAMES = 90
MAR_YAWN_WINDOW = 200
MAR_YAWN_COUNT = 2
NOD_CONSEC_FRAMES = 38
MAR_DROWSY = 1.68

PITCH_TOLERANCE = 30
ROLL_TOLERANCE = 30

CALIBRATION_FRAMES = 120
MINIMUM_EAR_FOR_BASELINE = 0.18
MAXIMUM_MAR_FOR_BASELINE = 0.80
MAXIMUM_PITCH_FOR_BASELINE = 30
MAXIMUM_ROLL_FOR_BASELINE = 30
BASELINE_WINDOW = 150


# DROWSINESS MONITOR CLASS
class DrowsinessMonitor:
    # INITIALIZE ALL VARIABLES
    def __init__(self):
        self.frame_count = 0

        self.eye_closure_history = collections.deque(maxlen=PERCLOS_WINDOW)
        self.perclos = 0.0

        self.mar_frame_count = 0
        self.mar_in_yawn = False
        self.yawn_history = collections.deque(maxlen=MAR_YAWN_WINDOW)
        self.recent_yawns = 0

        self.nod_frame_count = 0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0

        self.ear_threshold = 0.0
        self.mar_threshold = 0.0
        self.pitch_threshold = 0.0
        self.roll_threshold = 0.0

        self.calibrated = False
        self.is_calibrating = False
        self.calibration_frame_count = 0

        self.can_calibrate = True
        self.calibration_message = "Ready to calibrate"

        self.ear_baseline_history = collections.deque(maxlen=BASELINE_WINDOW)
        self.mar_baseline_history = collections.deque(maxlen=BASELINE_WINDOW)
        self.pitch_baseline_history = collections.deque(maxlen=BASELINE_WINDOW)
        self.roll_baseline_history = collections.deque(maxlen=BASELINE_WINDOW)

    # CALIBRATION STATE
    def start_calibration(self):
        self.is_calibrating = True
        self.calibrated = False
        self.calibration_frame_count = 0

        self.can_calibrate = True
        self.calibration_message = "Hold still with eyes open and mouth closed"

        self.ear_baseline_history.clear()
        self.mar_baseline_history.clear()
        self.pitch_baseline_history.clear()
        self.roll_baseline_history.clear()

        self.frame_count = 0
        self.mar_frame_count = 0
        self.mar_in_yawn = False
        self.nod_frame_count = 0
        self.yawn_history.clear()
        self.eye_closure_history.clear()
        self.recent_yawns = 0
        self.perclos = 0.0

    # RESET CALIBRATION STATE
    def reset_calibration(self):
        self.is_calibrating = False
        self.calibrated = False
        self.calibration_frame_count = 0

        self.can_calibrate = True
        self.calibration_message = "Calibration reset"

        self.ear_baseline_history.clear()
        self.mar_baseline_history.clear()
        self.pitch_baseline_history.clear()
        self.roll_baseline_history.clear()

        self.ear_threshold = 0
        self.mar_threshold = 0
        self.pitch_threshold = 0
        self.roll_threshold = 0

        self.frame_count = 0
        self.mar_frame_count = 0
        self.mar_in_yawn = False
        self.nod_frame_count = 0
        self.yawn_history.clear()
        self.eye_closure_history.clear()
        self.recent_yawns = 0
        self.perclos = 0.0

    # UPDATE THRESHOLDS DURING CALIBRATION
    def update_thresholds(self, ear, mar, pitch, roll):
        if not self.is_calibrating:
            return

        self.can_calibrate = True
        self.calibration_message = "Good calibration frame"

        if ear <= MINIMUM_EAR_FOR_BASELINE:
            self.can_calibrate = False
            self.calibration_message = "Cannot calibrate: open your eyes more"
        elif mar >= MAXIMUM_MAR_FOR_BASELINE:
            self.can_calibrate = False
            self.calibration_message = "Cannot calibrate: close your mouth"
        elif abs(pitch) >= MAXIMUM_PITCH_FOR_BASELINE:
            self.can_calibrate = False
            self.calibration_message = "Cannot calibrate: face forward"
        elif abs(roll) >= MAXIMUM_ROLL_FOR_BASELINE:
            self.can_calibrate = False
            self.calibration_message = "Cannot calibrate: keep head upright"

        if self.can_calibrate:
            self.ear_baseline_history.append(ear)
            self.mar_baseline_history.append(mar)
            self.pitch_baseline_history.append(pitch)
            self.roll_baseline_history.append(roll)
            self.calibration_frame_count += 1

        enough_data = (
            len(self.ear_baseline_history) >= CALIBRATION_FRAMES and
            len(self.mar_baseline_history) >= CALIBRATION_FRAMES and
            len(self.pitch_baseline_history) >= CALIBRATION_FRAMES and
            len(self.roll_baseline_history) >= CALIBRATION_FRAMES
        )

        # calculate thresholds once enough data is collected and end calibration
        if enough_data:
            ear_mean = np.mean(self.ear_baseline_history)
            mar_mean = np.mean(self.mar_baseline_history)
            pitch_mean = np.mean(self.pitch_baseline_history)
            roll_mean = np.mean(self.roll_baseline_history)

            self.ear_threshold = ear_mean * EAR_DROWSY
            self.mar_threshold = mar_mean * MAR_DROWSY
            self.pitch_threshold = pitch_mean
            self.roll_threshold = roll_mean

            self.calibrated = True
            self.is_calibrating = False
            self.can_calibrate = True
            self.calibration_message = "Calibration complete"

    # UPDATE EAR PER FRAME AND DETERMINE IF DROWSY MORE THAN SOME CONSECUTIVE FRAMES
    def _update_ear(self, ear):
        if ear < self.ear_threshold:
            self.frame_count += 1
            return self.frame_count >= EAR_CONSEC_FRAMES
        else:
            self.frame_count = 0
            return False

    # UPDATE PERCLOS OVER DEFINED FRAMES AND DETERMINE IF ABOVE THRESHOLD
    def _update_perclos(self, ear):
        self.eye_closure_history.append(ear < self.ear_threshold)

        if len(self.eye_closure_history) == PERCLOS_WINDOW:
            self.perclos = sum(self.eye_closure_history) / PERCLOS_WINDOW
            return self.perclos, self.perclos > PERCLOS_THRESHOLD

        return 0.0, False

    # UPDATE MAR AND YAWN HISTORY WITH YAWN EVENTS AND DETERMINE IF YAWNING OR ABOVE YAWN COUNT THRESHOLD
    def _update_mar(self, mar):
        yawn_event = False

        if mar > self.mar_threshold:
            self.mar_frame_count += 1
            self.mar_in_yawn = True
        else:
            if self.mar_in_yawn:
                if MAR_YAWN_MIN_FRAMES <= self.mar_frame_count <= MAR_YAWN_MAX_FRAMES:
                    yawn_event = True

            self.mar_frame_count = 0
            self.mar_in_yawn = False

        self.yawn_history.append(1 if yawn_event else 0)
        self.recent_yawns = sum(self.yawn_history)

        yawning_now = (
            self.mar_in_yawn and
            self.mar_frame_count >= MAR_YAWN_MIN_FRAMES
        )

        mar_drowsy = self.recent_yawns >= MAR_YAWN_COUNT
        return mar_drowsy, yawning_now, self.recent_yawns

    # UPDATE YAW AND PITCH AND DETERMINE DROWSY IF ABOVE THRESHOLD FOR SOME CONSECUTIVE FRAMES
    def _update_head_pose(self, pitch, roll):
        if (
            pitch > self.pitch_threshold + PITCH_TOLERANCE or
            pitch < self.pitch_threshold - PITCH_TOLERANCE or
            roll > self.roll_threshold + ROLL_TOLERANCE or
            roll < self.roll_threshold - ROLL_TOLERANCE
        ):
            self.nod_frame_count += 1
            head_drowsy = self.nod_frame_count >= NOD_CONSEC_FRAMES
        else:
            self.nod_frame_count = 0
            head_drowsy = False

        return head_drowsy

    # MAIN UPDATE FUNCTION PER FRAME
    def update(self, landmarks, w, h):
        # EAR
        left_eye = get_eye_landmarks(landmarks, LEFT_EYE, w, h)
        right_eye = get_eye_landmarks(landmarks, RIGHT_EYE, w, h)
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # MAR
        mouth = get_mouth_landmarks(landmarks, w, h)
        mar = mouth_aspect_ratio(mouth)

        # HEAD POSE
        pitch, yaw, roll = get_head_pose(landmarks, w, h)
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

        # CALIBRATE AND UPDATE THRESHOLDS
        self.update_thresholds(ear, mar, pitch, roll)

        if not self.calibrated:
            return (
                ear, 0.0, mar, pitch, yaw, roll,
                False, False, False, False,
                False, 0,
                self.ear_threshold, self.mar_threshold, self.pitch_threshold, self.roll_threshold,
                self.calibrated, self.is_calibrating, self.calibration_frame_count,
                self.can_calibrate, self.calibration_message
            )

        # RUN ALL DETECTORS FOR DROWSINESS
        ear_drowsy = self._update_ear(ear)
        perclos, perclos_drowsy = self._update_perclos(ear)
        mar_drowsy, yawning_now, recent_yawns = self._update_mar(mar)
        head_drowsy = self._update_head_pose(pitch, roll)

        # RETURN ALL VALUES FOR STREAMING AND STATUS
        return (
            ear, perclos, mar, pitch, yaw, roll,
            ear_drowsy, perclos_drowsy, mar_drowsy, head_drowsy,
            yawning_now, recent_yawns,
            self.ear_threshold, self.mar_threshold, self.pitch_threshold, self.roll_threshold,
            self.calibrated, self.is_calibrating, self.calibration_frame_count,
            self.can_calibrate, self.calibration_message
        )