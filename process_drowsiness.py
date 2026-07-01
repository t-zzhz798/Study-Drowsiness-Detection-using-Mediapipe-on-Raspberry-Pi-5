import cv2
import mediapipe as mp
from drowsiness_state import DrowsinessMonitor, CALIBRATION_FRAMES
import time
from data_logger import log_data, LOG_INTERVAL

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

monitor = DrowsinessMonitor()
current_state = "NOT READY YET"


# LABELING FUNCTION FOR STREAMING
def stream_label(image, ear, perclos, mar, pitch, yaw, roll,
                 ear_drowsy, perclos_drowsy, mar_drowsy, head_drowsy,
                 yawning_now, recent_yawns,
                 ear_threshold, mar_threshold, pitch_threshold, roll_threshold,
                 calibrated, is_calibrating, calibration_frame_count,
                 can_calibrate, calibration_message):

    # CALIBRATION BLOCK
    if is_calibrating:
        shown_count = min(calibration_frame_count, CALIBRATION_FRAMES)

        if can_calibrate:
            status_text = f"CALIBRATING... LOOK FOCUSED {shown_count}/{CALIBRATION_FRAMES}"
            status_color = (0, 255, 255)
        else:
            status_text = "CANNOT CALIBRATE"
            status_color = (0, 0, 255)

        cv2.putText(image, calibration_message, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(image, f"Valid frames: {calibration_frame_count}/{CALIBRATION_FRAMES}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    elif calibrated:
        status_text = "CALIBRATED"
        status_color = (0, 255, 0)
        cv2.putText(
            image,
            f"State: {'CRITICAL' if perclos_drowsy else 'WARNING' if (mar_drowsy or ear_drowsy or head_drowsy) else 'NORMAL'}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
    else:
        status_text = "NOT CALIBRATED"
        status_color = (0, 0, 255)

    # WARNING STATUS BLOCK
    cv2.putText(image, status_text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # ALERTS BLOCK
    if ear_drowsy:
        cv2.putText(image, "CLOSED EYES!", (250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if perclos_drowsy:
        cv2.putText(image, "PERCLOS ALERT!", (250, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if mar_drowsy:
        cv2.putText(image, f"YAWN ALERT! ({recent_yawns} yawns)", (250, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif yawning_now:
        cv2.putText(image, "Yawning...", (250, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    if head_drowsy:
        cv2.putText(image, "HEAD NOD DETECTED!", (250, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # LIVE METRICS BLOCK
    cv2.putText(image, f"EAR: {ear:.2f}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"PERCLOS: {perclos:.1%}", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"MAR: {mar:.2f}", (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Pitch: {pitch:.1f}", (10, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Yaw:   {yaw:.1f}", (10, 290),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, f"Roll:  {roll:.1f}", (10, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # THRESHOLDS BLOCK
    cv2.putText(image, f"EAR thr: {ear_threshold:.2f}", (10, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(image, f"MAR thr: {mar_threshold:.2f}", (10, 390),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(image, f"Pitch thr: {pitch_threshold:.1f}", (10, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(image, f"Roll thr: {roll_threshold:.1f}", (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


# MAIN FUNCTION TO PROCESS FRAMES AND UPDATE MONITOR
def process_frames(picam2, lock, output_frame_container):
    global monitor, current_state

    last_log_time = 0

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=True,
        smooth_segmentation=True,
        refine_face_landmarks=True
    ) as holistic:

        while True:
            image = picam2.capture_array()
            image = cv2.flip(image, 1)

            results = holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
            )
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # THEN UPDATE MONITOR WITH LANDMARKS TO GET DROWSINESS STATE AND LABEL IMAGE
            if results.face_landmarks:
                h, w, _ = image.shape
                (
                    ear, perclos, mar, pitch, yaw, roll,
                    ear_drowsy, perclos_drowsy, mar_drowsy, head_drowsy,
                    yawning_now, recent_yawns,
                    ear_threshold, mar_threshold, pitch_threshold, roll_threshold,
                    calibrated, is_calibrating, calibration_frame_count,
                    can_calibrate, calibration_message
                ) = monitor.update(results.face_landmarks.landmark, w, h)

                stream_label(
                    image, ear, perclos, mar, pitch, yaw, roll,
                    ear_drowsy, perclos_drowsy, mar_drowsy, head_drowsy,
                    yawning_now, recent_yawns,
                    ear_threshold, mar_threshold, pitch_threshold, roll_threshold,
                    calibrated, is_calibrating, calibration_frame_count,
                    can_calibrate, calibration_message
                )

                if calibrated:
                    if perclos_drowsy:
                        current_state = "CRITICAL"
                    elif mar_drowsy or ear_drowsy or head_drowsy:
                        current_state = "WARNING"
                    else:
                        current_state = "NORMAL"
                else:
                    current_state = "NOT READY YET"

                print(f"State: {current_state}")

                now = time.time()
                if calibrated and now - last_log_time >= LOG_INTERVAL:
                    log_data(
                        ear, perclos, mar, pitch, yaw, roll,
                        ear_drowsy, perclos_drowsy, mar_drowsy, head_drowsy,
                        recent_yawns, calibrated
                    )
                    last_log_time = now

            with lock:
                _, encoded = cv2.imencode(".jpg", image)
                output_frame_container[0] = encoded.tobytes()