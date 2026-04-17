import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from webhooks import (
    send_webhook_middle_finger,
    send_webhook_time_out,
    send_webhook_love_heart,
    send_webhook_tate,
    send_webhook_jjk,
    send_webhook_gang_signs,
    send_webhook_kawaii,
    send_webhook_angry,
    send_webhook_sob
)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

with open('gesture_model.pkl', 'rb') as f:
    gesture_model = pickle.load(f)

def get_landmark_array(hand_landmarks):
    lm = hand_landmarks.landmark
    return [coord for lm_point in lm for coord in (lm_point.x, lm_point.y, lm_point.z)]

def classify_gesture(multi_hand_landmarks):
    all_features = []
    for hand_landmarks in multi_hand_landmarks:
        all_features.extend(get_landmark_array(hand_landmarks))
    while len(all_features) < 126:
        all_features.extend([0.0] * 63)

    prediction = gesture_model.predict([all_features[:126]])[0]
    confidence = gesture_model.predict_proba([all_features[:126]]).max()
    return prediction, confidence

def process_frames(picam2, lock, output_frame_container):
    last_detected = 0  # single timer, not per gesture
    recognition_cooldown = 3  # seconds to pause ALL recognition after a detection

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while True:
            image = picam2.capture_array()
            image = cv2.flip(image, 1)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            now = time.time()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if now - last_detected > recognition_cooldown:  # only classify if cooldown passed
                    gesture, confidence = classify_gesture(results.multi_hand_landmarks)
                    print(f"Gesture: {gesture}, Confidence: {confidence:.2f}")

                    if confidence > 0.6 and gesture != 'none':
                        cv2.putText(image, f'{gesture} ({confidence:.0%})', (50, 150),
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6)
                        last_detected = now  # reset cooldown on ANY detection

                        if gesture == 'middle_up':
                            send_webhook_middle_finger()
                        elif gesture == 'time_out':
                            send_webhook_time_out()
                        elif gesture == 'love_heart':
                            send_webhook_love_heart()
                        elif gesture == 'andrew_tate':
                            send_webhook_tate()
                        elif gesture == 'jjk':
                            send_webhook_jjk()
                        elif gesture == 'gang_signs':
                            send_webhook_gang_signs()
                        elif gesture == 'kawaii':
                            send_webhook_kawaii()
                        elif gesture == 'angry':
                            send_webhook_angry()
                        elif gesture == 'sob':
                            send_webhook_sob()
                else:
                    # show cooldown timer on screen so you know when it resets
                    remaining = recognition_cooldown - (now - last_detected)
                    cv2.putText(image, f'Cooldown: {remaining:.1f}s', (50, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)

            with lock:
                _, encoded = cv2.imencode('.jpg', image)
                output_frame_container[0] = encoded.tobytes()