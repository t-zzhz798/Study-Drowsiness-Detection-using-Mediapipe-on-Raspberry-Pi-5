import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def dist(a,b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2) ** 0.5

def get_landmarks(hand_landmarks):
    lm=hand_landmarks.landmark
       
    return { 
        'wrist': lm[mp_hands.HandLandmark.WRIST],
        'thumb_tip': lm[mp_hands.HandLandmark.THUMB_TIP],
        'thumb_mcp': lm[mp_hands.HandLandmark.THUMB_MCP],
        'index_tip': lm[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        'index_pip': lm[mp_hands.HandLandmark.INDEX_FINGER_PIP],
        'middle_tip': lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        'middle_pip': lm[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
        'ring_tip': lm[mp_hands.HandLandmark.RING_FINGER_TIP],
        'ring_pip': lm[mp_hands.HandLandmark.RING_FINGER_PIP],
        'pinky_tip': lm[mp_hands.HandLandmark.PINKY_TIP],
        'pinky_pip': lm[mp_hands.HandLandmark.PINKY_PIP]
    }

#Middle Finger Recognition
def middle_up(hand_landmarks, handedness = "Right"):
    landmarks = hand_landmarks.landmark
    p=get_landmarks(hand_landmarks)
    wrist = p['wrist']

    middle_extended= dist(p["middle_tip"], wrist) > dist(p["middle_pip"], wrist)
    index_curled= dist(p["index_tip"],  wrist) < dist(p["index_pip"],  wrist)
    ring_curled= dist(p["ring_tip"],   wrist) < dist(p["ring_pip"],   wrist)
    pinky_curled= dist(p["pinky_tip"],  wrist) < dist(p["pinky_pip"],  wrist)

    if handedness == "Right":
        thumb_curled = p["thumb_tip"].x < p["thumb_mcp"].x
    else:
        thumb_curled = p["thumb_tip"].x > p["thumb_mcp"].x
    return middle_extended and index_curled and ring_curled and pinky_curled and thumb_curled

#Time Out Recognition
def time_out(hand_landmarks, handness_list, threshold=0.3):
    
    def right_hand_flat(hand_landmarks):
        p=get_landmarks(hand_landmarks)

        thumb_flat_right = p["thumb_tip"].x < p["thumb_mcp"].x and abs(p["thumb_tip"].y - p["thumb_mcp"].y) < 0.1
        index_flat_right = p["index_tip"].x < p["index_pip"].x and abs(p["index_tip"].y - p["index_pip"].y) < 0.1
        middle_flat_right = p["middle_tip"].x < p["middle_pip"].x and  abs(p["middle_tip"].y - p["middle_pip"].y) < 0.1
        ring_flat_right = p["ring_tip"].x < p["ring_pip"].x and abs(p["ring_tip"].y - p["ring_pip"].y) < 0.1
        pinky_flat_right = p["pinky_tip"].x < p["pinky_pip"].x and abs(p["pinky_tip"].y - p["pinky_pip"].y) < 0.1

        return thumb_flat_right and index_flat_right and middle_flat_right and ring_flat_right and pinky_flat_right

    def left_hand_flat(hand_landmarks):
        p=get_landmarks(hand_landmarks)

        thumb_flat_left = p["thumb_tip"].x > p["thumb_mcp"].x and abs(p["thumb_tip"].y - p["thumb_mcp"].y) < 0.1
        index_flat_left = p["index_tip"].x > p["index_pip"].x and abs(p["index_tip"].y - p["index_pip"].y) < 0.1
        middle_flat_left = p["middle_tip"].x > p["middle_pip"].x and abs(p["middle_tip"].y - p["middle_pip"].y) < 0.1
        ring_flat_left = p["ring_tip"].x > p["ring_pip"].x and abs(p["ring_tip"].y - p["ring_pip"].y) < 0.1
        pinky_flat_left = p["pinky_tip"].x > p["pinky_pip"].x and abs(p["pinky_tip"].y - p["pinky_pip"].y) < 0.1

        return thumb_flat_left and index_flat_left and middle_flat_left and ring_flat_left and pinky_flat_left

    def hand_vertical(hand_landmarks):
        p=get_landmarks(hand_landmarks)

        thumb_vertical = p["thumb_tip"].y < p["thumb_mcp"].y and abs(p["thumb_tip"].x - p["thumb_mcp"].x) < 0.1
        index_vertical = p["index_tip"].y < p["index_pip"].y and abs(p["index_tip"].x - p["index_pip"].x) < 0.1
        middle_vertical = p["middle_tip"].y < p["middle_pip"].y and abs(p["middle_tip"].x - p["middle_pip"].x) < 0.1
        ring_vertical = p["ring_tip"].y < p["ring_pip"].y and abs(p["ring_tip"].x - p["ring_pip"].x) < 0.1
        pinky_vertical = p["pinky_tip"].y < p["pinky_pip"].y and abs(p["pinky_tip"].x - p["pinky_pip"].x) < 0.1

        return thumb_vertical and index_vertical and middle_vertical and ring_vertical and pinky_vertical

    def get_flat_hand_center_x(hand_landmarks):
        p = get_landmarks(hand_landmarks)
        # Average x position of all fingertips on the flat hand
        return (p["index_tip"].x + p["middle_tip"].x + p["ring_tip"].x + p["pinky_tip"].x) / 4

    def get_vertical_middle_finger_tip_x(hand_landmarks):
        p = get_landmarks(hand_landmarks)
        return p["middle_tip"].x

    right_flat = False
    left_flat = False
    right_vertical = False
    left_vertical = False
    right_landmarks = None
    left_landmarks = None
    threshold = 0.3  # Adjust this threshold as needed
    
    for hand_landmark, handnessness in zip(hand_landmarks, handness_list):
        label = handnessness.classification[0].label
        
        if label == "Right":
            right_flat = right_hand_flat(hand_landmark)
            right_vertical = hand_vertical(hand_landmark)
            right_landmarks = hand_landmark
        else:
            left_flat = left_hand_flat(hand_landmark)
            left_vertical = hand_vertical(hand_landmark)
            left_landmarks = hand_landmark

    if right_flat and left_vertical and right_landmarks and left_landmarks:
        flat_center_x = get_flat_hand_center_x(right_landmarks)
        vertical_middle_x = get_vertical_middle_finger_tip_x(left_landmarks)
        if abs(flat_center_x - vertical_middle_x) < threshold:
            return True

    if left_flat and right_vertical and left_landmarks and right_landmarks:
        flat_center_x = get_flat_hand_center_x(left_landmarks)
        vertical_middle_x = get_vertical_middle_finger_tip_x(right_landmarks)
        if abs(flat_center_x - vertical_middle_x) < threshold:
            return True

    return False

#Love Heart Recognition
def love_heart(hand_landmarks, handness_list):

    def right_hand_curled(hand_landmarks):
        p=get_landmarks(hand_landmarks)

        thumb_curled_right = p["thumb_tip"].x < p["thumb_mcp"].x and p["thumb_tip"].y > p["thumb_mcp"].y
        index_curled_right = p["index_tip"].x < p["index_pip"].x and p["index_tip"].y > p["index_pip"].y
        middle_curled_right = p["middle_tip"].x < p["middle_pip"].x and p["middle_tip"].y > p["middle_pip"].y
        ring_curled_right = p["ring_tip"].x < p["ring_pip"].x and  p["ring_tip"].y > p["ring_pip"].y
        pinky_curled_right = p["pinky_tip"].x < p["pinky_pip"].x and p["pinky_tip"].y > p["pinky_pip"].y

        return thumb_curled_right and index_curled_right and middle_curled_right and ring_curled_right and pinky_curled_right

    def left_hand_curled(hand_landmarks):
        p=get_landmarks(hand_landmarks)

        thumb_curled_left = p["thumb_tip"].x > p["thumb_mcp"].x and p["thumb_tip"].y > p["thumb_mcp"].y
        index_curled_left = p["index_tip"].x > p["index_pip"].x and p["index_tip"].y > p["index_pip"].y
        middle_curled_left = p["middle_tip"].x > p["middle_pip"].x and p["middle_tip"].y > p["middle_pip"].y
        ring_curled_left = p["ring_tip"].x > p["ring_pip"].x and  p["ring_tip"].y > p["ring_pip"].y
        pinky_curled_left = p["pinky_tip"].x > p["pinky_pip"].x and p["pinky_tip"].y > p["pinky_pip"].y

        return thumb_curled_left and index_curled_left and middle_curled_left and ring_curled_left and pinky_curled_left

    right_curled=False
    left_curled=False

    for hand_landmarks, handnessness in zip(hand_landmarks, handness_list):
        label = handnessness.classification[0].label

        if label == "Right":
            right_curled = right_hand_curled(hand_landmarks)
        else:
            left_curled = left_hand_curled(hand_landmarks)
    
    return right_curled and left_curled

#Andrew Tate Hands
def andrew_tate_hands(hand_landmarks, handness_list, threshold=0.1):

    def hand_burns_shape(hand_landmarks):
        p = get_landmarks(hand_landmarks)

        thumb_up    = p["thumb_tip"].y  < p["thumb_ip"].y
        index_bent  = p["index_tip"].y  > p["index_pip"].y
        middle_bent = p["middle_tip"].y > p["middle_pip"].y
        ring_bent   = p["ring_tip"].y   > p["ring_pip"].y
        pinky_bent  = p["pinky_tip"].y  > p["pinky_pip"].y

        return thumb_up and index_bent and middle_bent and ring_bent and pinky_bent

    right_landmarks = None
    left_landmarks  = None

    for hand_landmark, handness in zip(hand_landmarks, handness_list):
        label = handness.classification[0].label
        if label == "Right":
            if hand_burns_shape(hand_landmark):
                right_landmarks = hand_landmark
        else:
            if hand_burns_shape(hand_landmark):
                left_landmarks = hand_landmark

    if right_landmarks and left_landmarks:
        p_r = get_landmarks(right_landmarks)
        p_l = get_landmarks(left_landmarks)

        knuckles = ["index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"]
        knuckles_close = all(abs(p_r[k].x - p_l[k].x) < threshold for k in knuckles)

        return knuckles_close

    return False

#JJK Hands
#Gang Signs
#KAWAIII
#GRRRRR