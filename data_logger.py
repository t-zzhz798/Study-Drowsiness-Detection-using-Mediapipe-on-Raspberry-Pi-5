import csv
import time
import os

LOG_FILE = "drowsiness_log.csv"
LOG_INTERVAL = 1.0  # seconds

# Create CSV with headers if it does not exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "EAR",
            "PERCLOS",
            "MAR",
            "Pitch",
            "Yaw",
            "Roll",
            "EAR_Drowsy",
            "PERCLOS_Drowsy",
            "MAR_Drowsy",
            "Head_Drowsy",
            "Recent_Yawns",
            "Calibrated"
        ])


def log_data(
    ear, perclos, mar, pitch, yaw, roll,
    ear_drowsy, perclos_drowsy, mar_drowsy, head_drowsy,
    recent_yawns, calibrated
):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            timestamp,
            round(ear, 3),
            round(perclos, 3),
            round(mar, 3),
            round(pitch, 3),
            round(yaw, 3),
            round(roll, 3),
            int(ear_drowsy),
            int(perclos_drowsy),
            int(mar_drowsy),
            int(head_drowsy),
            recent_yawns,
            int(calibrated)
        ])