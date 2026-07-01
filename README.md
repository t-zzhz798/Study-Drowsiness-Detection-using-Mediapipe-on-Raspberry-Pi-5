# Study-Drowsiness-Detection-using-Mediapipe-on-Raspberry-Pi-5

## Overview

This project implements a **real-time student study drowsiness detection system** using **MediaPipe** and computer vision on a **Raspberry Pi 5**.

It monitors facial features such as **eye closure, blink rate, and head posture** to detect signs of fatigue, which it then uses to trigger alerts.

---

## Description

* Calculates Eye aspect ratio (EAR)
* Uses Real-time facial landmark tracking using MediaPipe
* Audio alerts for warning and critical states
* Logs drowsiness events (`drowsiness_log.csv`)
* To be run on Raspberry Pi 5 with camera module 3.  
* Use the Bookworm (Debian 12, 64-bit) Raspberry Pi OS

---
## Things you can modify

* Update Drowsiness Detection Tresholds in drowsiness_state folder
* Notification audios in static folder
* Displays on live stream in process_drowsiness folder
---

## Project Structure

```
.
├── stream.py                 # Runs the stream
├── process_drowsiness.py     # Core processing logic
├── drowsiness_math.py        # Calculations (EAR, thresholds)
├── drowsiness_state.py       # State management
├── data_logger.py            # Logging system
├── static/                   # Audio alert files
│   ├── warning.mp3
│   └── critical.mp3
└── drowsiness_log.csv        # Output log file
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/t-zzhz798/Study-Drowsiness-Detection-using-Mediapipe-on-Raspberry-Pi-5.git
cd Study-Drowsiness-Detection-using-Mediapipe-on-Raspberry-Pi-5
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the main script:

```bash
python stream.py
```

The system will:

* Start camera input
* Track facial landmarks to detect posture, facial gestures
* Detect drowsiness in real time
* Trigger audio alerts when needed

---