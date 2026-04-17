# Study-Drowsiness-Detection-using-Mediapipe-on-Raspberry-Pi-5
# 🚗 Driver Drowsiness Detection using MediaPipe (Raspberry Pi 5)

## 📌 Overview

This project implements a **real-time student study drowsiness detection system** using **MediaPipe** and computer vision on a **Raspberry Pi 5**.

It monitors facial features such as **eye closure, blink rate, and head posture** to detect signs of fatigue and trigger alerts.

---

## ⚙️ Features

* 👁️ Eye aspect ratio (EAR) based drowsiness detection
* 🧠 Real-time facial landmark tracking using MediaPipe
* 🔊 Audio alerts for warning and critical states
* 📊 Logging of drowsiness events (`drowsiness_log.csv`)
* ⚡ Optimised to run on Raspberry Pi 5

---

## 🗂️ Project Structure

```
.
├── stream.py                  # Main entry point
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

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/t-zzhz798/Study-Drowsiness-Detection-using-Mediapipe-on-Raspberry-Pi-5.git
cd Study-Drowsiness-Detection-using-Mediapipe-on-Raspberry-Pi-5
```

### 2. Create virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script:

```bash
python stream.py
```

The system will:

* Start camera input
* Track facial landmarks
* Detect drowsiness in real time
* Trigger audio alerts when needed

---

## 📊 How It Works

1. MediaPipe detects facial landmarks
2. Eye Aspect Ratio (EAR) is calculated
3. Threshold-based logic determines:

   * 🟡 Warning
   * 🔴 Critical
4. Audio alerts are triggered accordingly
5. Events are logged for analysis

---

## 📈 Output

* Real-time alerts via audio
* Log file: `drowsiness_log.csv`

---

## 🚀 Future Improvements

* Add gesture recognition integration
* Improve accuracy with ML models
* Add web dashboard / UI
* Support multiple camera inputs

---

## 🧠 Tech Stack

* Python
* OpenCV
* MediaPipe
* Raspberry Pi 5

---

## ⚠️ Disclaimer

This project is for **educational and experimental purposes only** and should not be relied upon as a sole safety system in real-world driving conditions.

---

## 👤 Author

Timo
