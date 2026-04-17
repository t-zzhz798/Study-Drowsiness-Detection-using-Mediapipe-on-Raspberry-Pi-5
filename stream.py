import os
import signal
import threading
import time

from flask import Flask, Response, jsonify
from picamera2 import Picamera2

import process_drowsiness

app = Flask(__name__)

FRAME_SIZE = (640, 480)
HOST = "0.0.0.0"
PORT = 5000

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": FRAME_SIZE}
    )
)
picam2.start()

lock = threading.Lock()
output_frame_container = [None]


def generate():
    while True:
        with lock:
            frame = output_frame_container[0]

        if frame is None:
            time.sleep(0.01)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(0.01)


def get_status_payload():
    monitor = process_drowsiness.monitor
    calibrated = monitor.calibrated
    is_calibrating = monitor.is_calibrating

    if is_calibrating:
        status_text = "Calibrating"
        calibration_label = "Calibrating..."
        calibration_disabled = True
        calibration_color = "#999999"
    elif calibrated:
        status_text = "Calibrated"
        calibration_label = "Recalibrate"
        calibration_disabled = False
        calibration_color = "#2e8b57"
    else:
        status_text = "Not Calibrated"
        calibration_label = "Start Calibration"
        calibration_disabled = False
        calibration_color = "#007bff"

    return {
        "calibrated": calibrated,
        "is_calibrating": is_calibrating,
        "status_text": status_text,
        "calibration_label": calibration_label,
        "calibration_disabled": calibration_disabled,
        "calibration_color": calibration_color,
        "reset_disabled": not (calibrated or is_calibrating),
        "alert_state": process_drowsiness.current_state,
    }


@app.route("/video_feed")
def video_feed():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    return jsonify(get_status_payload())


@app.route("/calibration", methods=["POST"])
def calibration():
    if not process_drowsiness.monitor.is_calibrating:
        process_drowsiness.monitor.start_calibration()
    return ("", 204)


@app.route("/reset_calibration", methods=["POST"])
def reset_calibration():
    process_drowsiness.monitor.reset_calibration()
    return ("", 204)


@app.route("/stop_server", methods=["POST"])
def stop_server():
    def shutdown():
        time.sleep(0.2)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=shutdown, daemon=True).start()
    return ("", 204)


@app.route("/")
def index():
    state = get_status_payload()

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Drowsiness Monitor</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f4f4f4;
                margin: 0;
                padding: 20px;
            }}

            h2 {{
                margin-bottom: 20px;
            }}

            .buttons  {{
                margin-bottom: 20px;
            }}

            button {{
                font-size: 18px;
                padding: 12px 24px;
                margin: 8px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                color: white;
            }}

            .sound-btn {{
                background-color: #ff9800;
            }}

            .reset-btn {{
                background-color: #dc3545;
            }}

            .stop-btn {{
                background-color: #6c757d;
            }}

            .disabled-btn {{
                opacity: 0.6;
                cursor: not-allowed;
            }}

            img {{
                max-width: 100%;
                border: 3px solid #333;
                border-radius: 10px;
            }}

            .status-box {{
                margin: 15px auto 20px auto;
                padding: 12px 18px;
                width: fit-content;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.12);
                font-size: 18px;
            }}
        </style>
    </head>
    <body>
        <h2>Drowsiness Monitor</h2>

        <div class="status-box" id="statusBox">
            <strong>Status:</strong> {state["status_text"]} |
            <strong>Alert:</strong> {state["alert_state"]}
        </div>

        <div class="buttons ">
            <button id="enableSoundBtn" class="sound-btn">Enable Sound</button>

            <button id="calibrationBtn"
                    style="background-color:{state["calibration_color"]};"
                    {"disabled" if state["calibration_disabled"] else ""}
                    class="{"disabled-btn" if state["calibration_disabled"] else ""}">
                {state["calibration_label"]}
            </button>

            <button id="resetBtn"
                    class="reset-btn {"disabled-btn" if state["reset_disabled"] else ""}"
                    {"disabled" if state["reset_disabled"] else ""}>
                Reset
            </button>

            <button id="stopBtn" class="stop-btn">Terminate</button>
        </div>

        <audio id="warningSound" src="/static/warning.mp3" preload="auto"></audio>
        <audio id="criticalSound" src="/static/critical.mp3" preload="auto"></audio>

        <img src="/video_feed" alt="Video feed">

        <script>
            const statusBox = document.getElementById("statusBox");
            const calibrationBtn = document.getElementById("calibrationBtn");
            const resetBtn = document.getElementById("resetBtn");
            const stopBtn = document.getElementById("stopBtn");
            const enableSoundBtn = document.getElementById("enableSoundBtn");

            const warningSound = document.getElementById("warningSound");
            const criticalSound = document.getElementById("criticalSound");

            let soundEnabled = false;
            let lastAlertState = "{state["alert_state"]}";

            function setDisabled(button, disabled) {{
                button.disabled = disabled;
                button.classList.toggle("disabled-btn", disabled);
            }}

            async function playAlertSound(alertState) {{
                if (alertState === "WARNING") {{
                    warningSound.pause();
                    warningSound.currentTime = 0;
                    await warningSound.play();
                }} else if (alertState === "CRITICAL") {{
                    criticalSound.pause();
                    criticalSound.currentTime = 0;
                    await criticalSound.play();
                }}
            }}

            enableSoundBtn.addEventListener("click", async () => {{
                try {{
                    soundEnabled = true;
                    enableSoundBtn.textContent = "Sound Enabled";

                    warningSound.volume = 1.0;
                    criticalSound.volume = 1.0;

                    await warningSound.play();
                    warningSound.pause();
                    warningSound.currentTime = 0;

                    await criticalSound.play();
                    criticalSound.pause();
                    criticalSound.currentTime = 0;

                    lastAlertState = "NOT READY YET";

                    const response = await fetch("/status");
                    const data = await response.json();

                    await playAlertSound(data.alert_state);
                    lastAlertState = data.alert_state;
                }} catch (err) {{
                    console.error("Audio enable failed:", err);
                }}
            }});

            async function refreshUI() {{
                try {{
                    const response = await fetch("/status");
                    const data = await response.json();

                    statusBox.innerHTML =
                        `<strong>Status:</strong> ${{data.status_text}} | <strong>Alert:</strong> ${{data.alert_state}}`;

                    calibrationBtn.textContent = data.calibration_label;
                    calibrationBtn.style.backgroundColor = data.calibration_color;
                    setDisabled(calibrationBtn, data.calibration_disabled);
                    setDisabled(resetBtn, data.reset_disabled);

                    if (soundEnabled && data.alert_state !== lastAlertState) {{
                        await playAlertSound(data.alert_state);
                    }}

                    lastAlertState = data.alert_state;
                }} catch (err) {{
                    console.error("Failed to refresh UI:", err);
                }}
            }}

            calibrationBtn.addEventListener("click", async () => {{
                try {{
                    await fetch("/calibration", {{ method: "POST" }});
                    await refreshUI();
                }} catch (err) {{
                    console.error("Calibration request failed:", err);
                }}
            }});

            resetBtn.addEventListener("click", async () => {{
                try {{
                    await fetch("/reset_calibration", {{ method: "POST" }});
                    await refreshUI();
                }} catch (err) {{
                    console.error("Reset request failed:", err);
                }}
            }});

            stopBtn.addEventListener("click", async () => {{
                try {{
                    await fetch("/stop_server", {{ method: "POST" }});
                }} catch (err) {{
                    console.error("Stop server request failed:", err);
                }}
            }});

            setInterval(refreshUI, 1000);
            refreshUI();
        </script>
    </body>
    </html>
    """


if __name__ == "__main__":
    t = threading.Thread(
        target=process_drowsiness.process_frames,
        args=(picam2, lock, output_frame_container),
        daemon=True
    )
    t.start()

    app.run(host=HOST, port=PORT, threaded=True)