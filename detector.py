import cv2
import time
import subprocess
import threading

from ultralytics import YOLO

# YOUTUBE_URLx = "https://www.youtube.com/watch?v=alFU410jmgE"
YOUTUBE_URL = "https://www.youtube.com/watch?v=JJ3MWNYVCU4"

MODEL_PATH = "models/best_vehicle_seg.pt"

CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]

latest_frame = None

vehicle_count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

model = YOLO(MODEL_PATH)


def get_stream_url():

    cmd = ["yt-dlp", "-f", "best", "-g", YOUTUBE_URL]

    result = subprocess.check_output(cmd)

    return result.decode().strip()


def process_stream():

    global latest_frame
    global vehicle_count

    while True:

        try:

            print("Getting stream URL...")

            stream_url = get_stream_url()

            print("Stream URL acquired")

            cap = cv2.VideoCapture(stream_url)

            if not cap.isOpened():

                print("Failed open stream")

                time.sleep(5)

                continue

            while True:

                success, frame = cap.read()

                if not success:

                    print("Frame lost")

                    break

                results = model.predict(
                    source=frame, imgsz=640, conf=0.35, verbose=False
                )

                result = results[0]

                counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

                if result.boxes is not None:

                    for box in result.boxes:

                        cls = int(box.cls[0])

                        if cls < len(CLASS_NAMES):

                            counts[CLASS_NAMES[cls]] += 1

                vehicle_count = counts

                latest_frame = result.plot()

        except Exception as e:

            print("Detector Error:", e)

            time.sleep(5)


def start_detector():

    thread = threading.Thread(target=process_stream, daemon=True)

    thread.start()
