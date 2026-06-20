import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import os
import urllib.request
import wave
import struct
import math

# =====================================
# DOWNLOAD HAND MODEL
# =====================================

model_path = "hand_landmarker.task"

if not os.path.exists(model_path):
    print("Downloading MediaPipe Hand Model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded successfully.")

# =====================================
# CREATE MUSIC NOTES AUTOMATICALLY
# =====================================

pygame.mixer.init()

notes_freq = {
    "C": 261.63,
    "D": 293.66,
    "E": 329.63,
    "F": 349.23,
    "G": 392.00,
    "A": 440.00,
    "B": 493.88
}

sounds = {}

def generate_tone(filename, freq):
    sample_rate = 44100
    duration = 0.3
    amplitude = 16000

    wav_file = wave.open(filename, "w")
    wav_file.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))

    for i in range(int(sample_rate * duration)):
        value = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        data = struct.pack("<h", value)
        wav_file.writeframesraw(data)

    wav_file.close()

for note, freq in notes_freq.items():
    file_name = f"{note}.wav"

    if not os.path.exists(file_name):
        generate_tone(file_name, freq)

    sounds[note] = pygame.mixer.Sound(file_name)

# =====================================
# MEDIAPIPE SETUP
# =====================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    num_hands=1,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================
# CAMERA SETUP
# =====================================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =====================================
# PIANO KEYS
# x, y, width, height, note
# =====================================

keys = [
    (0, 550, 180, 150, "C"),
    (180, 550, 180, 150, "D"),
    (360, 550, 180, 150, "E"),
    (540, 550, 180, 150, "F"),
    (720, 550, 180, 150, "G"),
    (900, 550, 180, 150, "A"),
    (1080, 550, 180, 150, "B")
]

last_note = ""
last_time = 0

print("AI Virtual Piano Started")
print("Move index finger over piano keys")
print("Press Q to Exit")

# =====================================
# MAIN LOOP
# =====================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            print("Camera not found.")
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        results = landmarker.detect(mp_image)

        active_note = None
        index_x = None
        index_y = None

        # =====================================
        # HAND DETECTION
        # =====================================

        if results.hand_landmarks:

            hand = results.hand_landmarks[0]

            h_frame, w_frame, _ = frame.shape

            # Draw all hand landmarks
            for lm in hand:
                px = int(lm.x * w_frame)
                py = int(lm.y * h_frame)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

            # Index finger tip landmark number 8
            tip = hand[8]

            index_x = int(tip.x * w_frame)
            index_y = int(tip.y * h_frame)

            # Check which key finger is touching
            for x, y, w, h, note in keys:
                if x < index_x < x + w and y < index_y < y + h:
                    active_note = note

                    current = time.time()

                    if note != last_note or current - last_time > 0.25:
                        sounds[note].play()
                        last_note = note
                        last_time = current

        # =====================================
        # DRAW PIANO KEYS
        # =====================================

        for x, y, w, h, note in keys:

            if note == active_note:
                color = (0, 255, 0)
            else:
                color = (255, 255, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

            cv2.putText(
                frame,
                note,
                (x + 70, y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                3
            )

        # =====================================
        # DRAW FINGER EFFECTS
        # =====================================

        if index_x is not None and index_y is not None:

            # Finger glow
            cv2.circle(frame, (index_x, index_y), 25, (0, 255, 255), -1)
            cv2.circle(frame, (index_x, index_y), 40, (255, 255, 255), 2)

            # Music symbols
            for i in range(5):
                cv2.putText(
                    frame,
                    "♪",
                    (index_x + i * 15, index_y - i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2
                )

        # =====================================
        # TITLE TEXT
        # =====================================

        cv2.putText(
            frame,
            "AI Hand Gesture Virtual Piano",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        cv2.putText(
            frame,
            "Press Q to Exit",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Virtual Piano", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
pygame.quit()