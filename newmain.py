import cv2
import os
from collections import defaultdict, deque

from detection.detect_person_bike import PersonBikeDetector
from helmet.helmet_classifier import HelmetClassifier
from violation.passenger_violation import analyze_passengers
from utils.drawing import draw_results

from numberplate.plate_detector import PlateDetector
from numberplate.plate_ocr import PlateOCR
from numberplate.plate_logger import PlateLogger


# ================= CONFIG =================
# VIDEO_PATH = "hemltest/a3.MOV"
VIDEO_PATH = "r3.mp4"
OUT_VIDEO = "a3vanlpn.mp4"

HELMET_HISTORY = 5
HELMET_VOTE_THRESH = 3


# ================= INIT =================
detector = PersonBikeDetector("models/yolov8n.pt")
helmet_model = HelmetClassifier("models/vanhelmet.pt")

# number plate
plate_detector = PlateDetector("models/lpn.pt", conf_thresh=0.5)
plate_ocr = PlateOCR()
plate_logger = PlateLogger()

cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), "Could not open video"

fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)

helmet_memory = defaultdict(lambda: deque(maxlen=HELMET_HISTORY))
frame_idx = 0


# ================= PROCESS =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    persons, bikes = detector.detect(frame)

    helmet_map = {}

    # ---------- HELMET TEMPORAL LOGIC ----------
    for i, p in enumerate(persons):
        has_helmet = helmet_model.has_helmet(frame, p)
        helmet_memory[i].append(has_helmet)

        helmet_votes = sum(helmet_memory[i])
        helmet_map[i] = helmet_votes >= HELMET_VOTE_THRESH

    # ---------- PASSENGER ANALYSIS ----------
    bike_info = analyze_passengers(persons, bikes)

    # ---------- NUMBER PLATE DETECTION ----------
    plates = plate_detector.detect(frame)

    for plate in plates:
        x1, y1, x2, y2 = plate["bbox"]
        cropped = frame[y1:y2, x1:x2]

        text = plate_ocr.read(cropped)
        if text:
            plate_logger.log(text, plate["bbox"], plate["score"])

            # draw plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
            )

    # ---------- DRAW RIDER + BIKE RESULTS ----------
    out = draw_results(frame, persons, helmet_map, bike_info)
    writer.write(out)

    if frame_idx % 10 == 0:
        print(f"Processed frame {frame_idx}")

# ================= CLEANUP =================
cap.release()
writer.release()

plate_logger.export("data/plates.csv")

print("DONE. Saved:", OUT_VIDEO)
print("Plate log saved to data/plates.csv")
