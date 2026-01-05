# import cv2
# import os

# from detection.detect_person_bike import PersonBikeDetector
# from helmet.helmet_classifier import HelmetClassifier
# from violation.passenger_violation import analyze_passengers
# from utils.drawing import draw_results

# IMAGE_DIR = "data/images"
# OUT_DIR = "outputs"
# os.makedirs(OUT_DIR, exist_ok=True)

# detector = PersonBikeDetector("models/yolov8n.pt")
# helmet_model = HelmetClassifier("models/newg.pt")

# for img_name in os.listdir(IMAGE_DIR):
#     if not img_name.lower().endswith((".jpg",".png",".jpeg")):
#         continue

#     img_path = os.path.join(IMAGE_DIR, img_name)
#     img = cv2.imread(img_path)
#     if img is None:
#         continue

#     persons, bikes = detector.detect(img)

#     helmet_map = {}
#     for i, p in enumerate(persons):
#         helmet_map[i] = helmet_model.has_helmet(img, p)

#     bike_info = analyze_passengers(persons, bikes)

#     out = draw_results(img, persons, helmet_map, bike_info)
#     cv2.imwrite(os.path.join(OUT_DIR, img_name), out)

#     print(f"Processed {img_name}")
#-------------------------------------------------------------------------
import cv2
import os
from collections import defaultdict, deque

from detection.detect_person_bike import PersonBikeDetector
from helmet.helmet_classifier import HelmetClassifier
from violation.passenger_violation import analyze_passengers
from utils.drawing import draw_results

# ================= CONFIG =================
VIDEO_PATH = "r3.mp4"
OUT_VIDEO = "output_video.mp4"

HELMET_HISTORY = 5        # frames to remember
HELMET_VOTE_THRESH = 3    # majority vote

# ================= INIT =================
detector = PersonBikeDetector("models/yolov8n.pt")
helmet_model = HelmetClassifier("models/newg.pt")

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

# person_id -> helmet history
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

        # majority vote
        helmet_votes = sum(helmet_memory[i])
        helmet_map[i] = helmet_votes >= HELMET_VOTE_THRESH

    # ---------- PASSENGER ANALYSIS ----------
    bike_info = analyze_passengers(persons, bikes)

    # ---------- DRAW ----------
    out = draw_results(frame, persons, helmet_map, bike_info)

    writer.write(out)

    if frame_idx % 10 == 0:
        print(f"Processed frame {frame_idx}")

cap.release()
writer.release()

print("DONE. Saved:", OUT_VIDEO)
