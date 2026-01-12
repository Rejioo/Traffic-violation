from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path, conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        detections = self.model(frame)[0]
        plates = []

        for box in detections.boxes:
            score = float(box.conf)
            if score < self.conf_thresh:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plates.append({
                "bbox": (x1, y1, x2, y2),
                "score": round(score, 2)
            })

        return plates
