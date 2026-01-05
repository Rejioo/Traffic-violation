from ultralytics import YOLO

class HelmetClassifier:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def has_helmet(self, image, person_box):
        x1, y1, x2, y2 = map(int, person_box)

        head_y2 = int(y1 + 0.45 * (y2 - y1))
        head = image[y1:head_y2, x1:x2]

        if head.size == 0:
            return False

        res = self.model(head, conf=0.25, verbose=False)[0]

        if res.boxes is None:
            return False

        # class 0 = helmet
        return any(int(c) == 0 for c in res.boxes.cls)
