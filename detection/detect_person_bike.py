from ultralytics import YOLO

class PersonBikeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image, conf=0.4, verbose=False)[0]

        persons = []
        motorcycles = []

        for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
            cls = int(cls)
            box = box.tolist()

            if cls == 0:      # person
                persons.append(box)
            elif cls == 3:    # motorcycle
                motorcycles.append(box)

        return persons, motorcycles
