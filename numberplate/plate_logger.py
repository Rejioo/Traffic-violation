from datetime import datetime

class PlateLogger:
    def __init__(self):
        self.seen = set()
        self.logs = []

    def log(self, plate_text, bbox, score):
        if plate_text in self.seen:
            return

        self.seen.add(plate_text)
        x1, y1, x2, y2 = bbox

        self.logs.append({
            "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Confidence": score,
            "DetectedText": plate_text
        })

    def export(self, csv_path):
        import pandas as pd
        pd.DataFrame(self.logs).to_csv(csv_path, index=False)
