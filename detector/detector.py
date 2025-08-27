from ultralytics import YOLO
import numpy as np


class Detector:
    def __init__(self, model_path: str = "yolo11n.pt", device: str | None = None) -> None:
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

    def detect(self, frame: np.ndarray):
        """Return raw YOLO results (only persons)."""
        results = self.model(frame, classes=[0], verbose=False)
        return results[0]

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with YOLO annotations (only persons)."""
        results = self.detect(frame)
        return results.plot()
