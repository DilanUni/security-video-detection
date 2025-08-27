from ultralytics import YOLO

YOLO_CLASSES: list[int] = [0, 1, 2, 3, 43, 56, 59, 63, 64, 65, 66, 67, 73, 74, 76]
class Detector:
    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        self.model = YOLO(model_path)
        self.model.fuse()

    def detect(self, frame, conf: float = 0.4):
        return self.model(frame,
                          classes=YOLO_CLASSES,
                          conf=conf,
                          verbose=False)[0]

    def annotate(self, frame):
        return self.detect(frame).plot()
