from ultralytics import YOLO

YOLO_CLASSES: list[int] = [0, 44] # Person, knife
class Detector:
    def __init__(self, model_path: str = "yolo11n.pt") -> None:
        self.model = YOLO(model_path)
        self.model.fuse()

    def detect(self, frame, conf: float = 0.4):
        return self.model(frame,
                          classes=YOLO_CLASSES,
                          conf=conf,
                          verbose=True)[0]

    def annotate(self, frame):
        return self.detect(frame).plot()
