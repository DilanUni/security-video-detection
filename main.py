from camera.VideoManager import VideoManager
from detector.detector import Detector
from pipeline import Pipeline

MODEL: str = "models/yolo11n.pt"
MAX_FPS: int = 15

if __name__ == "__main__":
    sources: list = [
        0,
        "videos/a.mp4",
        "videos/video.mp4"
    ]

    manager: VideoManager = VideoManager(sources=sources, max_fps=MAX_FPS)
    detector: Detector = Detector(model_path=MODEL)
    pipeline: Pipeline = Pipeline(manager, detector)

    manager.start_cameras()
    pipeline.run()
