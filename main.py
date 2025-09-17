from utils.VideoSourceHelper import VideoSourceHelper
from utils.VideoManager import VideoManager
from detector.Detector import Detector
from pipeline import Pipeline
from typing import Final

MODEL_PATH: Final[str] = "models/yolo11n.pt"
MAX_FPS: Final[int] = 15
VIDEO_FOLDER: Final[str] = "videos"

def main() -> None:
    # 1 Get all available video sources (cameras + video files)
    sources = VideoSourceHelper.get_all_sources(VIDEO_FOLDER)
    print(f"Sources found: {sources}")

    if not sources:
        print("[FATAL] No video sources found.")
        return

    # 2 Initialize VideoManager with sources
    manager = VideoManager(sources=sources, max_fps=MAX_FPS)

    # 3 Start cameras / videos
    manager.start_cameras()

    # 4 Initialize detector and pipeline
    detector = Detector(model_path=MODEL_PATH)
    pipeline = Pipeline(manager, detector, grid=True)

    # 5 Run the pipeline
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user.")
    finally:
        # 6 Stop all sources
        manager.stop_all()
        print("[INFO] All video sources stopped.")

if __name__ == "__main__":
    main()
