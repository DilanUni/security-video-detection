import cv2 as cv
import numpy as np
from camera.VideoManager import VideoManager
from detector.detector import Detector


class Pipeline:
    def __init__(self, manager: VideoManager, detector: Detector, grid: bool = False) -> None:
        self.manager: VideoManager = manager
        self.detector: Detector = detector
        self.grid: bool = grid

    def run(self) -> None:
        print("[Controls] q: quit")

        while True:
            frames_out = {}
            any_frame = False

            for i, source in enumerate(self.manager.sources):
                frame: np.ndarray = self.manager.get_frame(i)
                if frame is None:
                    continue

                any_frame = True
                if self.detector:
                    frame = self.detector.annotate(frame)

                frames_out[source.name] = frame  # ← corregido

            if frames_out:
                if self.grid:
                    self._show_grid(frames_out)
                else:
                    for name, frame in frames_out.items():
                        cv.imshow(name, frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if not any_frame:
                break

        self.manager.stop_all()
        cv.destroyAllWindows()

    def _show_grid(self, frames: dict) -> None:
        if not frames:
            return

        h, w = 240, 320
        resized = [cv.resize(f, (w, h)) for f in frames.values()]
        cols = 2
        rows = (len(resized) + cols - 1) // cols

        while len(resized) < rows * cols:
            resized.append(np.zeros((h, w, 3), dtype=np.uint8))

        row_imgs = [np.hstack(resized[i * cols:(i + 1) * cols]) for i in range(rows)]
        grid = np.vstack(row_imgs)
        cv.imshow("Grid", grid)
