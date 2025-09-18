import cv2 as cv
import numpy as np
import time
from typing import Optional, Dict
import numpy.typing as npt
from utils.VideoManager import VideoManager
from detector.Detector import Detector

from PIL import Image
import datetime

class Pipeline:
    def __init__(self, manager: VideoManager, detector: Detector, grid: bool = False) -> None:
        self.manager: VideoManager = manager
        self.detector: Detector = detector
        self.grid: bool = grid
        self.grid_size: tuple[int, int] = (400, 400)  # height, width per cell
        self.cols: int = 4  
        self.enable_detection: bool = True  # Detection ON by default


    def run(self) -> None:
        print("[Controls] q: quit, s: save frame")

        loop_fps: int = 30
        last_time: float = time.time()

        while True:
            frames_out: Dict[str, npt.NDArray[np.uint8]] = {}
            any_frame: bool = False

            # Capture frames from all sources
            for idx, source in enumerate(self.manager.sources):
                frame: Optional[npt.NDArray[np.uint8]] = self.manager.get_frame(idx)
                if frame is None:
                    print(f'frame not found, {frame}')
                    continue

                any_frame = True
                if self.detector and self.enable_detection:
                    frame = self.detector.annotate(frame)

                frames_out[source.name] = frame

            if not any_frame:
                break

            # Display frames
            if self.grid:
                self._show_grid(frames_out)
            else:
                for name, frame in frames_out.items():
                    cv.imshow(name, frame)

            key: int = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                # Guardar todos los frames disponibles
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                for name, frame in frames_out.items():
                    # Convertir BGR (OpenCV) a RGB (PIL)
                    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    filename = f"frame_{name}_{timestamp}.png"
                    pil_image.save(filename)
                    print(f"Frame saved: {filename}")
                continue
            if key == ord("d"):
                self.enable_detection = not self.enable_detection
                print(f"[INFO] Detection {'ENABLED' if self.enable_detection else 'DISABLED'}")

            # Limit global FPS
            elapsed: float = time.time() - last_time
            min_loop_duration: float = 1 / loop_fps
            if elapsed < min_loop_duration:
                time.sleep(min_loop_duration - elapsed)
            last_time = time.time()

        self.manager.stop_all()
        cv.destroyAllWindows()

    def _show_grid(self, frames: Dict[str, npt.NDArray[np.uint8]]) -> None:
        if not frames:
            return

        h, w = self.grid_size

        resized_frames: list[npt.NDArray[np.uint8]] = [
            np.asarray(cv.resize(frame, (w, h), interpolation=cv.INTER_LINEAR), dtype=np.uint8)
            for frame in frames.values()
        ]

        rows: int = (len(resized_frames) + self.cols - 1) // self.cols

        # Fill empty slots with black frames
        while len(resized_frames) < rows * self.cols:
            blank: npt.NDArray[np.uint8] = np.zeros((h, w, 3), dtype=np.uint8)
            resized_frames.append(blank)

        row_imgs: list[npt.NDArray[np.uint8]] = [
            np.hstack(resized_frames[i * self.cols:(i + 1) * self.cols])
            for i in range(rows)
        ]

        grid_frame: npt.NDArray[np.uint8] = np.vstack(row_imgs)
        cv.imshow("Grid", grid_frame)