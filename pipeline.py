import cv2 as cv
import numpy as np
import time
import datetime
from typing import Optional, Dict
import numpy.typing as npt
from utils.VideoManager import VideoManager
from detector.Detector import Detector
from PIL import Image

class Pipeline:
    """Handles multiple video sources with optional detection and dynamic restart."""

    def __init__(self, manager: VideoManager, detector: Optional[Detector] = None, grid: bool = False) -> None:
        self.manager: VideoManager = manager
        self.detector: Optional[Detector] = detector
        self.grid: bool = grid
        self.grid_size: tuple[int, int] = (400, 400)
        self.cols: int = 4
        self.enable_detection: bool = True

    def run(self) -> None:
        print("[Controls] q: quit, s: save frame, d: toggle detection, r: restart sources")

        loop_fps: int = 30
        last_time: float = time.time()

        while True:
            frames_out: Dict[str, npt.NDArray[np.uint8]] = {}
            any_frame: bool = False

            active_sources = self.manager.get_active_sources()
            for source in active_sources:
                frame: Optional[npt.NDArray[np.uint8]] = source.read()
                if frame is None:
                    continue

                any_frame = True
                if self.detector and self.enable_detection:
                    frame = self.detector.annotate(frame)

                frames_out[source.name] = frame

            if not any_frame and not self.grid:
                print("[INFO] No active video sources remain. Exiting.")
                break

            if self.grid:
                self._show_grid(frames_out)
            else:
                for name, frame in frames_out.items():
                    cv.imshow(name, frame)

            key: int = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                self._save_frames(frames_out)
            elif key == ord("d"):
                self.enable_detection = not self.enable_detection
                print(f"[INFO] Detection {'ENABLED' if self.enable_detection else 'DISABLED'}")
            elif key == ord("r"):
                # Dynamic restart
                self.manager.restart_sources()

            elapsed: float = time.time() - last_time
            min_loop_duration: float = 1 / loop_fps
            if elapsed < min_loop_duration:
                time.sleep(min_loop_duration - elapsed)
            last_time = time.time()

        self.manager.stop_all()
        cv.destroyAllWindows()

    def _show_grid(self, frames: Dict[str, npt.NDArray[np.uint8]]) -> None:
        if not frames:
            empty_grid = np.zeros((self.grid_size[0], self.cols * self.grid_size[1], 3), dtype=np.uint8)
            cv.imshow("Grid", empty_grid)
            return

        h, w = self.grid_size
        resized_frames: list[npt.NDArray[np.uint8]] = [
            cv.resize(frame, (w, h), interpolation=cv.INTER_LINEAR)
            for frame in frames.values()
        ]
        rows: int = (len(resized_frames) + self.cols - 1) // self.cols
        while len(resized_frames) < rows * self.cols:
            resized_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        row_imgs: list[npt.NDArray[np.uint8]] = [
            np.hstack(resized_frames[i * self.cols:(i + 1) * self.cols]) for i in range(rows)
        ]
        grid_frame: npt.NDArray[np.uint8] = np.vstack(row_imgs)
        cv.imshow("Grid", grid_frame)

    def _save_frames(self, frames: Dict[str, npt.NDArray[np.uint8]]) -> None:
        timestamp: str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for name, frame in frames.items():
            rgb_frame: npt.NDArray[np.uint8] = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            Image.fromarray(rgb_frame).save(f"frame_{name}_{timestamp}.png")
            print(f"[INFO] Frame saved: frame_{name}_{timestamp}.png")
