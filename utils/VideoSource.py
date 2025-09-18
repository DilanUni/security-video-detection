import cv2
import threading
import time
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Any

SourceType = Union[int, str]

class VideoSource:
    def __init__(self, source: SourceType, name: Optional[str] = None) -> None:
        """
        Represent a single video source (camera or file).
        """
        self.source: SourceType = source
        self.name: str = name or str(source)
        self.cap: cv2.VideoCapture = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Cannot open source {self.source}")

        # Query source FPS if available
        fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps: float = fps if fps > 0 else 30.0  # default if unknown

        # Use a more flexible type annotation for OpenCV frames
        self.frame: Optional[npt.NDArray[Any]] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self) -> None:
        """
        Start reading frames in a background thread.
        """
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True, name=f"VideoThread-{self.name}")
        self.thread.start()

    def _update(self) -> None:
        delay: float = 1.0 / self.source_fps
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame
            time.sleep(delay)

        self.running = False

    def read(self) -> Optional[npt.NDArray[Any]]:
        """
        Return the latest available frame (a copy), or None if not ready.
        """
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        """
        Stop reading frames and release the capture.
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

    def restart(self) -> None:
        """
        Stop and restart the video source.
        """
        self.stop()
        self.start()