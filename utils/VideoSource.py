import cv2
import threading
import time
from typing import Optional, Union
import numpy as np

SourceType = Union[int, str]

class VideoSource:
    def __init__(self, source: SourceType, max_fps: int, name: Optional[str] = None) -> None:
        """
        Initialize a video source (camera or video file).
        """
        self.source: SourceType = source
        self.name: str = name or str(source)
        self.max_fps: int = max_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.actual_fps: Optional[float] = None

    def start(self) -> None:
        """
        Open the source and start reading frames in a separate thread.
        Raises RuntimeError if source cannot be opened or read.
        """
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Failed to open source: {self.source}")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.cap.release()
            raise RuntimeError(f"[{self.name}] Opened but cannot read frames")
        self.frame = frame

        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True, name=f"VideoThread-{self.name}")
        self.thread.start()

    def _update(self) -> None:
        delay = 1.0 / self.max_fps
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame
            time.sleep(delay)

        self.running = False
        if self.cap:
            self.cap.release()

    def read(self) -> Optional[np.ndarray]:
        """
        Returns a copy of the latest frame, or None if no frame is available.
        """
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        """
        Stop reading frames and release resources.
        """
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def restart(self) -> None:
        """
        Stop and restart the video source.
        """
        self.stop()
        self.start()
