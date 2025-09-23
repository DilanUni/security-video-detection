import cv2
import threading
import time
import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Any

SourceType = Union[int, str]

class VideoSource:
    """Represents a single video source (camera or video file) with its own thread and active state."""

    def __init__(self, source: SourceType, name: Optional[str] = None) -> None:
        self.source: SourceType = source
        self.name: str = name or str(source)
        self.cap: cv2.VideoCapture = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Cannot open source {self.source}")

        fps: float = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps: float = fps if fps > 0 else 30.0

        self.frame: Optional[npt.NDArray[Any]] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.active: bool = False

    def start(self) -> None:
        """Start reading frames in a background thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True, name=f"VideoThread-{self.name}")
        self.thread.start()
        self.active = True

    def _update(self) -> None:
        delay: float = 1.0 / self.source_fps
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self.active = False
                print(f"[INFO] Source '{self.name}' stopped or disconnected.")
                break
            with self.lock:
                self.frame = frame
            time.sleep(delay)
        self.running = False

    def read(self) -> Optional[npt.NDArray[Any]]:
        """Return the latest frame if active."""
        if not self.active:
            return None
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
        self.active = False

    def restart(self) -> None:
        """Stop and reopen the source."""
        self.stop()
        self.cap = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.start()
        else:
            self.active = False
            raise RuntimeError(f"[{self.name}] Cannot restart source {self.source}")

    def is_active(self) -> bool:
        return self.active
