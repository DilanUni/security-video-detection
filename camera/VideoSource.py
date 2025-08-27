import cv2
import threading
import time
from typing import Optional
import numpy as np


class VideoSource:
    def __init__(self, source: int | str, max_fps: int, name: str = "Source") -> None:
        self.source: int | str = source
        self.name: str = name
        self.max_fps: int = max_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.running: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self) -> None:
        # Abrir fuente (archivo o cámara)
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] Failed to open source: {self.source}")

        # Leer un frame de prueba
        ret, test_frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"[{self.name}] Opened but cannot read frames")

        # Guardar primer frame
        self.frame = test_frame

        # Lanzar hilo de captura
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self) -> None:
        delay: float = 1.0 / self.max_fps
        while self.running and self.cap:
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
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap:
            self.cap.release()
