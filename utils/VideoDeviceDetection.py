import subprocess
import re
from typing import List, Tuple, Final
import os

import cv2

class VideoDeviceDetection:
    """
    Detects video capture devices (e.g., webcams) on Windows using FFmpeg.
    This class is limited to video sources only.
    """

    DIR = os.path.dirname(os.path.abspath(__file__))
    FFMPEG_PATH: Final[str] = os.path.join(DIR, "FFMPEG", "ffmpeg.exe")

    @classmethod
    def get_devices(cls) -> List[str]:
        """
        Returns a list of available video devices.
        If no devices are found, returns an empty list.
        """
        cmd = [
            cls.FFMPEG_PATH,
            "-list_devices", "true",
            "-f", "dshow",
            "-i", "dummy"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            output = result.stderr
        except subprocess.TimeoutExpired:
            print("Timeout: FFmpeg device detection took too long.")
            return []
        except FileNotFoundError:
            print(f"FFmpeg not found at: {cls.FFMPEG_PATH}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

        return cls._parse_output(output)

    @staticmethod
    def _parse_output(output: str) -> List[str]:
        """
        Extracts video device names from FFmpeg output.
        """
        pattern = r'\"(.+?)\".*\(video\)'
        return re.findall(pattern, output)

    @classmethod
    def get_device_map(cls, max_test: int = 10) -> List[Tuple[int, str]]:
        """
        Devuelve una lista de tuplas (índice_opencv, nombre_ffmpeg).
        Ejemplo: [(0, "Integrated Webcam"), (1, "USB Camera")]
        """
        names = cls.get_devices()
        result: List[Tuple[int, str]] = []

        # Probar cada índice de OpenCV
        for i in range(max_test):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    name = names[len(result)] if len(result) < len(names) else f"Camera {i}"
                    result.append((i, name))
                cap.release()
        return result
    
    @classmethod
    def has_devices(cls) -> Tuple[bool, str]:
        """
        Checks if there are available video devices.
        Returns (status, message).
        """
        devices = cls.get_devices()
        if not devices:
            return False, "No video devices detected."
        return True, f"Found {len(devices)} video device(s)."
