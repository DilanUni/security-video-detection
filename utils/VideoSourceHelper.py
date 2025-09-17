from typing import List, Union
import os
from utils.VideoDeviceDetection import VideoDeviceDetection

SourceType = Union[int, str]

class VideoSourceHelper:
    @staticmethod
    def get_camera_sources() -> List[int]:
        device_map = VideoDeviceDetection.get_device_map()
        return [idx for idx, _ in device_map]

    @staticmethod
    def get_video_files(folder: str, extension: str = ".mp4") -> List[str]:
        if not os.path.isdir(folder):
            return []
        return [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(extension.lower())
        ]

    @staticmethod
    def get_all_sources(video_folder: str = "videos") -> List[SourceType]:
        cameras = VideoSourceHelper.get_camera_sources()
        videos = VideoSourceHelper.get_video_files(video_folder)
        return cameras + videos
