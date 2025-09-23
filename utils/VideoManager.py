import cv2
from typing import List, Union
from utils.VideoSource import VideoSource
from utils.VideoSourceHelper import VideoSourceHelper

SourceType = Union[int, str]

class VideoManager:
    """Manages multiple VideoSource objects and detects new videos or cameras dynamically."""

    def __init__(self, sources: List[SourceType], video_folder: str = "videos") -> None:
        self.video_folder: str = video_folder
        self.sources: List[VideoSource] = []

        # Initialize sources
        for i, src in enumerate(sources):
            self.sources.append(VideoSource(src, name=f"Source {i}"))

    def start_all(self) -> None:
        """Start all sources."""
        success_count = 0
        for source in self.sources:
            try:
                source.start()
                print(f"[OK] Started {source.name}")
                success_count += 1
            except Exception as e:
                print(f"[ERROR] {source.name} -> {e}")
        if success_count == 0:
            print("[FATAL] No video sources available.")
            exit(1)

    def get_active_sources(self) -> List[VideoSource]:
        """Return only active sources."""
        return [s for s in self.sources if s.is_active()]

    def add_new_source(self, source: SourceType, name: str) -> None:
        """Add a new video source dynamically."""
        try:
            new_source = VideoSource(source, name=name)
            new_source.start()
            self.sources.append(new_source)
            print(f"[INFO] Added new source '{name}'")
        except Exception as e:
            print(f"[ERROR] Could not add new source '{name}': {e}")

    def restart_sources(self) -> None:
        """
        Restart stopped sources and detect new ones automatically.
        """
        print("[INFO] Checking for stopped or new video sources...")

        # Detect all available sources
        detected_sources = VideoSourceHelper.get_all_sources(self.video_folder)

        # Add new sources that are not already managed
        existing_sources_set = {s.source for s in self.sources}
        for src in detected_sources:
            if src not in existing_sources_set:
                self.add_new_source(src, name=f"Source {len(self.sources)}")

        # Restart stopped sources
        for source in self.sources:
            if not source.is_active():
                try:
                    source.restart()
                    print(f"[OK] Restarted source '{source.name}'")
                except RuntimeError as e:
                    print(f"[ERROR] Could not restart '{source.name}': {e}")

    def stop_all(self) -> None:
        """Stop all video sources."""
        for source in self.sources:
            source.stop()
        cv2.destroyAllWindows()
