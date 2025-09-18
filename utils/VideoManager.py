import cv2
from typing import List, Union
from utils.VideoSource import VideoSource


class VideoManager:
    def __init__(self, sources: List[Union[int, str]]) -> None:
        self.sources: List[VideoSource] = []
        self.valid_sources = []  # ← Nuevo: lista de fuentes válidas

        for i, source in enumerate(sources):
            self.sources.append(VideoSource(source, name=f"Source {i}"))

    def start_cameras(self) -> None:
        success_count = 0
        for source in self.sources:
            try:
                source.start()
                print(f"[OK] Started {source.name}")
                self.valid_sources.append(source)  # ← Solo agregar fuentes válidas
                success_count += 1
            except Exception as e:
                print(f"[ERROR] {source.name} -> {e}")
        
        if success_count == 0:
            print("[FATAL] No se pudo abrir ninguna fuente de video")
            exit(1)

    def get_frame(self, idx: int):
        if 0 <= idx < len(self.sources) and self.sources[idx] in self.valid_sources:
            return self.sources[idx].read()
        return None
    
    def stop_all(self) -> None:
        for source in self.sources:
            source.stop()
        cv2.destroyAllWindows()