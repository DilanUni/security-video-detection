from utils.VideoSourceHelper import VideoSourceHelper
from utils.VideoManager import VideoManager

from detector.Detector import Detector
from Pipeline import Pipeline
from typing import Final

MODEL_PATH: Final[str] = "models/yolo11n.pt"
VIDEO_FOLDER: Final[str] = "videos"

def main() -> None:
    # 1. Obtener todas las fuentes de video disponibles (cámaras + archivos)
    sources = VideoSourceHelper.get_all_sources(VIDEO_FOLDER)
    print(f"Sources found: {sources}")

    if not sources:
        print("[FATAL] No video sources found.")
        return

    # 2. Inicializar VideoManager con las fuentes
    video_manager: VideoManager = VideoManager(sources=sources,video_folder=VIDEO_FOLDER)

    # 3. Iniciar cámaras / videos
    video_manager.start_all()

    # 4. Inicializar detector
    detector = Detector(model_path=MODEL_PATH)

    # 5. Inicializar pipeline sin heatmap
    pipeline: Pipeline = Pipeline(
        manager=video_manager,
        detector=detector,
        grid=True
    )

    # 6. Ejecutar pipeline
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrumpido por el usuario.")
    finally:
        # 7. Detener todas las fuentes
        video_manager.stop_all()
        print("[INFO] Todas las fuentes de video detenidas.")

if __name__ == "__main__":
    main()
