from ultralytics import YOLO
import torch
import numpy as np
import numpy.typing as npt

YOLO_CLASSES: list[int] = [0]  # Only detect persons by default

class Detector:
    """
    YOLO Detector class for detecting objects in images or video frames.
    """

    def __init__(self, model_path: str = "models/yolo11n.pt") -> None:
        """
        Initialize the YOLO detector.

        Args:
            model_path (str): Path to the YOLO .pt model file.
        """
        # Detect device automatically: CUDA -> CPU
        if torch.cuda.is_available():
            self.device: str = "cuda:0"
        else:
            self.device: str = "cpu"

        # Use FP16 precision only on CUDA devices
        self.half: bool = self.device.startswith("cuda")

        # Load YOLO model
        self.model: YOLO = YOLO(model_path)
        self.model.fuse()  # Fuse Conv+BN layers for faster inference

        # Performance parameters
        self.max_det: int = 20  # Maximum detections per frame
        self.imgsz: int = 640   # Inference image size
        self.verbose: bool = False  # Disable YOLO internal prints

        print(f"[INFO] Using device: {self.device}, FP16: {self.half}")

    def _detect(
        self,
        frame: npt.NDArray[np.uint8],
        conf: float = 0.6
    ):
        """
        Run YOLO detection on a single frame.

        Args:
            frame (np.ndarray): Image frame (H x W x C) in uint8 format.
            conf (float): Minimum confidence threshold for detections.
        """
        results= self.model(
            frame,
            classes=YOLO_CLASSES,
            conf=conf,
            device=self.device,
            half=self.half,
            verbose=self.verbose,
            max_det=self.max_det,
            imgsz=self.imgsz
        )[0]

        return results

    def annotate(
        self,
        frame: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.uint8]:
        """
        Annotate frame with detected bounding boxes and class labels.

        Args:
            frame (np.ndarray): Image frame (H x W x C) in uint8 format.

        Returns:
            np.ndarray: Annotated image frame with drawn bounding boxes.
        """
        results = self._detect(frame=frame)

        if results.boxes is not None and len(results.boxes) > 0:
            print(f"[DETECTED] {len(results.boxes)} object(s)")

        # Plot returns the annotated frame
        annotated_frame: npt.NDArray[np.uint8] = results.plot()
        return annotated_frame
