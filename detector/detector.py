from ultralytics import YOLO
import torch
import numpy as np
import numpy.typing as npt
import os
from typing import List, Final
from functools import lru_cache
import warnings

# Suppress unnecessary warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Global constants for YOLO model configuration
YOLO_CLASSES: Final[List[int]] = [0]  # Person class only for optimized detection
DEFAULT_MODEL_PATH: Final[str] = "models/yolo11n.pt"
MAX_DETECTIONS: Final[int] = 200
DEFAULT_CONFIDENCE: Final[float] = 0.50


class Detector:
    """
    Optimized YOLO object detector with resource-efficient configuration.
    
    This class provides high-performance object detection using YOLO models
    with automatic device selection and memory optimization techniques.
    
    Attributes:
        device (str): Computation device ('cuda:0' or 'cpu')
        half (bool): Whether to use FP16 precision for faster inference
        model (YOLO): Loaded YOLO model instance
        max_det (int): Maximum number of detections per frame
        verbose (bool): Whether to show verbose output
    """
    
    __slots__ = ('device', 'half', 'model', 'max_det', 'verbose')  # Memory optimization
    
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH) -> None:
        """
        Initialize the optimized YOLO detector.
        
        Args:
            model_path (str): Path to the YOLO .pt model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self._validate_model_path(model_path)
        self._setup_device()
        self._load_model(model_path)
        self._configure_parameters()
        
        print(f"[INFO] Detector initialized - Device: {self.device}, FP16: {self.half}")
    
    def _validate_model_path(self, model_path: str) -> None:
        """Validate that model file exists."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    def _setup_device(self) -> None:
        """Configure optimal computation device and precision."""
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.half: bool = self.device.startswith("cuda")
        
        # Optimize CUDA settings if available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    def _load_model(self, model_path: str) -> None:
        """Load and optimize YOLO model."""
        try:
            self.model: YOLO = YOLO(model_path)
            self.model.fuse()  # Fuse Conv+BN layers for 10-15% speed improvement
            
            # Move model to device and set precision
            if self.half:
                self.model.model.half()
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _configure_parameters(self) -> None:
        """Set optimized inference parameters."""
        self.max_det: int = MAX_DETECTIONS
        self.verbose: bool = False  # Reduce I/O overhead
    
    def _detect(self, frame: npt.NDArray[np.uint8], conf: float = DEFAULT_CONFIDENCE):
        """
        Execute optimized YOLO inference on input frame.
        
        Args:
            frame (npt.NDArray[np.uint8]): Input image array (H x W x C)
            conf (float): Confidence threshold for detections
            
        Returns:
            Detection results from YOLO model
            
        Note:
            Uses optimized parameters for maximum performance
        """
        # Use torch.no_grad() to save memory during inference
        with torch.no_grad():
            results = self.model(
                frame,
                classes=YOLO_CLASSES,  # Filter to specific classes only
                conf=conf,
                device=self.device,
                half=self.half,
                verbose=self.verbose,
                max_det=self.max_det,
                agnostic_nms=True,  # Faster NMS across all classes
            )[0]
        
        return results
    
    def annotate(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Generate annotated frame with detection results.
        
        Args:
            frame (npt.NDArray[np.uint8]): Input image frame (H x W x C)
            
        Returns:
            npt.NDArray[np.uint8]: Annotated frame with bounding boxes and labels
            
        Note:
            Only prints detection count if objects are found to reduce I/O overhead
        """
        results = self._detect(frame=frame)
        
        # Optimized detection logging - only print when detections exist
        if results.boxes is not None and len(results.boxes) > 0:
            print(f"[DETECTED] {len(results.boxes)} object(s)")
        
        # Generate annotated frame efficiently
        annotated_frame: npt.NDArray[np.uint8] = results.plot()
        return annotated_frame
