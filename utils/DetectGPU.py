import winreg
from typing import Optional, Final, Dict

import winreg
from functools import lru_cache

class DetectGPU:
    """
    Optimized GPU vendor detection with caching for Windows systems.
    
    Provides hardware-accelerated codec selection based on available GPU vendor.
    Uses registry-based detection with LRU caching for improved performance.
    
    Class Constants:
        NVIDIA_REG_PATH: Registry path for NVIDIA detection
        AMD_REG_PATH: Registry path for AMD detection
    """
    
    # Registry paths as class constants for better maintainability
    NVIDIA_REG_PATH: Final[str] = r"SOFTWARE\NVIDIA Corporation\Global\NvControlPanel2"
    AMD_REG_PATH: Final[str] = r"SOFTWARE\AMD"
    
    # Codec mapping for optimal hardware acceleration
    _CODEC_MAP: Final[Dict[str, str]] = {
        "nvidia": "hevc_nvenc",    # H.265 NVIDIA NVENC encoder
        "amd": "hevc_amf",         # H.265 AMD AMF encoder  
        "cpu": "libx265"           # H.265 software encoder
    }
    
    @staticmethod
    @lru_cache(maxsize=1)  # Cache result since hardware doesn't change during runtime
    def detect_gpu_vendor() -> str:
        """
        Detect GPU vendor through Windows registry analysis.
        
        Returns:
            str: GPU vendor identifier ('nvidia', 'amd', or 'cpu')
            
        Note:
            Result is cached for performance as hardware configuration is static
        """
        # Check NVIDIA first (more common for ML workloads)
        if DetectGPU._check_registry_key(DetectGPU.NVIDIA_REG_PATH):
            return "nvidia"
        
        # Check AMD as fallback
        if DetectGPU._check_registry_key(DetectGPU.AMD_REG_PATH):
            return "amd"
        
        print("[INFO] No dedicated GPU detected, falling back to CPU")
        return "cpu"
    
    @staticmethod
    def _check_registry_key(registry_path: str) -> bool:
        """
        Check if registry key exists (helper method).
        
        Args:
            registry_path (str): Windows registry path to check
            
        Returns:
            bool: True if registry key exists, False otherwise
        """
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, registry_path):
                return True
        except (FileNotFoundError, OSError):
            return False
    
    @staticmethod
    def get_optimal_codec(vendor: Optional[str] = None) -> str:
        """
        Get optimal FFmpeg codec based on hardware capabilities.
        
        Args:
            vendor (Optional[str]): GPU vendor override. Auto-detects if None
            
        Returns:
            str: Optimal FFmpeg codec string for the detected/specified hardware
            
        Example:
            >>> DetectGPU.get_optimal_codec()
            'hevc_nvenc'  # On NVIDIA systems
        """
        if vendor is None:
            vendor = DetectGPU.detect_gpu_vendor()
        
        return DetectGPU._CODEC_MAP.get(vendor, "libx265")