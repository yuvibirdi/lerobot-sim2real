"""
macOS AVFoundation stereo camera implementation.

Bypasses lerobot's OpenCVCamera to directly use AVFoundation backend
and handle stereo frame splitting (2560x720 → 1280x720).
"""

import logging
import time
from typing import Literal

import cv2
import numpy as np

from .macos_camera_config import MacOSStereoCameraConfig

logger = logging.getLogger(__name__)


class MacOSStereoCamera:
    """
    macOS AVFoundation stereo camera that captures 2560x720 and splits to 1280x720.
    
    Implements a compatible interface with lerobot's Camera class via duck-typing:
    - connect(warmup: bool = True)
    - read(color_mode: str = "rgb") -> np.ndarray
    - disconnect()
    - async_read(color_mode: str = "rgb") -> np.ndarray
    
    Example:
        config = MacOSStereoCameraConfig(index=0, stereo_side="left")
        camera = MacOSStereoCamera(config)
        camera.connect()
        frame = camera.read()  # Returns 1280x720 RGB image
        camera.disconnect()
    """
    
    def __init__(self, config: MacOSStereoCameraConfig):
        """
        Initialize macOS stereo camera.
        
        Args:
            config: Camera configuration
        """
        self.config = config
        self.index = config.index
        self.fps = config.fps
        self.stereo_side = config.stereo_side
        self.capture_width = config.capture_width
        self.capture_height = config.capture_height
        self.output_width = config.output_width
        self.output_height = config.output_height
        self.warmup_s = config.warmup_s
        self.default_color_mode = config.color_mode
        
        self.videocapture: cv2.VideoCapture | None = None
        self.is_connected = False
    
    def __repr__(self) -> str:
        return (
            f"MacOSStereoCamera(index={self.index}, "
            f"stereo_side={self.stereo_side!r}, "
            f"capture={self.capture_width}x{self.capture_height}, "
            f"output={self.output_width}x{self.output_height})"
        )
    
    def connect(self, warmup: bool = True) -> None:
        """
        Open camera with AVFoundation backend.
        
        Args:
            warmup: If True, wait for camera auto-adjustment
            
        Raises:
            ConnectionError: If camera fails to open
        """
        if self.is_connected:
            logger.warning(f"{self} is already connected")
            return
        
        logger.info(f"Connecting to {self} via AVFoundation...")
        
        # Use AVFoundation backend explicitly for macOS
        self.videocapture = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        
        if not self.videocapture.isOpened():
            raise ConnectionError(
                f"Failed to open camera index {self.index} via AVFoundation. "
                "Make sure the camera is connected and not in use by another app."
            )
        
        # Request stereo resolution
        self.videocapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        self.videocapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self.videocapture.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Verify settings (warn but don't fail on mismatch)
        actual_width = int(self.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.videocapture.get(cv2.CAP_PROP_FPS)
        
        if actual_width != self.capture_width:
            logger.warning(
                f"Requested width {self.capture_width}, got {actual_width}. "
                "Proceeding anyway."
            )
        if actual_height != self.capture_height:
            logger.warning(
                f"Requested height {self.capture_height}, got {actual_height}. "
                "Proceeding anyway."
            )
        if abs(actual_fps - self.fps) > 1:
            logger.warning(
                f"Requested FPS {self.fps}, got {actual_fps}. "
                "Proceeding anyway."
            )
        
        logger.info(
            f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )
        
        self.is_connected = True
        
        if warmup:
            logger.info(f"Warming up camera for {self.warmup_s}s...")
            time.sleep(self.warmup_s)
            # Read and discard a few frames to clear buffer
            for _ in range(5):
                self.videocapture.read()
    
    def read(self, color_mode: str | None = None) -> np.ndarray:
        """
        Capture frame, split stereo, return single image.
        
        Args:
            color_mode: "rgb" or "bgr". Defaults to config.color_mode
            
        Returns:
            Image array of shape (height, width, 3)
            
        Raises:
            RuntimeError: If camera is not connected or read fails
        """
        if not self.is_connected or self.videocapture is None:
            raise RuntimeError(f"{self} is not connected. Call connect() first.")
        
        ret, frame = self.videocapture.read()
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {self}")
        
        # Split stereo frame
        mid = frame.shape[1] // 2
        if self.stereo_side == "left":
            frame = frame[:, :mid]
        else:
            frame = frame[:, mid:]
        
        # Convert color if needed
        mode = color_mode or self.default_color_mode
        if mode == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame
    
    def disconnect(self) -> None:
        """Release camera resources."""
        if self.videocapture is not None:
            self.videocapture.release()
            self.videocapture = None
        self.is_connected = False
        logger.info(f"{self} disconnected")
    
    def async_read(self, color_mode: str | None = None) -> np.ndarray:
        """
        'Async' read method for compatibility with lerobot/mani_skill.
        
        Note: Despite the name, this is NOT an async coroutine.
        Lerobot's Camera.async_read() is actually synchronous and returns
        the image directly. This matches that interface for duck-typing.
        """
        return self.read(color_mode)
    
    @property
    def width(self) -> int:
        """Output image width (after stereo split)."""
        return self.output_width
    
    @property
    def height(self) -> int:
        """Output image height."""
        return self.output_height
    
    @staticmethod
    def find_cameras(max_index: int = 10) -> list[int]:
        """
        Scan for available cameras via AVFoundation.
        
        Args:
            max_index: Maximum camera index to scan
            
        Returns:
            List of valid camera indices
        """
        found = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                found.append(i)
                cap.release()
        return found
