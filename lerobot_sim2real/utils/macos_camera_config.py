"""Configuration dataclass for macOS stereo camera."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class MacOSStereoCameraConfig:
    """
    Configuration for macOS AVFoundation stereo camera.
    
    Captures 2560x720 stereo frames and splits to 1280x720.
    
    Attributes:
        index: Camera index (0, 1, 2, ...) for AVFoundation
        fps: Target frames per second (default 30)
        stereo_side: Which half of the stereo image to return ("left" or "right")
        capture_width: Full stereo frame width (default 2560)
        capture_height: Full stereo frame height (default 720)
        output_width: Output image width after split (default 1280)
        output_height: Output image height (default 720)
        warmup_s: Seconds to wait for camera auto-adjustment (default 1.0)
        color_mode: Output color format ("rgb" or "bgr")
    """
    index: int = 0
    fps: int = 30
    stereo_side: Literal["left", "right"] = "left"
    capture_width: int = 2560
    capture_height: int = 720
    output_width: int = 1280
    output_height: int = 720
    warmup_s: float = 1.0
    color_mode: Literal["rgb", "bgr"] = "rgb"
