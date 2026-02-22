"""
BuildingBobs — Pipeline Configuration

Central configuration for all pipeline parameters.
All values are tuneable via CLI flags or by editing defaults here.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StabiliserConfig:
    """FFmpeg vidstab stabilisation parameters."""
    smoothing: int = 30          # Smoothing radius (higher = smoother, more crop)
    shakiness: int = 10          # Shakiness detection sensitivity (1-10)
    accuracy: int = 15           # Analysis accuracy (1-15, higher = slower but better)
    optzoom: int = 1             # 0 = no zoom, 1 = optimal zoom to avoid borders, 2 = adaptive
    interpol: str = "bicubic"    # Interpolation method
    codec: str = "libx264"       # Output video codec
    crf: int = 18                # Quality (lower = better, 18 = visually lossless)


@dataclass
class QualityConfig:
    """Frame quality scoring parameters."""
    blur_threshold: float = 100.0    # Laplacian variance below this = blurry
    min_brightness: int = 30         # Mean intensity below this = too dark
    max_brightness: int = 240        # Mean intensity above this = too bright
    blur_weight: float = 0.7         # Weight for blur score in overall quality
    brightness_weight: float = 0.3   # Weight for brightness score in overall quality


@dataclass
class KeyframeConfig:
    """Key-frame extraction parameters."""
    sampling_fps: float = 2.0        # Uniform sampling rate (frames/sec) — user requested higher
    scene_threshold: float = 27.0    # PySceneDetect ContentDetector threshold
    min_quality_score: float = 30.0  # Minimum quality score (0-100) to keep a frame
    dedup_window_sec: float = 0.3    # Deduplicate frames within this time window
    output_format: str = "jpg"       # Output image format
    jpeg_quality: int = 95           # JPEG quality for saved frames


@dataclass
class DeblurConfig:
    """Motion deblurring parameters."""
    kernel_size: int = 15            # Size of the motion blur kernel
    snr: float = 25.0                # Signal-to-noise ratio for Wiener filter
    apply_threshold: float = 50.0    # Only deblur frames with quality score below this
    skip_threshold: float = 15.0     # Discard frames below this — too blurry to recover


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    input_dir: Path = field(default_factory=lambda: Path("./input"))
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    skip_stabilise: bool = False
    skip_deblur: bool = False
    video_extensions: tuple = (".mp4", ".mkv", ".avi", ".mov", ".MP4", ".MOV")

    stabiliser: StabiliserConfig = field(default_factory=StabiliserConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    deblur: DeblurConfig = field(default_factory=DeblurConfig)
