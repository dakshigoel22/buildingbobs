"""
BuildingBobs — Frame Quality Scorer

Assesses each video frame for visual quality using:
  1. Blur detection via Laplacian variance (higher = sharper)
  2. Brightness check via mean pixel intensity
  3. Combined weighted score on 0–100 scale

Used to filter out unusable frames before key-frame extraction and
to guide which frames need deblurring vs. which are unsalvageable.
"""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import QualityConfig

logger = logging.getLogger(__name__)


@dataclass
class FrameQuality:
    """Quality assessment for a single frame."""
    frame_index: int
    timestamp_sec: float
    blur_score: float          # Laplacian variance (raw)
    blur_normalised: float     # Normalised 0–100 (100 = perfectly sharp)
    brightness: float          # Mean pixel intensity 0–255
    brightness_normalised: float  # Normalised 0–100
    overall_score: float       # Weighted combination 0–100
    is_usable: bool            # Above minimum threshold?
    is_blurry: bool            # Below blur threshold?
    is_dark: bool
    is_bright: bool

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "blur_score_raw": round(self.blur_score, 2),
            "blur_normalised": round(self.blur_normalised, 1),
            "brightness": round(self.brightness, 1),
            "brightness_normalised": round(self.brightness_normalised, 1),
            "overall_score": round(self.overall_score, 1),
            "is_usable": self.is_usable,
            "is_blurry": self.is_blurry,
            "is_dark": self.is_dark,
            "is_bright": self.is_bright,
        }


def compute_blur_score(frame: np.ndarray) -> float:
    """
    Compute blur score using Laplacian variance.
    Higher value = sharper image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_brightness(frame: np.ndarray) -> float:
    """Compute mean brightness of a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(np.mean(gray))


def normalise_blur(raw_blur: float, max_blur: float = 1000.0) -> float:
    """Normalise raw blur score to 0–100 range."""
    # Clamp and scale — typical Laplacian variance ranges from 0 to ~2000
    clamped = min(raw_blur, max_blur)
    return (clamped / max_blur) * 100.0


def normalise_brightness(brightness: float, min_b: int = 30, max_b: int = 240) -> float:
    """
    Normalise brightness to 0–100 where:
    - 100 = optimal (midpoint between min_b and max_b)
    - 0 = too dark or too bright
    """
    midpoint = (min_b + max_b) / 2.0
    half_range = (max_b - min_b) / 2.0

    if brightness < min_b or brightness > max_b:
        return 0.0

    distance = abs(brightness - midpoint)
    return max(0.0, (1.0 - distance / half_range) * 100.0)


def score_frame(
    frame: np.ndarray,
    frame_index: int,
    timestamp_sec: float,
    config: QualityConfig | None = None,
) -> FrameQuality:
    """
    Score a single frame for quality.

    Args:
        frame: BGR numpy array
        frame_index: Index of the frame in the video
        timestamp_sec: Timestamp in seconds
        config: Quality thresholds

    Returns:
        FrameQuality dataclass
    """
    if config is None:
        config = QualityConfig()

    blur_raw = compute_blur_score(frame)
    brightness = compute_brightness(frame)

    blur_norm = normalise_blur(blur_raw)
    brightness_norm = normalise_brightness(
        brightness, config.min_brightness, config.max_brightness
    )

    overall = (
        config.blur_weight * blur_norm +
        config.brightness_weight * brightness_norm
    )

    is_blurry = blur_raw < config.blur_threshold
    is_dark = brightness < config.min_brightness
    is_bright = brightness > config.max_brightness
    is_usable = overall >= 15.0  # Very low bar — even marginal frames might be recoverable

    return FrameQuality(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        blur_score=blur_raw,
        blur_normalised=blur_norm,
        brightness=brightness,
        brightness_normalised=brightness_norm,
        overall_score=overall,
        is_usable=is_usable,
        is_blurry=is_blurry,
        is_dark=is_dark,
        is_bright=is_bright,
    )


def score_video_frames(
    video_path: str,
    config: QualityConfig | None = None,
    sample_every_n: int = 1,
) -> list[FrameQuality]:
    """
    Score all (or sampled) frames in a video.

    Args:
        video_path: Path to the video file
        config: Quality configuration
        sample_every_n: Score every Nth frame (1 = all frames)

    Returns:
        List of FrameQuality for each scored frame
    """
    if config is None:
        config = QualityConfig()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scores = []

    logger.info(f"Scoring frames: {video_path} ({total_frames} frames, {fps:.1f} fps)")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_every_n == 0:
            timestamp = frame_idx / fps
            quality = score_frame(frame, frame_idx, timestamp, config)
            scores.append(quality)

        frame_idx += 1

    cap.release()

    # Log summary statistics
    if scores:
        avg_quality = sum(s.overall_score for s in scores) / len(scores)
        usable_pct = sum(1 for s in scores if s.is_usable) / len(scores) * 100
        blurry_pct = sum(1 for s in scores if s.is_blurry) / len(scores) * 100
        logger.info(
            f"Quality summary: avg={avg_quality:.1f}, "
            f"usable={usable_pct:.0f}%, blurry={blurry_pct:.0f}%"
        )

    return scores
