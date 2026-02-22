"""
BuildingBobs — Key-Frame Extractor

Extracts representative key-frames from video using two strategies:
  1. Uniform sampling at configurable FPS (default 2 fps)
  2. Scene-change detection via PySceneDetect ContentDetector

Frames are merged, deduplicated, filtered by quality score, and
saved as numbered JPGs with quality score in the filename.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from .config import KeyframeConfig, QualityConfig
from .quality import FrameQuality, score_frame

logger = logging.getLogger(__name__)


def detect_scene_changes(video_path: str, threshold: float = 27.0) -> list[float]:
    """
    Detect scene-change timestamps using PySceneDetect.

    Returns:
        List of timestamps (seconds) where scene changes occur.
    """
    try:
        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)

        scene_list = scene_manager.get_scene_list()
        timestamps = []
        for scene in scene_list:
            # Get the start timestamp of each scene (i.e., the cut point)
            start_sec = scene[0].get_seconds()
            timestamps.append(start_sec)

        logger.info(f"Detected {len(timestamps)} scene changes in {video_path}")
        return timestamps

    except Exception as e:
        logger.warning(f"Scene detection failed: {e}. Falling back to uniform sampling only.")
        return []


def compute_uniform_timestamps(duration_sec: float, fps: float) -> list[float]:
    """Generate uniform sample timestamps at the given FPS."""
    if fps <= 0 or duration_sec <= 0:
        return []

    interval = 1.0 / fps
    timestamps = []
    t = 0.0
    while t < duration_sec:
        timestamps.append(t)
        t += interval
    return timestamps


def merge_and_dedup(
    uniform_ts: list[float],
    scene_ts: list[float],
    dedup_window: float = 0.3,
) -> list[float]:
    """
    Merge two timestamp lists and remove duplicates within a window.
    """
    all_ts = sorted(set(uniform_ts + scene_ts))
    if not all_ts:
        return []

    deduped = [all_ts[0]]
    for t in all_ts[1:]:
        if t - deduped[-1] >= dedup_window:
            deduped.append(t)
    return deduped


def extract_keyframes(
    video_path: str,
    output_dir: str,
    kf_config: KeyframeConfig | None = None,
    q_config: QualityConfig | None = None,
    pre_scored: list[FrameQuality] | None = None,
) -> dict:
    """
    Extract key-frames from a video and save as images.

    Args:
        video_path: Path to the (stabilised) video.
        output_dir: Directory to save key-frame images.
        kf_config: Key-frame extraction config.
        q_config: Quality thresholds for scoring extracted frames.
        pre_scored: Optional pre-computed quality scores (from quality.py).

    Returns:
        dict with: total_candidates, saved_count, rejected_count, keyframes[]
    """
    if kf_config is None:
        kf_config = KeyframeConfig()
    if q_config is None:
        q_config = QualityConfig()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build a quality lookup from pre-scored data (frame_idx → FrameQuality)
    quality_lookup: dict[int, FrameQuality] = {}
    if pre_scored:
        for q in pre_scored:
            quality_lookup[q.frame_index] = q

    # Open video and get properties
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return {"total_candidates": 0, "saved_count": 0, "rejected_count": 0, "keyframes": []}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(
        f"Extracting key-frames: {video_path} "
        f"({total_frames} frames, {fps:.1f} fps, {duration:.1f}s)"
    )

    # Generate candidate timestamps
    uniform_ts = compute_uniform_timestamps(duration, kf_config.sampling_fps)
    scene_ts = detect_scene_changes(video_path, kf_config.scene_threshold)
    candidate_ts = merge_and_dedup(uniform_ts, scene_ts, kf_config.dedup_window_sec)

    logger.info(
        f"Candidates: {len(uniform_ts)} uniform + {len(scene_ts)} scene-change "
        f"= {len(candidate_ts)} after dedup"
    )

    saved = []
    rejected = 0

    for ts in candidate_ts:
        frame_idx = int(ts * fps)
        if frame_idx >= total_frames:
            continue

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Score quality (use pre-scored if available, else compute)
        if frame_idx in quality_lookup:
            quality = quality_lookup[frame_idx]
        else:
            quality = score_frame(frame, frame_idx, ts, q_config)

        # Filter by quality
        if quality.overall_score < kf_config.min_quality_score:
            rejected += 1
            continue

        # Save frame
        score_int = int(quality.overall_score)
        filename = f"frame_{frame_idx:06d}_q{score_int}.{kf_config.output_format}"
        filepath = output_path / filename

        if kf_config.output_format == "jpg":
            cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, kf_config.jpeg_quality])
        else:
            cv2.imwrite(str(filepath), frame)

        saved.append({
            "filename": filename,
            "frame_index": frame_idx,
            "timestamp_sec": round(ts, 3),
            "quality_score": round(quality.overall_score, 1),
            "is_scene_change": ts in scene_ts,
        })

    cap.release()

    logger.info(f"Key-frames: {len(saved)} saved, {rejected} rejected (below quality threshold)")

    return {
        "total_candidates": len(candidate_ts),
        "saved_count": len(saved),
        "rejected_count": rejected,
        "keyframes": saved,
    }
