"""
BuildingBobs — SSR Builder

Transforms raw MLLM analysis output into the Structured Scene
Representation (SSR) defined in the PRD. Also builds clip-level
SSR timelines for downstream agent consumption.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_frame_index(filename: str) -> int:
    """Extract frame index from filename like 'frame_000119_q30.jpg'."""
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else 0


def extract_quality_score(filename: str) -> int:
    """Extract quality score from filename like 'frame_000119_q30.jpg'."""
    match = re.search(r"_q(\d+)", filename)
    return int(match.group(1)) if match else 0


def build_ssr(analysis: dict, frame_filename: str, video_fps: float = 60.0) -> dict:
    """
    Build a Structured Scene Representation from raw MLLM analysis.

    Args:
        analysis: Raw JSON dict from the MLLM analyzer.
        frame_filename: Original frame filename (for index/timestamp extraction).
        video_fps: FPS of the source video (for timestamp calculation).

    Returns:
        SSR dict matching the PRD schema.
    """
    frame_index = extract_frame_index(frame_filename)
    quality_score = extract_quality_score(frame_filename)
    timestamp_sec = round(frame_index / video_fps, 3)

    # Extract fields with safe fallbacks
    activity = analysis.get("activity", {})
    hand_state = analysis.get("hand_state", {})
    environment = analysis.get("environment", {})
    visible_objects = analysis.get("visible_objects", [])
    frame_quality = analysis.get("frame_quality", {})

    ssr = {
        "frame_id": f"frame_{frame_index:06d}",
        "frame_index": frame_index,
        "timestamp_sec": timestamp_sec,
        "quality_score": quality_score,

        "hand_state": {
            "visible": hand_state.get("visible", False),
            "left": hand_state.get("left", "not_visible"),
            "right": hand_state.get("right", "not_visible"),
            "held_objects": hand_state.get("held_objects", []),
        },

        "visible_objects": [
            {
                "label": obj.get("label", "unknown"),
                "region": obj.get("region", "unknown"),
                "confidence": round(obj.get("confidence", 0.0), 2),
            }
            for obj in visible_objects
        ],

        "environment": {
            "type": environment.get("type", "unknown"),
            "lighting": environment.get("lighting", "unknown"),
            "weather": environment.get("weather", "unknown"),
            "surface": environment.get("surface", "unknown"),
        },

        "activity": {
            "label": activity.get("label", "unknown"),
            "category": activity.get("category", "unclear"),
            "confidence": round(activity.get("confidence", 0.0), 2),
            "reasoning": activity.get("reasoning", ""),
        },

        "frame_usable": frame_quality.get("usable", False),
        "blur_level": frame_quality.get("blur_level", "unknown"),
    }

    return ssr


def build_clip_timeline(
    ssr_list: list[dict],
) -> dict:
    """
    Build a chronological SSR timeline for an entire clip.
    Also computes aggregate statistics for the clip.

    Args:
        ssr_list: List of SSR dicts, one per analyzed frame.

    Returns:
        Clip-level timeline with frames and aggregate stats.
    """
    # Sort by frame index
    sorted_frames = sorted(ssr_list, key=lambda s: s["frame_index"])

    # Compute aggregate stats
    total = len(sorted_frames)
    if total == 0:
        return {"frames": [], "stats": {}}

    # Activity distribution
    activity_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for s in sorted_frames:
        label = s["activity"]["label"]
        cat = s["activity"]["category"]
        activity_counts[label] = activity_counts.get(label, 0) + 1
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Time coverage
    timestamps = [s["timestamp_sec"] for s in sorted_frames]
    duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

    # Unique objects seen
    all_objects = set()
    for s in sorted_frames:
        for obj in s["visible_objects"]:
            all_objects.add(obj["label"])

    # Hand activity
    hands_visible_count = sum(1 for s in sorted_frames if s["hand_state"]["visible"])

    stats = {
        "total_frames_analyzed": total,
        "clip_duration_sec": round(duration, 2),
        "activity_distribution": {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in sorted(activity_counts.items(), key=lambda x: -x[1])
        },
        "category_distribution": {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in sorted(category_counts.items(), key=lambda x: -x[1])
        },
        "productivity_summary": {
            "productive_pct": round(
                category_counts.get("productive", 0) / total * 100, 1
            ),
            "supportive_pct": round(
                category_counts.get("supportive", 0) / total * 100, 1
            ),
            "non_productive_pct": round(
                category_counts.get("non_productive", 0) / total * 100, 1
            ),
            "unclear_pct": round(
                category_counts.get("unclear", 0) / total * 100, 1
            ),
        },
        "objects_seen": sorted(all_objects),
        "hands_visible_pct": round(hands_visible_count / total * 100, 1),
        "avg_activity_confidence": round(
            sum(s["activity"]["confidence"] for s in sorted_frames) / total, 2
        ),
    }

    return {
        "frames": sorted_frames,
        "stats": stats,
    }


def save_ssr(ssr: dict, output_dir: Path, filename: str) -> Path:
    """Save an SSR dict to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(ssr, f, indent=2)
    return filepath
