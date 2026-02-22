"""
BuildingBobs — Video Stabiliser

Two-pass FFmpeg vidstab stabilisation for egocentric body-cam footage.

Pass 1: Analyse video for motion transforms → generate .trf file
Pass 2: Apply stabilisation transforms → output stabilised video
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import StabiliserConfig

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """Check if FFmpeg is installed and has vidstab support."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-filters"],
            capture_output=True, text=True, timeout=10
        )
        if "vidstab" in result.stdout or "vidstab" in result.stderr:
            return True
        # vidstab might not show in -filters but still work
        logger.warning("vidstab filter not confirmed in FFmpeg filters list. Will attempt anyway.")
        return True
    except FileNotFoundError:
        logger.error("FFmpeg not found. Install with: brew install ffmpeg")
        return False
    except subprocess.TimeoutExpired:
        logger.warning("FFmpeg check timed out. Will attempt stabilisation anyway.")
        return True


def stabilise_video(
    input_path: Path,
    output_path: Path,
    config: StabiliserConfig | None = None,
) -> dict:
    """
    Stabilise a video file using FFmpeg's vidstab filter (two-pass).

    Args:
        input_path: Path to the raw input video.
        output_path: Path where stabilised video will be written.
        config: Stabilisation configuration. Uses defaults if None.

    Returns:
        dict with keys: success, input_path, output_path, message
    """
    if config is None:
        config = StabiliserConfig()

    result = {
        "success": False,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "message": "",
    }

    if not input_path.exists():
        result["message"] = f"Input file not found: {input_path}"
        logger.error(result["message"])
        return result

    if not check_ffmpeg_available():
        result["message"] = "FFmpeg not available. Skipping stabilisation."
        logger.error(result["message"])
        return result

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a temp file for the transform data
    with tempfile.TemporaryDirectory() as tmpdir:
        transform_file = Path(tmpdir) / "transforms.trf"

        # --- Pass 1: Detect motion ---
        logger.info(f"Stabilisation Pass 1 (analyse): {input_path.name}")
        pass1_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", (
                f"vidstabdetect="
                f"shakiness={config.shakiness}:"
                f"accuracy={config.accuracy}:"
                f"result='{transform_file}'"
            ),
            "-f", "null", "-"
        ]

        try:
            proc = subprocess.run(
                pass1_cmd,
                capture_output=True, text=True, timeout=600
            )
            if proc.returncode != 0:
                # Try without quotes around the path (some FFmpeg builds)
                pass1_cmd[-3] = (
                    f"vidstabdetect="
                    f"shakiness={config.shakiness}:"
                    f"accuracy={config.accuracy}:"
                    f"result={transform_file}"
                )
                proc = subprocess.run(
                    pass1_cmd,
                    capture_output=True, text=True, timeout=600
                )
                if proc.returncode != 0:
                    result["message"] = f"Pass 1 failed: {proc.stderr[-500:]}"
                    logger.error(result["message"])
                    return result
        except subprocess.TimeoutExpired:
            result["message"] = "Pass 1 timed out (>600s)"
            logger.error(result["message"])
            return result

        if not transform_file.exists():
            result["message"] = "Transform file not generated in Pass 1"
            logger.error(result["message"])
            return result

        # --- Pass 2: Apply stabilisation ---
        logger.info(f"Stabilisation Pass 2 (transform): {input_path.name}")
        pass2_cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", (
                f"vidstabtransform="
                f"input='{transform_file}':"
                f"smoothing={config.smoothing}:"
                f"optzoom={config.optzoom}:"
                f"interpol={config.interpol}"
            ),
            "-c:v", config.codec,
            "-crf", str(config.crf),
            "-an",  # Strip audio (per PRD privacy: video only)
            str(output_path)
        ]

        try:
            proc = subprocess.run(
                pass2_cmd,
                capture_output=True, text=True, timeout=600
            )
            if proc.returncode != 0:
                # Retry without quotes
                pass2_cmd[5] = (
                    f"vidstabtransform="
                    f"input={transform_file}:"
                    f"smoothing={config.smoothing}:"
                    f"optzoom={config.optzoom}:"
                    f"interpol={config.interpol}"
                )
                proc = subprocess.run(
                    pass2_cmd,
                    capture_output=True, text=True, timeout=600
                )
                if proc.returncode != 0:
                    result["message"] = f"Pass 2 failed: {proc.stderr[-500:]}"
                    logger.error(result["message"])
                    return result
        except subprocess.TimeoutExpired:
            result["message"] = "Pass 2 timed out (>600s)"
            logger.error(result["message"])
            return result

    if output_path.exists() and output_path.stat().st_size > 0:
        result["success"] = True
        result["message"] = "Stabilisation complete"
        logger.info(f"Stabilised video saved: {output_path}")
    else:
        result["message"] = "Output file missing or empty after stabilisation"
        logger.error(result["message"])

    return result
