"""
BuildingBobs — Phase 1 Pipeline CLI

Orchestrates the full video pre-processing pipeline:
  1. Stabilise raw body-cam footage (FFmpeg vidstab)
  2. Score all frames for quality (Laplacian + brightness)
  3. Extract key-frames (uniform + scene-change, quality-filtered)
  4. Optionally deblur marginal frames
  5. Output: stabilised video, key-frames, quality report, metadata

Usage:
    python -m pipeline.main --input ./input --output ./output
    python -m pipeline.main --input ./input --output ./output --fps 3.0 --quality-threshold 25
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from tqdm import tqdm

from .config import PipelineConfig
from .stabiliser import stabilise_video
from .quality import score_video_frames, score_frame
from .keyframes import extract_keyframes
from .deblur import deblur_keyframe_if_needed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("buildingbobs")


def discover_videos(input_dir: Path, extensions: tuple) -> list[Path]:
    """Find all video files in the input directory."""
    videos = []
    for ext in extensions:
        videos.extend(input_dir.glob(f"*{ext}"))
    videos.sort()
    return videos


def process_single_clip(
    video_path: Path,
    output_root: Path,
    config: PipelineConfig,
) -> dict:
    """
    Process a single body-cam clip through the full pipeline.

    Returns:
        dict with processing results and metadata.
    """
    clip_name = video_path.stem
    clip_output = output_root / clip_name
    clip_output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "clip_name": clip_name,
        "input_file": str(video_path),
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "stages": {},
    }

    overall_start = time.time()

    # ==========================================
    # Stage 1: Video Stabilisation
    # ==========================================
    if not config.skip_stabilise:
        logger.info(f"━━━ Stage 1/4: Stabilising {clip_name} ━━━")
        stage_start = time.time()

        stabilised_path = clip_output / "stabilised.mp4"
        stab_result = stabilise_video(
            video_path, stabilised_path, config.stabiliser
        )

        metadata["stages"]["stabilisation"] = {
            "success": stab_result["success"],
            "message": stab_result["message"],
            "duration_sec": round(time.time() - stage_start, 2),
            "output_file": str(stabilised_path) if stab_result["success"] else None,
        }

        # Use stabilised video for downstream processing
        processing_video = stabilised_path if stab_result["success"] else video_path
        if not stab_result["success"]:
            logger.warning("Stabilisation failed. Continuing with raw video.")
    else:
        logger.info(f"━━━ Stage 1/4: Stabilisation SKIPPED ━━━")
        processing_video = video_path
        metadata["stages"]["stabilisation"] = {"skipped": True}

    # ==========================================
    # Stage 2: Frame Quality Scoring
    # ==========================================
    logger.info(f"━━━ Stage 2/4: Scoring frame quality ━━━")
    stage_start = time.time()

    # Score every 5th frame for speed (we'll re-score key-frame candidates precisely)
    quality_scores = score_video_frames(
        str(processing_video), config.quality, sample_every_n=5
    )

    # Save quality report
    quality_report = {
        "video": str(processing_video),
        "total_scored": len(quality_scores),
        "avg_quality": round(
            sum(s.overall_score for s in quality_scores) / max(len(quality_scores), 1), 1
        ),
        "usable_pct": round(
            sum(1 for s in quality_scores if s.is_usable) / max(len(quality_scores), 1) * 100, 1
        ),
        "blurry_pct": round(
            sum(1 for s in quality_scores if s.is_blurry) / max(len(quality_scores), 1) * 100, 1
        ),
        "frames": [s.to_dict() for s in quality_scores],
    }

    quality_path = clip_output / "quality_report.json"
    with open(quality_path, "w") as f:
        json.dump(quality_report, f, indent=2)

    metadata["stages"]["quality_scoring"] = {
        "total_scored": len(quality_scores),
        "avg_quality": quality_report["avg_quality"],
        "usable_pct": quality_report["usable_pct"],
        "blurry_pct": quality_report["blurry_pct"],
        "duration_sec": round(time.time() - stage_start, 2),
    }

    logger.info(
        f"Quality: avg={quality_report['avg_quality']}, "
        f"usable={quality_report['usable_pct']}%, "
        f"blurry={quality_report['blurry_pct']}%"
    )

    # ==========================================
    # Stage 3: Key-Frame Extraction
    # ==========================================
    logger.info(f"━━━ Stage 3/4: Extracting key-frames ━━━")
    stage_start = time.time()

    keyframes_dir = clip_output / "keyframes"
    kf_result = extract_keyframes(
        str(processing_video),
        str(keyframes_dir),
        config.keyframe,
        config.quality,
        quality_scores,
    )

    metadata["stages"]["keyframe_extraction"] = {
        "total_candidates": kf_result["total_candidates"],
        "saved_count": kf_result["saved_count"],
        "rejected_count": kf_result["rejected_count"],
        "sampling_fps": config.keyframe.sampling_fps,
        "duration_sec": round(time.time() - stage_start, 2),
    }

    logger.info(
        f"Key-frames: {kf_result['saved_count']} saved, "
        f"{kf_result['rejected_count']} rejected"
    )

    # ==========================================
    # Stage 4: Deblurring (optional)
    # ==========================================
    if not config.skip_deblur and kf_result["saved_count"] > 0:
        logger.info(f"━━━ Stage 4/4: Deblurring marginal frames ━━━")
        stage_start = time.time()

        import cv2
        deblurred_count = 0

        for kf in kf_result["keyframes"]:
            if kf["quality_score"] < config.deblur.apply_threshold:
                filepath = keyframes_dir / kf["filename"]
                if filepath.exists():
                    frame = cv2.imread(str(filepath))
                    if frame is not None:
                        deblurred, was_deblurred = deblur_keyframe_if_needed(
                            frame, kf["quality_score"], config.deblur
                        )
                        if was_deblurred:
                            cv2.imwrite(str(filepath), deblurred,
                                       [cv2.IMWRITE_JPEG_QUALITY, config.keyframe.jpeg_quality])
                            deblurred_count += 1

        metadata["stages"]["deblurring"] = {
            "deblurred_count": deblurred_count,
            "duration_sec": round(time.time() - stage_start, 2),
        }
        logger.info(f"Deblurred {deblurred_count} frames")
    else:
        logger.info(f"━━━ Stage 4/4: Deblurring SKIPPED ━━━")
        metadata["stages"]["deblurring"] = {"skipped": True}

    # ==========================================
    # Save metadata
    # ==========================================
    metadata["total_duration_sec"] = round(time.time() - overall_start, 2)

    metadata_path = clip_output / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"✅ {clip_name} complete in {metadata['total_duration_sec']}s — "
        f"{kf_result['saved_count']} key-frames extracted"
    )

    return metadata


@click.command()
@click.option("--input", "input_dir", type=click.Path(exists=True), default="./input",
              help="Directory containing raw body-cam MP4 clips.")
@click.option("--output", "output_dir", type=click.Path(), default="./output",
              help="Directory to write all processed outputs.")
@click.option("--fps", type=float, default=2.0,
              help="Key-frame sampling rate in frames per second.")
@click.option("--quality-threshold", type=float, default=30.0,
              help="Minimum quality score (0-100) for a frame to be kept.")
@click.option("--skip-stabilise", is_flag=True, default=False,
              help="Skip video stabilisation (for pre-stabilised input).")
@click.option("--skip-deblur", is_flag=True, default=False,
              help="Skip deblurring step.")
def main(input_dir, output_dir, fps, quality_threshold, skip_stabilise, skip_deblur):
    """
    BuildingBobs Phase 1 — Video Pre-Processing Pipeline

    Processes raw egocentric body-cam footage into clean, quality-scored
    key-frames ready for downstream MLLM analysis.
    """
    config = PipelineConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        skip_stabilise=skip_stabilise,
        skip_deblur=skip_deblur,
    )
    config.keyframe.sampling_fps = fps
    config.keyframe.min_quality_score = quality_threshold

    # Banner
    click.echo("=" * 60)
    click.echo("  🏗️  BuildingBobs — Video Pre-Processing Pipeline")
    click.echo("=" * 60)
    click.echo(f"  Input:  {config.input_dir.resolve()}")
    click.echo(f"  Output: {config.output_dir.resolve()}")
    click.echo(f"  FPS:    {fps}  |  Quality threshold: {quality_threshold}")
    click.echo(f"  Stabilise: {'SKIP' if skip_stabilise else 'ON'}  |  Deblur: {'SKIP' if skip_deblur else 'ON'}")
    click.echo("=" * 60)

    # Discover videos
    videos = discover_videos(config.input_dir, config.video_extensions)
    if not videos:
        click.echo(f"\n❌ No video files found in {config.input_dir.resolve()}")
        click.echo(f"   Supported formats: {', '.join(config.video_extensions)}")
        sys.exit(1)

    click.echo(f"\n📹 Found {len(videos)} video(s):")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        click.echo(f"   • {v.name} ({size_mb:.1f} MB)")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each clip
    all_results = []
    pipeline_start = time.time()

    for video in tqdm(videos, desc="Processing clips", unit="clip"):
        result = process_single_clip(video, config.output_dir, config)
        all_results.append(result)

    # Write pipeline summary
    pipeline_summary = {
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_clips": len(all_results),
        "total_duration_sec": round(time.time() - pipeline_start, 2),
        "config": {
            "sampling_fps": fps,
            "quality_threshold": quality_threshold,
            "stabilise": not skip_stabilise,
            "deblur": not skip_deblur,
        },
        "clips": [],
    }

    total_keyframes = 0
    for r in all_results:
        kf_stage = r.get("stages", {}).get("keyframe_extraction", {})
        total_keyframes += kf_stage.get("saved_count", 0)
        pipeline_summary["clips"].append({
            "clip_name": r["clip_name"],
            "keyframes_extracted": kf_stage.get("saved_count", 0),
            "avg_quality": r.get("stages", {}).get("quality_scoring", {}).get("avg_quality", 0),
            "processing_time_sec": r.get("total_duration_sec", 0),
        })

    pipeline_summary["total_keyframes"] = total_keyframes

    summary_path = config.output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(pipeline_summary, f, indent=2)

    # Final summary
    click.echo("\n" + "=" * 60)
    click.echo("  ✅  Pipeline Complete!")
    click.echo("=" * 60)
    click.echo(f"  Clips processed:    {len(all_results)}")
    click.echo(f"  Total key-frames:   {total_keyframes}")
    click.echo(f"  Total time:         {pipeline_summary['total_duration_sec']}s")
    click.echo(f"  Output directory:   {config.output_dir.resolve()}")
    click.echo(f"  Summary:            {summary_path.resolve()}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
