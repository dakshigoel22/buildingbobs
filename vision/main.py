"""
BuildingBobs — Vision Pipeline CLI

Orchestrates frame analysis using Gemini or Ollama.

Input modes:
  --video <path>      Raw video → runs Phase 1 first, then vision
  --clip-dir <path>   Phase 1 output → analyzes existing key-frames

Provider modes:
  --provider gemini    Google Gemini API (default, needs API key)
  --provider ollama    Local/remote Ollama server (no API key needed)
"""

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from tqdm import tqdm

from .config import VisionConfig
from .analyzer import analyze_frame
from .ssr_builder import build_ssr, build_clip_timeline, save_ssr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("buildingbobs.vision")


def find_keyframes(clip_dir: Path) -> list[Path]:
    """Find all key-frame images in a clip output directory."""
    kf_dir = clip_dir / "keyframes"
    if not kf_dir.exists():
        return []
    return sorted(kf_dir.glob("frame_*.jpg")) + sorted(kf_dir.glob("frame_*.png"))


def run_phase1_on_video(video_path: Path, output_dir: Path, fps: float) -> Path:
    """Run Phase 1 pipeline on a raw video, return the clip output directory."""
    from pipeline.main import process_single_clip
    from pipeline.config import PipelineConfig

    config = PipelineConfig(
        input_dir=video_path.parent,
        output_dir=output_dir,
        skip_stabilise=True,
    )
    config.keyframe.sampling_fps = fps

    logger.info(f"Running Phase 1 pre-processing on {video_path.name}...")
    process_single_clip(video_path, output_dir, config)
    return output_dir / video_path.stem


def analyze_clip(clip_dir: Path, config: VisionConfig) -> dict:
    """Analyze all key-frames in a clip directory."""
    keyframes = find_keyframes(clip_dir)
    if not keyframes:
        logger.error(f"No key-frames found in {clip_dir}/keyframes/")
        return {"error": "no_keyframes"}

    logger.info(f"Found {len(keyframes)} key-frames to analyze")

    analysis_dir = clip_dir / "analysis"
    ssr_dir = clip_dir / "ssr"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    ssr_dir.mkdir(parents=True, exist_ok=True)

    video_fps = 60.0
    all_ssrs = []
    start_time = time.time()

    for i, frame_path in enumerate(tqdm(keyframes, desc="Analyzing frames", unit="frame")):
        logger.info(f"[{i+1}/{len(keyframes)}] Analyzing {frame_path.name}...")
        raw_analysis = analyze_frame(frame_path, config)

        # Save raw analysis
        if config.save_raw_response:
            analysis_path = analysis_dir / f"{frame_path.stem}.json"
            with open(analysis_path, "w") as f:
                json.dump(raw_analysis, f, indent=2)

        # Build SSR
        ssr = build_ssr(raw_analysis, frame_path.name, video_fps)
        save_ssr(ssr, ssr_dir, f"{frame_path.stem}_ssr.json")
        all_ssrs.append(ssr)

        # Rate limiting — only needed for cloud APIs
        if config.provider == "gemini" and i < len(keyframes) - 1:
            time.sleep(config.request_delay_sec)

    # Build clip timeline
    timeline = build_clip_timeline(all_ssrs)
    timeline_path = clip_dir / "clip_ssr_timeline.json"
    with open(timeline_path, "w") as f:
        json.dump(timeline, f, indent=2)

    # Write vision summary
    model_name = config.model if config.provider == "gemini" else config.ollama_model
    summary = {
        "clip_name": clip_dir.name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "provider": config.provider,
        "model": model_name,
        "frames_analyzed": len(all_ssrs),
        "processing_time_sec": round(time.time() - start_time, 2),
        "stats": timeline.get("stats", {}),
    }

    summary_path = clip_dir / "vision_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


@click.command()
@click.option("--video", "video_path", type=click.Path(exists=True), default=None,
              help="Raw body-cam video file. Runs Phase 1 first, then vision.")
@click.option("--clip-dir", type=click.Path(exists=True), default=None,
              help="Phase 1 output clip directory (already has key-frames).")
@click.option("--output", "output_dir", type=click.Path(), default="./output",
              help="Output root directory (used with --video).")
@click.option("--provider", type=click.Choice(["gemini", "ollama"], case_sensitive=False),
              default="gemini", help="Vision model provider (default: gemini).")
@click.option("--api-key", type=str, default=None,
              help="Gemini API key (overrides .env / GEMINI_API_KEY env var).")
@click.option("--ollama-url", type=str, default="http://localhost:11434",
              help="Ollama server URL (default: http://localhost:11434).")
@click.option("--ollama-model", type=str, default="llava:7b",
              help="Ollama vision model name (default: llava:7b).")
@click.option("--fps", type=float, default=3.0,
              help="Key-frame sampling FPS (only with --video).")
def main(video_path, clip_dir, output_dir, provider, api_key, ollama_url, ollama_model, fps):
    """
    BuildingBobs Sub-Track A — Egocentric Vision Analysis

    Analyzes body-cam key-frames using Gemini or Ollama to produce
    Structured Scene Representations (SSRs).
    """
    config = VisionConfig(provider=provider.lower())

    # Provider-specific overrides
    if provider.lower() == "gemini":
        if api_key:
            config.api_key = api_key
    elif provider.lower() == "ollama":
        config.ollama_url = ollama_url
        config.ollama_model = ollama_model
        # No API rate limiting needed for local Ollama
        config.request_delay_sec = 0.0

    # Validate
    try:
        config.validate()
    except ValueError as e:
        click.echo(f"\n❌ {e}")
        sys.exit(1)

    # Input mode
    if video_path and clip_dir:
        click.echo("❌ Specify either --video or --clip-dir, not both.")
        sys.exit(1)
    if not video_path and not clip_dir:
        click.echo("❌ Provide either --video <path> or --clip-dir <path>.")
        sys.exit(1)

    # Resolve model name for display
    model_name = config.model if config.provider == "gemini" else config.ollama_model

    # Banner
    click.echo("=" * 60)
    click.echo("  🧠  BuildingBobs — Egocentric Vision Analysis")
    click.echo("=" * 60)

    if video_path:
        video_path = Path(video_path)
        output_root = Path(output_dir)
        click.echo(f"  Mode:     Raw video → Phase 1 → Vision")
        click.echo(f"  Video:    {video_path}")
        clip_path = run_phase1_on_video(video_path, output_root, fps)
    else:
        clip_path = Path(clip_dir)
        click.echo(f"  Mode:     Analyze existing key-frames")
        click.echo(f"  Clip dir: {clip_path}")

    click.echo(f"  Provider: {config.provider}")
    click.echo(f"  Model:    {model_name}")
    if config.provider == "ollama":
        click.echo(f"  Server:   {config.ollama_url}")
    if config.provider == "gemini":
        click.echo(f"  Delay:    {config.request_delay_sec}s between frames")
    click.echo("=" * 60)

    # Run
    summary = analyze_clip(clip_path, config)

    if "error" in summary:
        click.echo(f"\n❌ Failed: {summary['error']}")
        sys.exit(1)

    # Results
    stats = summary.get("stats", {})
    prod = stats.get("productivity_summary", {})

    click.echo("\n" + "=" * 60)
    click.echo("  ✅  Vision Analysis Complete!")
    click.echo("=" * 60)
    click.echo(f"  Frames analyzed:    {summary['frames_analyzed']}")
    click.echo(f"  Processing time:    {summary['processing_time_sec']}s")
    click.echo(f"  Avg confidence:     {stats.get('avg_activity_confidence', 0)}")
    click.echo(f"")
    click.echo(f"  📊 Productivity Breakdown:")
    click.echo(f"     Productive:      {prod.get('productive_pct', 0)}%")
    click.echo(f"     Supportive:      {prod.get('supportive_pct', 0)}%")
    click.echo(f"     Non-productive:  {prod.get('non_productive_pct', 0)}%")
    click.echo(f"     Unclear:         {prod.get('unclear_pct', 0)}%")
    click.echo(f"")
    click.echo(f"  Objects seen:       {', '.join(stats.get('objects_seen', [])[:10])}")
    click.echo(f"")
    click.echo(f"  📁 Outputs:")
    click.echo(f"     {clip_path / 'analysis/'}")
    click.echo(f"     {clip_path / 'ssr/'}")
    click.echo(f"     {clip_path / 'clip_ssr_timeline.json'}")
    click.echo(f"     {clip_path / 'vision_summary.json'}")
    click.echo("=" * 60)


if __name__ == "__main__":
    main()
