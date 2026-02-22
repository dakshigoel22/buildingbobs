"""
BuildingBobs — MLLM Frame Analyzer

Sends egocentric key-frames to either Google Gemini or Ollama for
structured analysis. Includes rate-limit-aware retry logic with
exponential backoff.
"""

import base64
import json
import logging
import time
from pathlib import Path

import requests

from .config import VisionConfig
from .prompts import SYSTEM_PROMPT, ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


def _encode_image_base64(image_path: Path) -> str:
    """Read an image file and return base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _get_mime_type(image_path: Path) -> str:
    """Get MIME type from file extension."""
    ext = image_path.suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")


def _parse_response(raw_text: str, image_path: Path) -> dict:
    """Parse LLM response text into a JSON dict."""
    try:
        text = raw_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        parsed = json.loads(text)
        parsed["_meta"] = {
            "source_frame": image_path.name,
            "parse_success": True,
        }
        return parsed

    except json.JSONDecodeError as e:
        # Try to extract JSON from surrounding text
        try:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            parsed = json.loads(raw_text[start:end])
            parsed["_meta"] = {"source_frame": image_path.name, "parse_success": True}
            return parsed
        except (ValueError, json.JSONDecodeError):
            pass

        logger.warning(f"JSON parse failed for {image_path.name}: {e}")
        return {
            "_meta": {"source_frame": image_path.name, "parse_success": False, "error": str(e)},
            "visible_objects": [],
            "environment": {"type": "unknown"},
            "hand_state": {"visible": False},
            "activity": {"label": "frame_too_blurry", "category": "unclear", "confidence": 0.0},
            "frame_quality": {"usable": False, "blur_level": "unknown", "issues": ["parse_error"]},
        }


# ─── Gemini Backend ──────────────────────────────────────────────

def _call_gemini(image_path: Path, config: VisionConfig) -> dict:
    """Make a single Gemini API call for one frame."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=config.api_key)

    img_bytes = open(image_path, "rb").read()
    mime = _get_mime_type(image_path)

    response = client.models.generate_content(
        model=config.model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime),
            ANALYSIS_PROMPT,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    return _parse_response(response.text, image_path)


# ─── Ollama Backend ──────────────────────────────────────────────

def _call_ollama(image_path: Path, config: VisionConfig) -> dict:
    """Make a single Ollama API call for one frame using the /api/chat endpoint."""
    img_b64 = _encode_image_base64(image_path)

    # Combine system prompt and analysis prompt into the messages
    payload = {
        "model": config.ollama_model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": ANALYSIS_PROMPT,
                "images": [img_b64],
            },
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    url = f"{config.ollama_url.rstrip('/')}/api/chat"
    resp = requests.post(url, json=payload, timeout=config.timeout_sec)
    resp.raise_for_status()

    data = resp.json()
    raw_text = data.get("message", {}).get("content", "")
    return _parse_response(raw_text, image_path)


# ─── Dispatcher ──────────────────────────────────────────────────

def analyze_frame(image_path: Path, config: VisionConfig) -> dict:
    """
    Analyze a single frame with retry logic.
    Routes to Gemini or Ollama based on config.provider.
    Handles errors with exponential backoff.
    """
    call_fn = _call_gemini if config.provider == "gemini" else _call_ollama

    for attempt in range(1, config.max_retries + 1):
        try:
            result = call_fn(image_path, config)
            return result
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "quota" in error_str.lower()
            wait = config.retry_delay_sec if is_rate_limit else (5.0 * attempt)

            logger.warning(
                f"Attempt {attempt}/{config.max_retries} failed for {image_path.name}: "
                f"{'Rate limited' if is_rate_limit else 'Error'}: {error_str}. Waiting {wait:.0f}s..."
            )
            if attempt < config.max_retries:
                time.sleep(wait)

    logger.error(f"All {config.max_retries} retries failed for {image_path.name}")
    return {
        "_meta": {"source_frame": image_path.name, "parse_success": False, "error": "all_retries_failed"},
        "visible_objects": [],
        "environment": {"type": "unknown"},
        "hand_state": {"visible": False},
        "activity": {"label": "frame_too_blurry", "category": "unclear", "confidence": 0.0},
        "frame_quality": {"usable": False, "issues": ["api_error"]},
    }
