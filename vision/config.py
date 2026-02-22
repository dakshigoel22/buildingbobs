"""
BuildingBobs — Vision Pipeline Configuration

Supports multiple providers:
  - gemini   → Google Gemini 2.0 Flash (API key required)
  - ollama   → Local or remote Ollama server (no API key needed)

API key loaded from .env file, CLI --api-key flag, or GEMINI_API_KEY env var.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _resolve_gemini_key() -> str:
    """Resolve Gemini API key from env var or .env file."""
    return os.getenv("GEMINI_API_KEY", "")


@dataclass
class VisionConfig:
    """Configuration for the vision analysis pipeline."""

    # Provider: "gemini" or "ollama"
    provider: str = "gemini"

    # Gemini settings
    api_key: str = field(default_factory=_resolve_gemini_key)
    model: str = "gemini-2.0-flash"

    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llava:7b"

    # Rate limiting — critical for free tier
    request_delay_sec: float = 4.0   # Delay between API calls (free tier: 15 RPM)
    max_retries: int = 3
    retry_delay_sec: float = 30.0    # Wait longer on rate limit errors
    timeout_sec: int = 120

    # Output
    save_raw_response: bool = True

    def validate(self) -> None:
        """Check that required config is present for the chosen provider."""
        if self.provider == "gemini":
            if not self.api_key:
                raise ValueError(
                    "No Gemini API key found. Set GEMINI_API_KEY in .env file "
                    "or pass --api-key on the command line.\n"
                    "Get a free key at: https://aistudio.google.com"
                )
        elif self.provider == "ollama":
            # Check that the Ollama URL looks valid
            if not self.ollama_url.startswith("http"):
                raise ValueError(
                    f"Invalid Ollama URL: {self.ollama_url}\n"
                    "Expected format: http://localhost:11434 or http://<vast-ip>:11434"
                )
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                "Supported: gemini, ollama"
            )
