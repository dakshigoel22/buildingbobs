"""
BuildingBobs — Motion Deblurring

Lightweight deblurring for MVP using OpenCV Wiener deconvolution.
Applied selectively to frames flagged as moderately blurry — frames
below a minimum threshold are discarded as unsalvageable.

Note: For production, upgrade to DeblurGAN-v2 / NAFNet on vast.ai GPU.
"""

import logging

import cv2
import numpy as np

from .config import DeblurConfig

logger = logging.getLogger(__name__)


def create_motion_kernel(size: int, angle: float = 0.0) -> np.ndarray:
    """
    Create a linear motion blur kernel.

    Args:
        size: Length of the motion blur in pixels.
        angle: Angle of the motion in degrees.

    Returns:
        Normalised motion blur kernel.
    """
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2

    # Create a horizontal line kernel, then rotate
    kernel[center, :] = 1.0

    if angle != 0.0:
        rotation = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (size, size))

    return kernel / kernel.sum()


def wiener_deconvolve(
    image: np.ndarray,
    kernel: np.ndarray,
    snr: float = 25.0,
) -> np.ndarray:
    """
    Apply Wiener deconvolution to deblur an image.

    Args:
        image: Blurred input image (BGR or grayscale).
        kernel: Estimated blur kernel.
        snr: Signal-to-noise ratio estimate.

    Returns:
        Deblurred image.
    """
    # Work in float
    img_float = image.astype(np.float64) / 255.0

    if len(img_float.shape) == 3:
        # Process each channel separately
        channels = cv2.split(img_float)
        deblurred_channels = []
        for ch in channels:
            deblurred_ch = _wiener_channel(ch, kernel, snr)
            deblurred_channels.append(deblurred_ch)
        result = cv2.merge(deblurred_channels)
    else:
        result = _wiener_channel(img_float, kernel, snr)

    # Convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result


def _wiener_channel(channel: np.ndarray, kernel: np.ndarray, snr: float) -> np.ndarray:
    """Apply Wiener deconvolution to a single channel."""
    h, w = channel.shape

    # Pad kernel to image size
    kernel_padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = kernel.shape
    y_offset = h // 2 - kh // 2
    x_offset = w // 2 - kw // 2
    kernel_padded[y_offset:y_offset + kh, x_offset:x_offset + kw] = kernel

    # FFT
    F_img = np.fft.fft2(channel)
    F_kernel = np.fft.fft2(kernel_padded)

    # Wiener filter: H* / (|H|^2 + 1/SNR)
    F_kernel_conj = np.conj(F_kernel)
    F_kernel_sq = np.abs(F_kernel) ** 2
    noise_inv = 1.0 / snr

    F_result = (F_kernel_conj / (F_kernel_sq + noise_inv)) * F_img

    # Inverse FFT
    result = np.real(np.fft.ifft2(F_result))

    # Shift to correct position
    result = np.fft.fftshift(result)

    return np.clip(result, 0, 1)


def deblur_frame(
    frame: np.ndarray,
    config: DeblurConfig | None = None,
) -> np.ndarray | None:
    """
    Attempt to deblur a frame using Wiener deconvolution.

    Args:
        frame: BGR image as numpy array.
        config: Deblur parameters.

    Returns:
        Deblurred frame, or None if deblurring not attempted.
    """
    if config is None:
        config = DeblurConfig()

    kernel = create_motion_kernel(config.kernel_size)
    deblurred = wiener_deconvolve(frame, kernel, config.snr)
    return deblurred


def deblur_keyframe_if_needed(
    frame: np.ndarray,
    quality_score: float,
    config: DeblurConfig | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Conditionally deblur a frame based on its quality score.

    Args:
        frame: BGR image.
        quality_score: Overall quality score (0-100).
        config: Deblur parameters.

    Returns:
        Tuple of (processed_frame, was_deblurred).
    """
    if config is None:
        config = DeblurConfig()

    # Too blurry to recover — skip / discard
    if quality_score < config.skip_threshold:
        logger.debug(f"Frame quality {quality_score:.1f} below skip threshold — discarding")
        return frame, False

    # Sharp enough — no deblurring needed
    if quality_score >= config.apply_threshold:
        return frame, False

    # In the "recoverable" range — attempt deblurring
    logger.debug(f"Deblurring frame with quality score {quality_score:.1f}")
    deblurred = deblur_frame(frame, config)
    if deblurred is not None:
        return deblurred, True

    return frame, False
