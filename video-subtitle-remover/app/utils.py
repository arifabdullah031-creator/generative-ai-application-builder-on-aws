"""
Utility functions for Video Subtitle Remover.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> str:
    """Ensure a directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_temp_dir() -> str:
    """Get a temporary directory for processing."""
    temp_dir = os.path.join(tempfile.gettempdir(), "video_subtitle_remover")
    return ensure_dir(temp_dir)


def cleanup_temp_files(keep_recent: int = 5):
    """Clean up old temporary files."""
    temp_dir = get_temp_dir()
    if not os.path.exists(temp_dir):
        return

    files = []
    for f in Path(temp_dir).iterdir():
        if f.is_file():
            files.append((f, f.stat().st_mtime))

    # Sort by modification time
    files.sort(key=lambda x: x[1], reverse=True)

    # Remove old files
    for f, _ in files[keep_recent:]:
        try:
            os.unlink(f)
        except Exception:
            pass


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0


def format_duration(seconds: int) -> str:
    """Format duration in seconds to HH:MM:SS."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available."""
    return shutil.which("ffmpeg") is not None


def check_gpu_available() -> bool:
    """Check if GPU (CUDA) is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_system_info() -> dict:
    """Get system information for debugging."""
    import platform

    info = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "ffmpeg_available": check_ffmpeg(),
        "gpu_available": check_gpu_available(),
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "Not installed"

    return info


def validate_video_file(path: str) -> bool:
    """Validate that a file is a valid video."""
    import cv2

    if not os.path.exists(path):
        return False

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False

    # Try to read a frame
    ret, _ = cap.read()
    cap.release()

    return ret


def estimate_processing_time(
    video_duration: int,
    method: str = "telea",
    has_gpu: bool = True
) -> str:
    """
    Estimate processing time based on video duration and settings.
    Returns human-readable string.
    """
    # Rough estimates (frames per second processing rate)
    fps_estimates = {
        "telea": 15 if has_gpu else 10,
        "ns": 10 if has_gpu else 5,
        "lama": 5 if has_gpu else 1,
    }

    fps_rate = fps_estimates.get(method, 10)

    # Assume 30 fps video
    total_frames = video_duration * 30
    estimated_seconds = total_frames / fps_rate

    return format_duration(int(estimated_seconds))
