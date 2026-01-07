"""
YouTube Video Downloader Module
Downloads videos from YouTube using yt-dlp.
"""

import os
import yt_dlp
from typing import Optional, Tuple
from pathlib import Path


class VideoDownloader:
    """Handles downloading videos from YouTube and other sources."""

    def __init__(self, output_dir: str = "temp"):
        """
        Initialize the downloader.

        Args:
            output_dir: Directory to save downloaded videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        url: str,
        max_resolution: int = 1080,
        audio: bool = True
    ) -> Tuple[str, dict]:
        """
        Download a video from URL.

        Args:
            url: YouTube URL or other supported video URL
            max_resolution: Maximum video resolution (default 1080p)
            audio: Whether to include audio track

        Returns:
            Tuple of (downloaded file path, video metadata)
        """
        # Configure yt-dlp options
        ydl_opts = {
            'outtmpl': str(self.output_dir / '%(id)s.%(ext)s'),
            'format': f'bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]' if audio
                      else f'bestvideo[height<={max_resolution}]',
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(url, download=False)
            video_id = info.get('id', 'video')

            # Download the video
            ydl.download([url])

            # Find the downloaded file
            downloaded_file = self.output_dir / f"{video_id}.mp4"

            # If mp4 doesn't exist, try to find other formats
            if not downloaded_file.exists():
                for ext in ['mp4', 'webm', 'mkv', 'avi']:
                    potential_file = self.output_dir / f"{video_id}.{ext}"
                    if potential_file.exists():
                        downloaded_file = potential_file
                        break

            metadata = {
                'id': video_id,
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'resolution': info.get('resolution', 'Unknown'),
                'fps': info.get('fps', 30),
                'uploader': info.get('uploader', 'Unknown'),
            }

            return str(downloaded_file), metadata

    def download_from_file(self, file_path: str) -> Tuple[str, dict]:
        """
        Copy a local video file to the working directory.

        Args:
            file_path: Path to local video file

        Returns:
            Tuple of (copied file path, basic metadata)
        """
        import shutil
        import cv2

        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # Copy to temp directory
        dest = self.output_dir / source.name
        shutil.copy2(source, dest)

        # Extract basic metadata using OpenCV
        cap = cv2.VideoCapture(str(dest))
        metadata = {
            'id': source.stem,
            'title': source.name,
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        cap.release()

        return str(dest), metadata


def is_youtube_url(url: str) -> bool:
    """Check if the URL is a YouTube URL."""
    youtube_patterns = [
        'youtube.com/watch',
        'youtu.be/',
        'youtube.com/shorts/',
        'youtube.com/v/',
        'youtube.com/embed/',
    ]
    return any(pattern in url for pattern in youtube_patterns)


if __name__ == "__main__":
    # Test the downloader
    downloader = VideoDownloader(output_dir="temp")
    # Add test code here
