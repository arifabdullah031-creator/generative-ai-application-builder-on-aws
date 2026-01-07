"""
Video Processing Pipeline
Main module that orchestrates subtitle detection and removal.
"""

import cv2
import numpy as np
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading

from .detector import SubtitleDetector, TextRegion
from .inpainter import VideoInpainter, InpaintMethod


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    languages: List[str] = None
    subtitle_region_ratio: float = 0.35
    min_confidence: float = 0.3
    inpaint_method: InpaintMethod = InpaintMethod.OPENCV_TELEA
    mask_expansion: int = 10
    use_gpu: bool = True
    batch_size: int = 1
    preserve_audio: bool = True
    output_quality: int = 23  # CRF value (lower = better quality, 18-28 recommended)

    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en']


class VideoProcessor:
    """Main video processing class for subtitle removal."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the video processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.detector = SubtitleDetector(
            languages=self.config.languages,
            gpu=self.config.use_gpu,
            subtitle_region_ratio=self.config.subtitle_region_ratio,
            min_confidence=self.config.min_confidence
        )
        self.inpainter = VideoInpainter(
            method=self.config.inpaint_method,
            device="cuda" if self.config.use_gpu else "cpu"
        )
        self._progress_callback = None
        self._cancel_flag = threading.Event()

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback

    def cancel(self):
        """Cancel ongoing processing."""
        self._cancel_flag.set()

    def _update_progress(self, progress: float, message: str):
        """Update progress via callback."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def process_video(
        self,
        input_path: str,
        output_path: str,
        preview_callback: Optional[Callable[[np.ndarray], None]] = None
    ) -> bool:
        """
        Process a video to remove subtitles.

        Args:
            input_path: Path to input video
            output_path: Path for output video
            preview_callback: Optional callback to receive preview frames

        Returns:
            True if successful, False otherwise
        """
        self._cancel_flag.clear()

        # Open input video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._update_progress(0, f"Processing video: {total_frames} frames at {fps} fps")

        # Create temporary file for video without audio
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video_path = temp_video.name
        temp_video.close()

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            raise ValueError("Could not create output video writer")

        try:
            frame_count = 0
            pbar = tqdm(total=total_frames, desc="Processing frames")

            while True:
                if self._cancel_flag.is_set():
                    print("Processing cancelled")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                # Detect subtitles
                regions = self.detector.detect(frame)

                # Create mask and inpaint if subtitles found
                if regions:
                    mask = self.detector.create_rectangular_mask(
                        frame.shape,
                        regions,
                        padding=self.config.mask_expansion
                    )
                    processed_frame = self.inpainter.inpaint(frame, mask)
                else:
                    processed_frame = frame

                # Write processed frame
                writer.write(processed_frame)

                # Send preview if callback provided
                if preview_callback and frame_count % 30 == 0:
                    preview_callback(processed_frame)

                frame_count += 1
                pbar.update(1)

                # Update progress
                progress = (frame_count / total_frames) * 100
                self._update_progress(progress, f"Processed {frame_count}/{total_frames} frames")

            pbar.close()

        finally:
            cap.release()
            writer.release()

        if self._cancel_flag.is_set():
            os.unlink(temp_video_path)
            return False

        # Merge with original audio if needed
        self._update_progress(95, "Merging audio...")

        if self.config.preserve_audio:
            success = self._merge_audio(input_path, temp_video_path, output_path)
        else:
            # Just copy the temp video to output
            import shutil
            shutil.move(temp_video_path, output_path)
            success = True

        # Clean up temp file
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)

        self._update_progress(100, "Processing complete!")
        return success

    def _merge_audio(
        self,
        original_video: str,
        processed_video: str,
        output_path: str
    ) -> bool:
        """
        Merge audio from original video with processed video.

        Args:
            original_video: Path to original video with audio
            processed_video: Path to processed video without audio
            output_path: Path for final output

        Returns:
            True if successful
        """
        try:
            # Use FFmpeg to merge audio and video
            cmd = [
                'ffmpeg', '-y',
                '-i', processed_video,
                '-i', original_video,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', str(self.config.output_quality),
                '-c:a', 'aac',
                '-map', '0:v:0',
                '-map', '1:a:0?',
                '-shortest',
                output_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                print(f"FFmpeg warning: {result.stderr}")
                # If audio merge fails, just copy video without audio
                import shutil
                shutil.copy(processed_video, output_path)

            return True

        except FileNotFoundError:
            print("FFmpeg not found. Output video will not have audio.")
            import shutil
            shutil.copy(processed_video, output_path)
            return True
        except Exception as e:
            print(f"Error merging audio: {e}")
            import shutil
            shutil.copy(processed_video, output_path)
            return True

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[TextRegion]]:
        """
        Process a single frame.

        Args:
            frame: Input frame

        Returns:
            Tuple of (processed frame, detected regions)
        """
        regions = self.detector.detect(frame)

        if regions:
            mask = self.detector.create_rectangular_mask(
                frame.shape,
                regions,
                padding=self.config.mask_expansion
            )
            processed = self.inpainter.inpaint(frame, mask)
        else:
            processed = frame

        return processed, regions

    def preview_detection(self, frame: np.ndarray) -> np.ndarray:
        """
        Preview subtitle detection on a frame (draw boxes around detected text).

        Args:
            frame: Input frame

        Returns:
            Frame with detection boxes drawn
        """
        regions = self.detector.detect(frame)
        preview = frame.copy()

        for region in regions:
            pts = np.array(region.bbox, dtype=np.int32)
            cv2.polylines(preview, [pts], True, (0, 255, 0), 2)
            # Draw confidence
            cv2.putText(
                preview,
                f"{region.confidence:.2f}",
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )

        return preview


class BatchVideoProcessor:
    """Process multiple videos in batch."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """Initialize batch processor."""
        self.config = config or ProcessingConfig()
        self.processor = VideoProcessor(config)

    def process_batch(
        self,
        input_paths: List[str],
        output_dir: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None
    ) -> List[Tuple[str, bool]]:
        """
        Process multiple videos.

        Args:
            input_paths: List of input video paths
            output_dir: Directory for output videos
            progress_callback: Callback(current_index, total, status)

        Returns:
            List of (output_path, success) tuples
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, input_path in enumerate(input_paths):
            if progress_callback:
                progress_callback(i + 1, len(input_paths), f"Processing: {Path(input_path).name}")

            output_name = f"{Path(input_path).stem}_no_subtitles.mp4"
            output_path = os.path.join(output_dir, output_name)

            try:
                success = self.processor.process_video(input_path, output_path)
                results.append((output_path, success))
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
                results.append((output_path, False))

        return results


def get_video_info(video_path: str) -> dict:
    """Get information about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return info


if __name__ == "__main__":
    # Test the processor
    config = ProcessingConfig(
        languages=['en'],
        inpaint_method=InpaintMethod.OPENCV_TELEA
    )
    processor = VideoProcessor(config)
    # Add test code here
