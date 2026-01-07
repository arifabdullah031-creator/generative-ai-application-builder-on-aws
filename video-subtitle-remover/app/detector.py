"""
Text/Subtitle Detection Module
Detects text regions in video frames using EasyOCR.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import easyocr


@dataclass
class TextRegion:
    """Represents a detected text region in an image."""
    bbox: List[List[int]]  # Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str
    confidence: float
    center_y: int  # Y coordinate of center (for subtitle detection)

    @property
    def is_subtitle_region(self) -> bool:
        """Check if this text region is likely a subtitle (bottom portion of frame)."""
        return True  # Will be refined based on position


class SubtitleDetector:
    """Detects subtitle text regions in video frames."""

    def __init__(
        self,
        languages: List[str] = ['en'],
        gpu: bool = True,
        subtitle_region_ratio: float = 0.35,
        min_confidence: float = 0.3
    ):
        """
        Initialize the subtitle detector.

        Args:
            languages: List of language codes for OCR (e.g., ['en', 'es', 'fr'])
            gpu: Whether to use GPU acceleration
            subtitle_region_ratio: Portion of frame height to consider as subtitle region (from bottom)
            min_confidence: Minimum confidence threshold for text detection
        """
        self.languages = languages
        self.gpu = gpu
        self.subtitle_region_ratio = subtitle_region_ratio
        self.min_confidence = min_confidence
        self.reader = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of EasyOCR reader."""
        if not self._initialized:
            print(f"Initializing OCR reader with languages: {self.languages}")
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            self._initialized = True

    def detect(
        self,
        frame: np.ndarray,
        detect_full_frame: bool = False
    ) -> List[TextRegion]:
        """
        Detect text regions in a video frame.

        Args:
            frame: Video frame as numpy array (BGR format)
            detect_full_frame: If True, detect text in entire frame, not just subtitle region

        Returns:
            List of detected TextRegion objects
        """
        self._initialize()

        height, width = frame.shape[:2]
        regions = []

        # Determine the region to scan for subtitles
        if detect_full_frame:
            scan_frame = frame
            y_offset = 0
        else:
            # Only scan the bottom portion of the frame where subtitles typically appear
            subtitle_start_y = int(height * (1 - self.subtitle_region_ratio))
            scan_frame = frame[subtitle_start_y:, :]
            y_offset = subtitle_start_y

        # Run OCR detection
        results = self.reader.readtext(scan_frame)

        for bbox, text, confidence in results:
            if confidence < self.min_confidence:
                continue

            # Adjust bbox coordinates to full frame coordinates
            adjusted_bbox = []
            for point in bbox:
                adjusted_bbox.append([int(point[0]), int(point[1]) + y_offset])

            # Calculate center Y position
            center_y = int(np.mean([p[1] for p in adjusted_bbox]))

            region = TextRegion(
                bbox=adjusted_bbox,
                text=text,
                confidence=confidence,
                center_y=center_y
            )
            regions.append(region)

        return regions

    def detect_with_tracking(
        self,
        frame: np.ndarray,
        previous_regions: Optional[List[TextRegion]] = None,
        iou_threshold: float = 0.3
    ) -> List[TextRegion]:
        """
        Detect text regions with temporal tracking from previous frame.
        This helps maintain consistency across frames.

        Args:
            frame: Current video frame
            previous_regions: Text regions from previous frame
            iou_threshold: IOU threshold for matching regions

        Returns:
            List of detected TextRegion objects
        """
        current_regions = self.detect(frame)

        if previous_regions is None:
            return current_regions

        # If no text detected but previous frame had subtitles,
        # the subtitle might still be there (OCR missed it)
        # Return previous regions with lower confidence
        if not current_regions and previous_regions:
            return []  # Trust current detection

        return current_regions

    def create_mask(
        self,
        frame_shape: Tuple[int, int, int],
        regions: List[TextRegion],
        expansion: int = 5
    ) -> np.ndarray:
        """
        Create a binary mask for the detected text regions.

        Args:
            frame_shape: Shape of the video frame (height, width, channels)
            regions: List of TextRegion objects
            expansion: Pixels to expand the mask around detected text

        Returns:
            Binary mask (255 for text regions, 0 for background)
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for region in regions:
            # Get bounding box points
            pts = np.array(region.bbox, dtype=np.int32)

            # Expand the bounding box
            center = np.mean(pts, axis=0)
            expanded_pts = []
            for pt in pts:
                direction = pt - center
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                expanded_pt = pt + direction * expansion
                expanded_pts.append(expanded_pt)

            expanded_pts = np.array(expanded_pts, dtype=np.int32)

            # Fill the polygon
            cv2.fillPoly(mask, [expanded_pts], 255)

        # Dilate the mask slightly for better coverage
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def create_rectangular_mask(
        self,
        frame_shape: Tuple[int, int, int],
        regions: List[TextRegion],
        padding: int = 10
    ) -> np.ndarray:
        """
        Create a rectangular binary mask for detected text regions.
        This creates cleaner masks that work better with some inpainting methods.

        Args:
            frame_shape: Shape of the video frame
            regions: List of TextRegion objects
            padding: Padding around detected text

        Returns:
            Binary mask
        """
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        for region in regions:
            pts = np.array(region.bbox)
            x_min = max(0, int(np.min(pts[:, 0])) - padding)
            x_max = min(width, int(np.max(pts[:, 0])) + padding)
            y_min = max(0, int(np.min(pts[:, 1])) - padding)
            y_max = min(height, int(np.max(pts[:, 1])) + padding)

            mask[y_min:y_max, x_min:x_max] = 255

        return mask


class SubtitleRegionDetector:
    """
    Alternative detector that focuses on finding the subtitle region
    without OCR - useful for consistent subtitle styles.
    """

    def __init__(self, subtitle_region_ratio: float = 0.25):
        """
        Initialize the region detector.

        Args:
            subtitle_region_ratio: Portion of frame to consider as subtitle region
        """
        self.subtitle_region_ratio = subtitle_region_ratio

    def detect_by_color(
        self,
        frame: np.ndarray,
        text_colors: List[Tuple[int, int, int]] = [(255, 255, 255), (255, 255, 0)]
    ) -> np.ndarray:
        """
        Detect subtitle regions by color (for common white/yellow subtitles).

        Args:
            frame: Video frame
            text_colors: List of BGR colors to detect

        Returns:
            Binary mask of detected regions
        """
        height, width = frame.shape[:2]
        subtitle_start_y = int(height * (1 - self.subtitle_region_ratio))

        # Only process subtitle region
        roi = frame[subtitle_start_y:, :]
        mask = np.zeros((height, width), dtype=np.uint8)

        for color in text_colors:
            # Create color range
            lower = np.array([max(0, c - 30) for c in color])
            upper = np.array([min(255, c + 30) for c in color])

            # Detect color
            color_mask = cv2.inRange(roi, lower, upper)

            # Add to main mask
            mask[subtitle_start_y:, :] = cv2.bitwise_or(
                mask[subtitle_start_y:, :],
                color_mask
            )

        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask


if __name__ == "__main__":
    # Test the detector
    detector = SubtitleDetector(languages=['en'])
    # Add test code here
