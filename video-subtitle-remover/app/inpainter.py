"""
Video Inpainting Module
Fills in regions where subtitles were removed using various inpainting techniques.
"""

import cv2
import numpy as np
from typing import Optional, Literal
from enum import Enum
from pathlib import Path


class InpaintMethod(Enum):
    """Available inpainting methods."""
    OPENCV_NS = "opencv_ns"  # OpenCV Navier-Stokes based
    OPENCV_TELEA = "opencv_telea"  # OpenCV Fast Marching Method
    LAMA = "lama"  # LaMa deep learning based (best quality)


class VideoInpainter:
    """Handles inpainting of masked regions in video frames."""

    def __init__(
        self,
        method: InpaintMethod = InpaintMethod.OPENCV_TELEA,
        device: str = "cuda"
    ):
        """
        Initialize the inpainter.

        Args:
            method: Inpainting method to use
            device: Device for deep learning methods ('cuda' or 'cpu')
        """
        self.method = method
        self.device = device
        self.lama_model = None
        self._lama_initialized = False

    def _initialize_lama(self):
        """Initialize LaMa model for deep learning based inpainting."""
        if self._lama_initialized:
            return

        try:
            from simple_lama_inpainting import SimpleLama
            self.lama_model = SimpleLama()
            self._lama_initialized = True
            print("LaMa inpainting model initialized successfully")
        except ImportError:
            print("Warning: simple-lama-inpainting not installed. Falling back to OpenCV.")
            self.method = InpaintMethod.OPENCV_TELEA
        except Exception as e:
            print(f"Warning: Failed to initialize LaMa model: {e}. Falling back to OpenCV.")
            self.method = InpaintMethod.OPENCV_TELEA

    def inpaint(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        radius: int = 5
    ) -> np.ndarray:
        """
        Inpaint the masked regions in a frame.

        Args:
            frame: Video frame (BGR format)
            mask: Binary mask (255 for regions to inpaint)
            radius: Inpainting radius for OpenCV methods

        Returns:
            Inpainted frame
        """
        if mask is None or np.sum(mask) == 0:
            return frame

        if self.method == InpaintMethod.LAMA:
            return self._inpaint_lama(frame, mask)
        elif self.method == InpaintMethod.OPENCV_NS:
            return self._inpaint_opencv_ns(frame, mask, radius)
        else:  # OPENCV_TELEA
            return self._inpaint_opencv_telea(frame, mask, radius)

    def _inpaint_opencv_telea(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        radius: int = 5
    ) -> np.ndarray:
        """
        Inpaint using OpenCV's Fast Marching Method (Telea).
        Good for small regions, fast processing.
        """
        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Ensure mask is binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Apply inpainting
        result = cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)
        return result

    def _inpaint_opencv_ns(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        radius: int = 5
    ) -> np.ndarray:
        """
        Inpaint using OpenCV's Navier-Stokes based method.
        Better quality than Telea but slower.
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        result = cv2.inpaint(frame, mask, radius, cv2.INPAINT_NS)
        return result

    def _inpaint_lama(
        self,
        frame: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """
        Inpaint using LaMa (Large Mask Inpainting) model.
        Best quality, especially for larger regions.
        """
        self._initialize_lama()

        if self.lama_model is None:
            # Fallback to OpenCV if LaMa failed to initialize
            return self._inpaint_opencv_telea(frame, mask)

        try:
            from PIL import Image

            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Ensure mask is single channel
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            mask_image = Image.fromarray(mask)

            # Run LaMa inpainting
            result = self.lama_model(image, mask_image)

            # Convert back to BGR numpy array
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

            return result_bgr

        except Exception as e:
            print(f"LaMa inpainting failed: {e}. Using OpenCV fallback.")
            return self._inpaint_opencv_telea(frame, mask)


class TemporalInpainter:
    """
    Advanced inpainter that uses temporal information from adjacent frames
    to improve inpainting quality.
    """

    def __init__(self, base_inpainter: VideoInpainter):
        """
        Initialize temporal inpainter.

        Args:
            base_inpainter: Base VideoInpainter to use
        """
        self.base_inpainter = base_inpainter
        self.frame_buffer = []
        self.buffer_size = 5

    def inpaint_with_temporal(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        prev_frames: Optional[list] = None,
        next_frames: Optional[list] = None
    ) -> np.ndarray:
        """
        Inpaint using temporal information from surrounding frames.

        Args:
            frame: Current frame to inpaint
            mask: Binary mask for current frame
            prev_frames: List of previous frames
            next_frames: List of next frames

        Returns:
            Inpainted frame
        """
        if mask is None or np.sum(mask) == 0:
            return frame

        # For now, use simple temporal averaging approach
        # First do base inpainting
        result = self.base_inpainter.inpaint(frame, mask)

        # If we have adjacent frames, blend the inpainted region
        if prev_frames or next_frames:
            reference_frames = []
            if prev_frames:
                reference_frames.extend(prev_frames[-2:])
            if next_frames:
                reference_frames.extend(next_frames[:2])

            if reference_frames:
                # Average the masked region from reference frames
                ref_sum = np.zeros_like(frame, dtype=np.float32)
                for ref in reference_frames:
                    ref_sum += ref.astype(np.float32)
                ref_avg = (ref_sum / len(reference_frames)).astype(np.uint8)

                # Blend the inpainted result with reference average
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if len(mask.shape) == 2 else mask
                mask_float = mask_3ch.astype(np.float32) / 255.0

                # Use weighted blend
                result = (
                    result.astype(np.float32) * 0.7 +
                    ref_avg.astype(np.float32) * 0.3
                ).astype(np.uint8)

        return result


class PropaGateInpainter:
    """
    Frame propagation based inpainting.
    Uses optical flow to propagate pixels from adjacent frames.
    """

    def __init__(self):
        """Initialize the propagation inpainter."""
        pass

    def propagate_and_inpaint(
        self,
        frames: list,
        masks: list
    ) -> list:
        """
        Propagate clean pixels from frames without subtitles to fill gaps.

        Args:
            frames: List of video frames
            masks: List of corresponding masks

        Returns:
            List of inpainted frames
        """
        result_frames = []

        for i, (frame, mask) in enumerate(zip(frames, masks)):
            if mask is None or np.sum(mask) == 0:
                result_frames.append(frame)
                continue

            # Find nearest frame without subtitle in masked region
            clean_frame = self._find_clean_reference(frames, masks, i)

            if clean_frame is not None:
                # Use optical flow to warp the clean frame
                warped = self._warp_frame(clean_frame, frame)
                # Blend
                result = self._blend_with_mask(frame, warped, mask)
            else:
                # Fallback to simple inpainting
                result = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)

            result_frames.append(result)

        return result_frames

    def _find_clean_reference(
        self,
        frames: list,
        masks: list,
        current_idx: int,
        search_range: int = 30
    ) -> Optional[np.ndarray]:
        """Find a nearby frame that doesn't have subtitle in the same region."""
        current_mask = masks[current_idx]
        if current_mask is None:
            return None

        # Search forward and backward
        for offset in range(1, search_range):
            for idx in [current_idx - offset, current_idx + offset]:
                if 0 <= idx < len(frames):
                    other_mask = masks[idx]
                    if other_mask is None or np.sum(other_mask) == 0:
                        return frames[idx]
                    # Check if masks don't overlap
                    overlap = cv2.bitwise_and(current_mask, other_mask)
                    if np.sum(overlap) < np.sum(current_mask) * 0.5:
                        return frames[idx]

        return None

    def _warp_frame(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> np.ndarray:
        """Warp source frame to align with target using optical flow."""
        # Convert to grayscale for optical flow
        source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            source_gray, target_gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Create coordinate grid
        h, w = source.shape[:2]
        coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Apply flow
        coords[0] += flow[:, :, 1]
        coords[1] += flow[:, :, 0]

        # Warp the source image
        warped = cv2.remap(
            source,
            coords[1], coords[0],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )

        return warped

    def _blend_with_mask(
        self,
        original: np.ndarray,
        reference: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Blend reference frame into original using mask."""
        # Ensure mask is 3 channel
        if len(mask.shape) == 2:
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_3ch = mask

        # Feather the mask edges for smoother blending
        mask_float = mask_3ch.astype(np.float32) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (21, 21), 0)

        # Blend
        result = (
            original.astype(np.float32) * (1 - mask_blurred) +
            reference.astype(np.float32) * mask_blurred
        ).astype(np.uint8)

        return result


if __name__ == "__main__":
    # Test the inpainter
    inpainter = VideoInpainter(method=InpaintMethod.OPENCV_TELEA)
    # Add test code here
