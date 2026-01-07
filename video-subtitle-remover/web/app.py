"""
Web UI for Video Subtitle Remover
Built with Gradio for easy-to-use interface.
"""

import gradio as gr
import os
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.downloader import VideoDownloader, is_youtube_url
from app.processor import VideoProcessor, ProcessingConfig, get_video_info
from app.inpainter import InpaintMethod


# Global processor instance
processor = None
downloader = None


def initialize_components():
    """Initialize processor and downloader."""
    global processor, downloader

    temp_dir = tempfile.mkdtemp()
    downloader = VideoDownloader(output_dir=temp_dir)

    config = ProcessingConfig(
        languages=['en'],
        inpaint_method=InpaintMethod.OPENCV_TELEA,
        use_gpu=True,
        preserve_audio=True
    )
    processor = VideoProcessor(config)


def process_video(
    video_input,
    youtube_url: str,
    language: str,
    inpaint_method: str,
    subtitle_region: float,
    progress=gr.Progress()
):
    """
    Main processing function for the web UI.

    Args:
        video_input: Uploaded video file
        youtube_url: YouTube URL to download
        language: Language for OCR detection
        inpaint_method: Inpainting method to use
        subtitle_region: Portion of frame to scan for subtitles
        progress: Gradio progress tracker
    """
    global processor, downloader

    if processor is None:
        initialize_components()

    # Determine input source
    input_path = None
    video_info = {}

    progress(0, desc="Preparing video...")

    try:
        if youtube_url and youtube_url.strip():
            # Download from YouTube
            progress(0.1, desc="Downloading video from YouTube...")
            input_path, video_info = downloader.download(youtube_url.strip())
            progress(0.2, desc="Download complete!")
        elif video_input is not None:
            input_path = video_input
            video_info = get_video_info(input_path)
        else:
            return None, "Please provide a video file or YouTube URL."

        # Update processor configuration
        method_map = {
            "Fast (OpenCV Telea)": InpaintMethod.OPENCV_TELEA,
            "Quality (OpenCV NS)": InpaintMethod.OPENCV_NS,
            "Best (LaMa AI)": InpaintMethod.LAMA,
        }

        processor.config.languages = [language.lower()[:2]]
        processor.config.inpaint_method = method_map.get(inpaint_method, InpaintMethod.OPENCV_TELEA)
        processor.config.subtitle_region_ratio = subtitle_region

        # Reinitialize detector with new settings
        from app.detector import SubtitleDetector
        processor.detector = SubtitleDetector(
            languages=processor.config.languages,
            gpu=processor.config.use_gpu,
            subtitle_region_ratio=processor.config.subtitle_region_ratio
        )

        # Create output path
        output_dir = tempfile.mkdtemp()
        input_name = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{input_name}_clean.mp4")

        # Set up progress callback
        def update_progress(pct, msg):
            progress(0.2 + (pct / 100) * 0.75, desc=msg)

        processor.set_progress_callback(update_progress)

        # Process the video
        progress(0.2, desc="Processing video frames...")
        success = processor.process_video(input_path, output_path)

        if success:
            progress(1.0, desc="Complete!")
            status = f"Successfully processed video!\n"
            status += f"- Original: {video_info.get('duration', 'N/A')} seconds\n"
            status += f"- Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}\n"
            status += f"- Frames processed: {video_info.get('frame_count', 'N/A')}"
            return output_path, status
        else:
            return None, "Processing failed or was cancelled."

    except Exception as e:
        return None, f"Error: {str(e)}"


def preview_frame(
    video_input,
    youtube_url: str,
    frame_number: int,
    language: str
):
    """Preview subtitle detection on a single frame."""
    global processor, downloader

    if processor is None:
        initialize_components()

    try:
        import cv2

        # Get video path
        if youtube_url and youtube_url.strip():
            input_path, _ = downloader.download(youtube_url.strip())
        elif video_input is not None:
            input_path = video_input
        else:
            return None, "Please provide a video file or YouTube URL."

        # Open video and get frame
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Clamp frame number
        frame_number = max(0, min(frame_number, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, "Could not read frame."

        # Update detector language
        processor.config.languages = [language.lower()[:2]]
        from app.detector import SubtitleDetector
        processor.detector = SubtitleDetector(
            languages=processor.config.languages,
            gpu=processor.config.use_gpu,
            subtitle_region_ratio=processor.config.subtitle_region_ratio
        )

        # Get preview with detection boxes
        preview = processor.preview_detection(frame)

        # Convert BGR to RGB for display
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        return preview_rgb, f"Showing frame {frame_number}/{total_frames}"

    except Exception as e:
        return None, f"Error: {str(e)}"


def create_ui():
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Video Subtitle Remover",
        theme=gr.themes.Soft()
    ) as app:

        gr.Markdown("""
        # Video Subtitle Remover

        Remove hardcoded/burned-in subtitles from videos using AI-powered detection and inpainting.

        **How to use:**
        1. Upload a video file OR paste a YouTube URL
        2. Configure detection settings (language, region)
        3. Choose inpainting quality
        4. Click "Remove Subtitles" and wait for processing

        **Note:** Processing time depends on video length and chosen quality settings.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"]
                )

                youtube_url = gr.Textbox(
                    label="Or paste YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=..."
                )

                gr.Markdown("### Settings")

                language = gr.Dropdown(
                    choices=[
                        "English", "Spanish", "French", "German",
                        "Italian", "Portuguese", "Russian", "Chinese",
                        "Japanese", "Korean", "Arabic", "Hindi"
                    ],
                    value="English",
                    label="Subtitle Language"
                )

                inpaint_method = gr.Radio(
                    choices=[
                        "Fast (OpenCV Telea)",
                        "Quality (OpenCV NS)",
                        "Best (LaMa AI)"
                    ],
                    value="Fast (OpenCV Telea)",
                    label="Inpainting Quality"
                )

                subtitle_region = gr.Slider(
                    minimum=0.1,
                    maximum=0.5,
                    value=0.35,
                    step=0.05,
                    label="Subtitle Region (bottom % of frame)"
                )

                process_btn = gr.Button("Remove Subtitles", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Output")

                output_video = gr.Video(label="Processed Video")
                status_text = gr.Textbox(label="Status", lines=4)

        # Preview section
        with gr.Accordion("Preview Detection", open=False):
            gr.Markdown("Test subtitle detection on a single frame before processing.")

            with gr.Row():
                frame_slider = gr.Slider(
                    minimum=0,
                    maximum=1000,
                    value=0,
                    step=1,
                    label="Frame Number"
                )
                preview_btn = gr.Button("Preview Frame")

            preview_image = gr.Image(label="Detection Preview")
            preview_status = gr.Textbox(label="Preview Status")

        # Event handlers
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                youtube_url,
                language,
                inpaint_method,
                subtitle_region
            ],
            outputs=[output_video, status_text]
        )

        preview_btn.click(
            fn=preview_frame,
            inputs=[video_input, youtube_url, frame_slider, language],
            outputs=[preview_image, preview_status]
        )

        gr.Markdown("""
        ---
        ### Tips
        - **Fast processing:** Use "Fast (OpenCV Telea)" for quick results
        - **Better quality:** Use "Best (LaMa AI)" for cleaner results (slower)
        - **Adjust region:** If subtitles are missed, increase the subtitle region slider
        - **Multiple languages:** The detector works best with the correct language selected

        ### Limitations
        - Works best with clearly visible subtitles
        - Complex backgrounds may show some artifacts after inpainting
        - Very stylized or decorative fonts may not be detected
        """)

    return app


# Create and launch the app
app = create_ui()

if __name__ == "__main__":
    initialize_components()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
