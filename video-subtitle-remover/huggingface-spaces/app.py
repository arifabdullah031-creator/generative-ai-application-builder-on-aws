"""
Video Subtitle Remover - Hugging Face Spaces
Remove hardcoded subtitles from videos using AI
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os

# Global detector (lazy loaded)
detector = None


def get_detector():
    """Lazy load EasyOCR detector."""
    global detector
    if detector is None:
        import easyocr
        print("Loading OCR model...")
        detector = easyocr.Reader(['en', 'es', 'fr', 'de', 'it', 'pt'], gpu=False, verbose=False)
        print("OCR model loaded!")
    return detector


def process_video(video_file, subtitle_region, progress=gr.Progress()):
    """
    Process video to remove subtitles.

    Args:
        video_file: Input video file path
        subtitle_region: Portion of frame to scan (from bottom)
        progress: Gradio progress tracker
    """
    if video_file is None:
        return None, "❌ Please upload a video file."

    progress(0, desc="🚀 Starting...")

    try:
        # Load detector
        progress(0.05, desc="🤖 Loading AI model...")
        reader = get_detector()

        # Open video
        progress(0.1, desc="📹 Opening video...")
        cap = cv2.VideoCapture(video_file)

        if not cap.isOpened():
            return None, "❌ Could not open video file."

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            return None, "❌ Video has no frames."

        # Create output file
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            return None, "❌ Could not create output video."

        # Calculate subtitle region
        subtitle_start_y = int(height * (1 - subtitle_region))

        frame_count = 0
        subtitles_removed = 0

        progress(0.15, desc="🎬 Processing frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only scan bottom portion for subtitles
            roi = frame[subtitle_start_y:, :]

            # Detect text in subtitle region
            try:
                results = reader.readtext(roi, paragraph=False)
            except:
                results = []

            # Create mask for detected text
            mask = np.zeros((height, width), dtype=np.uint8)
            detected_text = False

            for bbox, text, conf in results:
                if conf > 0.25 and len(text.strip()) > 1:
                    detected_text = True
                    pts = np.array(bbox, dtype=np.int32)
                    pts[:, 1] += subtitle_start_y  # Adjust Y for ROI offset

                    # Get bounding rectangle with padding
                    x_min = max(0, int(np.min(pts[:, 0])) - 15)
                    x_max = min(width, int(np.max(pts[:, 0])) + 15)
                    y_min = max(0, int(np.min(pts[:, 1])) - 10)
                    y_max = min(height, int(np.max(pts[:, 1])) + 10)

                    mask[y_min:y_max, x_min:x_max] = 255

            # Inpaint if text was detected
            if detected_text:
                frame = cv2.inpaint(frame, mask, 7, cv2.INPAINT_TELEA)
                subtitles_removed += 1

            writer.write(frame)
            frame_count += 1

            # Update progress every 10 frames
            if frame_count % 10 == 0:
                pct = 0.15 + (frame_count / total_frames) * 0.80
                progress(pct, desc=f"🎬 Frame {frame_count}/{total_frames}")

        cap.release()
        writer.release()

        progress(0.98, desc="✅ Finalizing...")

        # Create status message
        duration = frame_count / fps if fps > 0 else 0
        status = f"""✅ **Processing Complete!**

📊 **Statistics:**
- Frames processed: {frame_count}
- Frames with subtitles: {subtitles_removed}
- Video duration: {duration:.1f} seconds
- Resolution: {width}x{height}

⬇️ **Download your clean video below!**"""

        progress(1.0, desc="✅ Done!")
        return output_path, status

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def preview_frame(video_file, frame_num, subtitle_region):
    """Preview subtitle detection on a single frame."""

    if video_file is None:
        return None, "Please upload a video first."

    try:
        reader = get_detector()

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Clamp frame number
        frame_num = max(0, min(int(frame_num), total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None, "Could not read frame."

        # Detect text
        subtitle_start_y = int(height * (1 - subtitle_region))
        roi = frame[subtitle_start_y:, :]

        results = reader.readtext(roi, paragraph=False)

        # Draw detection boxes
        for bbox, text, conf in results:
            if conf > 0.25:
                pts = np.array(bbox, dtype=np.int32)
                pts[:, 1] += subtitle_start_y

                # Draw green box
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                # Draw label
                label = f"{text[:20]}... ({conf:.0%})" if len(text) > 20 else f"{text} ({conf:.0%})"
                cv2.putText(frame, label, (pts[0][0], pts[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw subtitle region line
        cv2.line(frame, (0, subtitle_start_y), (width, subtitle_start_y), (255, 0, 0), 2)
        cv2.putText(frame, "Subtitle Region", (10, subtitle_start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        status = f"Frame {frame_num}/{total_frames} | Found {len(results)} text regions"
        return frame_rgb, status

    except Exception as e:
        return None, f"Error: {str(e)}"


# Build Gradio Interface
with gr.Blocks(
    title="Video Subtitle Remover",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 1200px !important; }
        .title { text-align: center; margin-bottom: 1rem; }
    """
) as demo:

    gr.HTML("""
        <div class="title">
            <h1>🎬 Video Subtitle Remover</h1>
            <p>Remove hardcoded/burned-in subtitles from videos using AI</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Video")

            video_input = gr.Video(
                label="Upload your video",
                sources=["upload"]
            )

            gr.Markdown("### ⚙️ Settings")

            subtitle_region = gr.Slider(
                minimum=0.15,
                maximum=0.50,
                value=0.35,
                step=0.05,
                label="Subtitle Region (bottom % of frame)",
                info="Increase if subtitles are higher up"
            )

            process_btn = gr.Button(
                "🚀 Remove Subtitles",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=1):
            gr.Markdown("### 📥 Result")

            video_output = gr.Video(label="Processed Video")
            status_output = gr.Markdown(label="Status")

    with gr.Accordion("🔍 Preview Detection (Optional)", open=False):
        gr.Markdown("Test subtitle detection on a single frame before processing the entire video.")

        with gr.Row():
            frame_slider = gr.Slider(
                minimum=0,
                maximum=1000,
                value=100,
                step=1,
                label="Frame Number"
            )
            preview_btn = gr.Button("👁️ Preview Frame")

        preview_image = gr.Image(label="Detection Preview")
        preview_status = gr.Textbox(label="Detection Status")

    gr.Markdown("""
    ---
    ### 📖 How to Use
    1. **Upload** your video (MP4, AVI, MOV, MKV)
    2. **Adjust** the subtitle region slider if needed
    3. **Click** "Remove Subtitles" and wait
    4. **Download** your clean video!

    ### ⚠️ Notes
    - Works best with **clearly visible** subtitles
    - Processing time depends on video length
    - Shorter videos (under 2 minutes) work best on free tier

    ### 🛠️ Powered by
    - **EasyOCR** for text detection
    - **OpenCV** for video processing & inpainting
    """)

    # Event handlers
    process_btn.click(
        fn=process_video,
        inputs=[video_input, subtitle_region],
        outputs=[video_output, status_output]
    )

    preview_btn.click(
        fn=preview_frame,
        inputs=[video_input, frame_slider, subtitle_region],
        outputs=[preview_image, preview_status]
    )


# Launch
if __name__ == "__main__":
    demo.launch()
