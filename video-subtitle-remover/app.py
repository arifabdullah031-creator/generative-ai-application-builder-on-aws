"""
Video Subtitle Remover - Hugging Face Spaces Entry Point
"""

import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# Initialize components lazily
detector = None
inpainter = None


def get_detector():
    global detector
    if detector is None:
        import easyocr
        detector = easyocr.Reader(['en'], gpu=False, verbose=False)
    return detector


def process_video(video_file, language, subtitle_region, progress=gr.Progress()):
    """Process video to remove subtitles."""

    if video_file is None:
        return None, "Please upload a video file."

    progress(0, desc="Initializing...")
    reader = get_detector()

    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None, "Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output file
    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    subtitle_start_y = int(height * (1 - subtitle_region))
    frame_count = 0

    progress(0.1, desc="Processing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only scan bottom portion for subtitles
        roi = frame[subtitle_start_y:, :]

        # Detect text
        results = reader.readtext(roi)

        # Create mask for detected text
        mask = np.zeros((height, width), dtype=np.uint8)

        for bbox, text, conf in results:
            if conf > 0.3:
                pts = np.array(bbox, dtype=np.int32)
                pts[:, 1] += subtitle_start_y  # Adjust for ROI offset

                # Expand bounding box
                x_min = max(0, int(np.min(pts[:, 0])) - 10)
                x_max = min(width, int(np.max(pts[:, 0])) + 10)
                y_min = max(0, int(np.min(pts[:, 1])) - 10)
                y_max = min(height, int(np.max(pts[:, 1])) + 10)

                mask[y_min:y_max, x_min:x_max] = 255

        # Inpaint if text detected
        if np.sum(mask) > 0:
            frame = cv2.inpaint(frame, mask, 5, cv2.INPAINT_TELEA)

        writer.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            progress(0.1 + (frame_count / total_frames) * 0.85,
                    desc=f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    writer.release()

    progress(1.0, desc="Complete!")

    return output_path, f"Processed {frame_count} frames successfully!"


def preview_detection(video_file, frame_num, subtitle_region):
    """Preview subtitle detection on a single frame."""

    if video_file is None:
        return None, "Please upload a video file."

    reader = get_detector()

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_num = min(frame_num, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Could not read frame."

    subtitle_start_y = int(height * (1 - subtitle_region))
    roi = frame[subtitle_start_y:, :]

    results = reader.readtext(roi)

    for bbox, text, conf in results:
        if conf > 0.3:
            pts = np.array(bbox, dtype=np.int32)
            pts[:, 1] += subtitle_start_y
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", tuple(pts[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, f"Frame {frame_num}/{total_frames} - Found {len(results)} text regions"


# Create Gradio interface
with gr.Blocks(title="Video Subtitle Remover", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Video Subtitle Remover

    Remove hardcoded/burned-in subtitles from videos using AI-powered text detection.

    **How to use:**
    1. Upload a video file
    2. Adjust the subtitle region if needed
    3. Click "Remove Subtitles"
    """)

    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")

            language = gr.Dropdown(
                choices=["English", "Spanish", "French", "German", "Chinese", "Japanese", "Korean"],
                value="English",
                label="Subtitle Language"
            )

            subtitle_region = gr.Slider(
                minimum=0.1, maximum=0.5, value=0.35, step=0.05,
                label="Subtitle Region (bottom % of frame)"
            )

            process_btn = gr.Button("Remove Subtitles", variant="primary")

        with gr.Column():
            video_output = gr.Video(label="Processed Video")
            status = gr.Textbox(label="Status")

    with gr.Accordion("Preview Detection", open=False):
        frame_slider = gr.Slider(minimum=0, maximum=1000, value=0, step=1, label="Frame")
        preview_btn = gr.Button("Preview")
        preview_image = gr.Image(label="Detection Preview")
        preview_status = gr.Textbox(label="Preview Status")

    process_btn.click(
        process_video,
        inputs=[video_input, language, subtitle_region],
        outputs=[video_output, status]
    )

    preview_btn.click(
        preview_detection,
        inputs=[video_input, frame_slider, subtitle_region],
        outputs=[preview_image, preview_status]
    )

if __name__ == "__main__":
    demo.launch()
