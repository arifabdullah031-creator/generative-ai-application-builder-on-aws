"""
Flask Web Server for Video Subtitle Remover
Run this to start the local web app.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import threading
import cv2
import numpy as np
import tempfile
from pathlib import Path

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Store for processing tasks
tasks = {}

# Lazy load detector
detector = None


def get_detector(language='en'):
    global detector
    if detector is None:
        import easyocr
        print("Loading OCR model... (this may take a moment)")
        detector = easyocr.Reader([language], gpu=False, verbose=False)
    return detector


def process_video_task(task_id, video_path, output_path, language, quality, region):
    """Background task to process video."""
    global tasks

    try:
        tasks[task_id]['status'] = 'processing'
        tasks[task_id]['progress'] = 0
        tasks[task_id]['message'] = 'Initializing...'

        # Get detector
        tasks[task_id]['message'] = 'Loading AI model...'
        reader = get_detector(language)

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        subtitle_start_y = int(height * (1 - region))
        frame_count = 0

        # Select inpainting method
        inpaint_method = cv2.INPAINT_TELEA
        inpaint_radius = 5
        if quality == 'quality':
            inpaint_method = cv2.INPAINT_NS
            inpaint_radius = 7

        tasks[task_id]['message'] = 'Processing frames...'

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect text in subtitle region
            roi = frame[subtitle_start_y:, :]
            results = reader.readtext(roi)

            # Create mask
            mask = np.zeros((height, width), dtype=np.uint8)

            for bbox, text, conf in results:
                if conf > 0.3:
                    pts = np.array(bbox, dtype=np.int32)
                    pts[:, 1] += subtitle_start_y

                    x_min = max(0, int(np.min(pts[:, 0])) - 10)
                    x_max = min(width, int(np.max(pts[:, 0])) + 10)
                    y_min = max(0, int(np.min(pts[:, 1])) - 10)
                    y_max = min(height, int(np.max(pts[:, 1])) + 10)

                    mask[y_min:y_max, x_min:x_max] = 255

            # Inpaint if text detected
            if np.sum(mask) > 0:
                frame = cv2.inpaint(frame, mask, inpaint_radius, inpaint_method)

            writer.write(frame)
            frame_count += 1

            # Update progress
            progress = int((frame_count / total_frames) * 100)
            tasks[task_id]['progress'] = progress
            tasks[task_id]['message'] = f'Processing frame {frame_count}/{total_frames}'

        cap.release()
        writer.release()

        # Clean up input file
        os.unlink(video_path)

        tasks[task_id]['status'] = 'complete'
        tasks[task_id]['progress'] = 100
        tasks[task_id]['message'] = 'Complete!'
        tasks[task_id]['output_path'] = output_path

    except Exception as e:
        tasks[task_id]['status'] = 'error'
        tasks[task_id]['message'] = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400

    video = request.files['video']
    language = request.form.get('language', 'en')
    quality = request.form.get('quality', 'fast')
    region = float(request.form.get('region', 0.35))

    # Save uploaded video
    task_id = str(uuid.uuid4())
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, 'input.mp4')
    output_path = os.path.join(temp_dir, 'output.mp4')

    video.save(input_path)

    # Initialize task
    tasks[task_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Queued...',
        'output_path': None
    }

    # Start processing in background
    thread = threading.Thread(
        target=process_video_task,
        args=(task_id, input_path, output_path, language, quality, region)
    )
    thread.start()

    return jsonify({'task_id': task_id})


@app.route('/progress/<task_id>')
def get_progress(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    return jsonify({
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message']
    })


@app.route('/download/<task_id>')
def download_video(task_id):
    if task_id not in tasks:
        return jsonify({'error': 'Task not found'}), 404

    task = tasks[task_id]
    if task['status'] != 'complete' or not task['output_path']:
        return jsonify({'error': 'Video not ready'}), 400

    return send_file(
        task['output_path'],
        mimetype='video/mp4',
        as_attachment=True,
        download_name='video_no_subtitles.mp4'
    )


if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Video Subtitle Remover")
    print("="*50)
    print("\n  Open your browser and go to:")
    print("  http://localhost:5000")
    print("\n" + "="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
