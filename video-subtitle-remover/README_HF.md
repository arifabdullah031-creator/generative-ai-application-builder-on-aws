---
title: Video Subtitle Remover
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Video Subtitle Remover

Remove hardcoded/burned-in subtitles from videos using AI-powered text detection and inpainting.

## Features
- Upload any video file
- AI-powered text detection (EasyOCR)
- Automatic subtitle removal via inpainting
- Preview detection before processing

## How to Use
1. Upload a video file
2. Adjust subtitle region if needed (default 35% from bottom)
3. Click "Remove Subtitles"
4. Download the processed video

## Limitations
- Works best with clearly visible subtitles
- Processing time depends on video length
- Complex backgrounds may show minor artifacts
