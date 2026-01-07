# Video Subtitle Remover

A powerful tool to remove hardcoded/burned-in subtitles from videos using AI-powered text detection and inpainting.

## Features

- **YouTube Support**: Download and process videos directly from YouTube
- **Multiple Languages**: Detect subtitles in 12+ languages
- **AI-Powered Detection**: Uses EasyOCR for accurate text detection
- **Multiple Inpainting Methods**:
  - Fast (OpenCV Telea) - Quick processing
  - Quality (OpenCV NS) - Better quality
  - Best (LaMa AI) - State-of-the-art deep learning inpainting
- **Web UI**: Easy-to-use Gradio interface
- **CLI Support**: Command-line interface for automation
- **Audio Preservation**: Keeps original audio track

## Installation

### Prerequisites

- Python 3.9 or higher
- FFmpeg (for audio processing)

### Install FFmpeg

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH.

### Install Python Dependencies

```bash
cd video-subtitle-remover
pip install -r requirements.txt
```

### GPU Support (Recommended)

For faster processing with GPU:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Web UI (Recommended)

Launch the web interface:
```bash
python run.py --web
```

Then open http://localhost:7860 in your browser.

**Options:**
```bash
python run.py --web --port 8080        # Custom port
python run.py --web --share            # Create public link
```

### Command Line

**Process a local video:**
```bash
python run.py -i input.mp4 -o output.mp4
```

**Download and process YouTube video:**
```bash
python run.py -i "https://youtube.com/watch?v=VIDEO_ID" -o output.mp4 --youtube
```

**Use high-quality inpainting:**
```bash
python run.py -i input.mp4 -o output.mp4 --method lama
```

**Process Spanish subtitles:**
```bash
python run.py -i input.mp4 -o output.mp4 --language es
```

**Full options:**
```bash
python run.py --help
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--language` | Subtitle language code (en, es, fr, etc.) | en |
| `--method` | Inpainting method (telea, ns, lama) | telea |
| `--no-gpu` | Disable GPU acceleration | False |

## Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Chinese (ch)
- Japanese (ja)
- Korean (ko)
- Arabic (ar)
- Hindi (hi)

## How It Works

1. **Frame Extraction**: Video is split into individual frames
2. **Text Detection**: EasyOCR identifies subtitle regions in the bottom portion of each frame
3. **Mask Generation**: Binary masks are created for detected text areas
4. **Inpainting**: Masked regions are filled using the selected algorithm
5. **Video Reconstruction**: Processed frames are combined back into a video
6. **Audio Merge**: Original audio track is preserved and merged with the clean video

## Inpainting Methods

### Fast (OpenCV Telea)
- Uses Fast Marching Method
- Best for: Quick processing, small text
- Speed: Fast

### Quality (OpenCV NS)
- Uses Navier-Stokes based method
- Best for: Better quality with reasonable speed
- Speed: Medium

### Best (LaMa AI)
- Uses Large Mask Inpainting neural network
- Best for: Complex backgrounds, large text areas
- Speed: Slow (requires GPU for reasonable speed)

## Project Structure

```
video-subtitle-remover/
├── app/
│   ├── __init__.py
│   ├── downloader.py    # YouTube video download
│   ├── detector.py      # Subtitle text detection
│   ├── inpainter.py     # Video inpainting
│   └── processor.py     # Main processing pipeline
├── web/
│   ├── __init__.py
│   └── app.py           # Gradio web interface
├── temp/                # Temporary files
├── output/              # Output videos
├── requirements.txt
├── run.py               # Main entry point
└── README.md
```

## Limitations

- **Hardcoded subtitles only**: Cannot remove soft subtitles (use a video player for those)
- **Complex backgrounds**: May show artifacts in areas with complex textures
- **Stylized fonts**: Decorative or unusual fonts may not be detected well
- **Performance**: Processing is frame-by-frame, so longer videos take more time

## Troubleshooting

### "FFmpeg not found"
Install FFmpeg and ensure it's in your system PATH.

### "CUDA out of memory"
Use `--no-gpu` flag or reduce video resolution before processing.

### "Subtitles not detected"
- Try adjusting the subtitle region slider in web UI
- Ensure correct language is selected
- Some stylized fonts may not be recognized

### Slow processing
- Use "Fast (OpenCV Telea)" method
- Enable GPU if available
- Process shorter video clips

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Text detection
- [LaMa](https://github.com/saic-mdal/lama) - Inpainting model
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloading
- [Gradio](https://gradio.app/) - Web interface
