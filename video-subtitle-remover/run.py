#!/usr/bin/env python3
"""
Video Subtitle Remover - Main Entry Point
Run this script to start the web UI or process videos from command line.
"""

import argparse
import sys
from pathlib import Path


def run_web_ui(host: str = "0.0.0.0", port: int = 7860, share: bool = False):
    """Launch the Gradio web interface."""
    from web.app import app, initialize_components

    print("Starting Video Subtitle Remover Web UI...")
    print(f"Local URL: http://localhost:{port}")

    initialize_components()
    app.launch(
        server_name=host,
        server_port=port,
        share=share
    )


def run_cli(
    input_path: str,
    output_path: str,
    youtube: bool = False,
    language: str = "en",
    method: str = "telea",
    gpu: bool = True
):
    """Run subtitle removal from command line."""
    from app.downloader import VideoDownloader
    from app.processor import VideoProcessor, ProcessingConfig
    from app.inpainter import InpaintMethod

    # Map method names
    method_map = {
        "telea": InpaintMethod.OPENCV_TELEA,
        "ns": InpaintMethod.OPENCV_NS,
        "lama": InpaintMethod.LAMA,
    }

    config = ProcessingConfig(
        languages=[language],
        inpaint_method=method_map.get(method, InpaintMethod.OPENCV_TELEA),
        use_gpu=gpu,
        preserve_audio=True
    )

    processor = VideoProcessor(config)

    # Handle YouTube URL
    actual_input = input_path
    if youtube:
        print(f"Downloading video from YouTube: {input_path}")
        downloader = VideoDownloader(output_dir="temp")
        actual_input, metadata = downloader.download(input_path)
        print(f"Downloaded: {metadata.get('title', 'Unknown')}")

    # Process the video
    print(f"Processing video: {actual_input}")
    print(f"Output will be saved to: {output_path}")

    def progress_callback(progress, message):
        print(f"[{progress:.1f}%] {message}")

    processor.set_progress_callback(progress_callback)

    success = processor.process_video(actual_input, output_path)

    if success:
        print(f"\nSuccess! Output saved to: {output_path}")
    else:
        print("\nProcessing failed.")
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Video Subtitle Remover - Remove hardcoded subtitles from videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch web UI
  python run.py --web

  # Launch web UI with public sharing link
  python run.py --web --share

  # Process a local video file
  python run.py -i input.mp4 -o output.mp4

  # Download and process YouTube video
  python run.py -i "https://youtube.com/watch?v=..." -o output.mp4 --youtube

  # Use LaMa inpainting for better quality
  python run.py -i input.mp4 -o output.mp4 --method lama

  # Process with Spanish subtitle detection
  python run.py -i input.mp4 -o output.mp4 --language es
        """
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web UI (Gradio interface)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for web UI (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for web UI (default: 7860)"
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public sharing link for web UI"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Input video file path or YouTube URL"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output video file path"
    )

    parser.add_argument(
        "--youtube",
        action="store_true",
        help="Input is a YouTube URL"
    )

    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Subtitle language code (default: en)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["telea", "ns", "lama"],
        default="telea",
        help="Inpainting method: telea (fast), ns (quality), lama (best)"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )

    args = parser.parse_args()

    # Determine mode
    if args.web:
        run_web_ui(
            host=args.host,
            port=args.port,
            share=args.share
        )
    elif args.input and args.output:
        run_cli(
            input_path=args.input,
            output_path=args.output,
            youtube=args.youtube,
            language=args.language,
            method=args.method,
            gpu=not args.no_gpu
        )
    else:
        # Default to web UI if no arguments
        print("No arguments provided. Launching web UI...")
        print("Use --help for command line options.\n")
        run_web_ui()


if __name__ == "__main__":
    main()
