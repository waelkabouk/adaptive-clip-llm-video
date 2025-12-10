#!/usr/bin/env python
"""Download and prepare MSR-VTT dataset for evaluation."""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset URLs (note: these may need to be updated)
DATASET_URLS = {
    "msrvtt_qa": {
        "annotations": "https://github.com/xudejing/video-question-answering/raw/master/data/",
        "description": "MSR-VTT QA annotations (requires manual video download)",
    },
}


def download_file(url: str, output_path: str, description: str = "") -> bool:
    """Download a file with progress."""
    logger.info(f"Downloading {description or url}...")
    
    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rProgress: {percent}%")
            sys.stdout.flush()
            
        urlretrieve(url, output_path, progress_hook)
        print()  # New line after progress
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def prepare_msrvtt_qa(output_dir: Path):
    """Prepare MSR-VTT QA dataset structure."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample QA file for testing
    sample_qa = [
        {
            "video_id": "video0",
            "question": "What is happening in the video?",
            "answer": "A person is walking"
        },
        {
            "video_id": "video1",
            "question": "What color is the object?",
            "answer": "red"
        },
    ]
    
    import json
    qa_file = output_dir / "test_qa.json"
    with open(qa_file, 'w') as f:
        json.dump(sample_qa, f, indent=2)
        
    logger.info(f"Created sample QA file: {qa_file}")
    
    # Create videos directory
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    
    # Instructions
    readme_content = """# MSR-VTT Dataset

## Video Download

MSR-VTT videos need to be downloaded separately due to size and licensing.

### Option 1: Official Source
1. Visit the Microsoft Research website
2. Request access to the MSR-VTT dataset
3. Download and extract videos to this directory

### Option 2: Academic Mirrors
Search for "MSR-VTT video dataset download" for academic mirrors.

### Video Format
- Place video files in the `videos/` subdirectory
- Videos should be named as `video{id}.mp4` (e.g., video0.mp4, video1.mp4)

## Annotations

QA annotations should be in JSON format:
```json
[
    {
        "video_id": "video0",
        "question": "What is happening?",
        "answer": "A person is walking"
    }
]
```

Place annotation files in this directory:
- `test_qa.json` - Test split QA pairs
- `train_qa.json` - Training split QA pairs (optional)
"""
    
    readme_file = output_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)
        
    logger.info(f"Created README: {readme_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare evaluation datasets"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="msrvtt_qa",
        choices=list(DATASET_URLS.keys()),
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.dataset
    
    logger.info(f"Preparing {args.dataset} dataset...")
    
    if args.dataset == "msrvtt_qa":
        prepare_msrvtt_qa(output_dir)
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Location: {output_dir}")
    print("\nNext steps:")
    print("1. Download videos manually (see README in dataset directory)")
    print("2. Place videos in the 'videos' subdirectory")
    print("3. Run benchmark: python scripts/run_benchmark.py --data-dir", output_dir)
    print("="*60)


if __name__ == "__main__":
    main()

