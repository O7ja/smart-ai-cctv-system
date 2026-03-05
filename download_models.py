"""
Download models from Hugging Face Hub before running the main system.
This is optional - models will auto-download on first run if not present.
"""
from pathlib import Path
import sys

try:
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
except ImportError as e:
    print(f"Error: Missing required package. Please install: pip install -r requirements.txt")
    print(f"Missing: {e}")
    sys.exit(1)


def download_model(repo_id, filename, local_name, model_dir):
    """Download a model from Hugging Face Hub."""
    print(f"\n{'='*60}")
    print(f"Downloading {local_name}...")
    print(f"Repository: {repo_id}")
    print(f"File: {filename}")
    print(f"{'='*60}")
    
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(model_dir),
        )
        print(f"✓ Successfully downloaded to: {model_path}")
        
        # Verify the model can be loaded
        print(f"Verifying model...")
        model = YOLO(model_path)
        print(f"✓ Model verified and ready to use!")
        return model_path
    except Exception as e:
        print(f"✗ Error downloading {local_name}: {e}")
        print(f"  You can still run the system - it will try to download automatically.")
        return None


def main():
    """Download all required models."""
    print("Smart CCTV System - Model Downloader")
    print("=" * 60)
    
    # Create models directory
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models will be saved to: {model_dir}")
    
    models_to_download = [
        {
            "repo_id": "Subh775/Firearm_Detection_Yolov8n",
            "filename": "weights/best.pt",  # Correct path
            "local_name": "Weapon Detection Model",
        },
        {
            "repo_id": "SalahALHaismawi/yolov26-fire-detection",
            "filename": "best.pt",  # Public model, no gating
            "local_name": "Fire Detection Model",
        },
    ]
    
    print(f"\nThis will download {len(models_to_download)} models.")
    print("Note: This may take several minutes depending on your internet speed.")
    
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled. You can run this script again later.")
        return
    
    downloaded = []
    failed = []
    
    for model_info in models_to_download:
        path = download_model(
            model_info["repo_id"],
            model_info["filename"],
            model_info["local_name"],
            model_dir,
        )
        if path:
            downloaded.append(model_info["local_name"])
        else:
            failed.append(model_info["local_name"])
    
    # Summary
    print(f"\n{'='*60}")
    print("Download Summary:")
    print(f"{'='*60}")
    print(f"✓ Successfully downloaded: {len(downloaded)}/{len(models_to_download)}")
    for name in downloaded:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed to download: {len(failed)}/{len(models_to_download)}")
        for name in failed:
            print(f"  - {name}")
        print("\nNote: The system will attempt to download these automatically on first run.")
    
    print(f"\n{'='*60}")
    print("Done! You can now run: python main.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
