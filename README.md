# Smart AI CCTV System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

An advanced intelligent video surveillance system powered by AI and deep learning. Detects weapons, identifies fighting behavior, recognizes fire/smoke, and monitors loitering activity in real-time with desktop notifications.

## 🎯 Features

- **🔫 Weapon Detection** - Identifies guns and firearms in video streams using YOLOv8
- **🥊 Fight Detection** - Detects aggressive poses and fighting behavior using MediaPipe pose estimation
- **🔥 Fire & Smoke Detection** - Real-time detection of fire and smoke using specialized models
- **👥 Loitering Detection** - Monitors zone-based person tracking and alerts on prolonged presence
- **🔔 Desktop Notifications** - Instant alerts via system notifications (Plyer)
- **📸 Auto Screenshot** - Captures evidence images when threats are detected
- **🎬 Multi-Source Support** - Works with webcams, video files, or RTSP streams
- **⚙️ Configurable Zones** - Define custom monitoring zones with adjustable thresholds
- **🚀 Real-time Processing** - Optimized for live video feeds with minimal latency

## 📋 Requirements

- Python 3.8 or higher
- Windows/Linux/macOS
- Webcam or video file
- GPU recommended for better performance (CUDA-enabled NVIDIA GPU)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/O7ja/smart-ai-cctv-system.git
cd smart-ai-cctv-system
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n ai-cctv python=3.9
conda activate ai-cctv

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models (Optional)
```bash
python download_models.py
```

This will automatically download pre-trained models from Hugging Face for weapon and fire detection.

## 🚀 Usage

### Basic Usage (Webcam)
```bash
python main.py
```

### Using Video File
```bash
python main.py --source path/to/video.mp4
```

### Custom Model Paths
```bash
python main.py \
  --source 0 \
  --weapon-model models/weapon_detector.pt \
  --fire-model models/fire_detector.pt
```

### Disable Specific Features
```bash
# Disable weapon detection
python main.py --no-weapon

# Disable multiple features
python main.py --no-weapon --no-fire --no-fight

# Disable Hugging Face model downloads
python main.py --no-hf
```

## ⚙️ Configuration

Configure detection parameters in `zones/config.json`:

```json
{
  "loitering_zones": [
    {
      "id": "zone_1",
      "name": "Main Entrance",
      "points": [[100, 100], [400, 100], [400, 350], [100, 350]],
      "threshold_seconds": 30
    }
  ],
  "weapon_detection": {
    "enabled": true,
    "confidence_threshold": 0.6
  },
  "fire_detection": {
    "enabled": true,
    "confidence_threshold": 0.5
  },
  "fight_detection": {
    "enabled": true,
    "pose_confidence_threshold": 0.7
  }
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `loitering_zones` | Define custom monitoring zones with polygon points | See config.json |
| `threshold_seconds` | Time in seconds before loitering alert triggers | 30 |
| `confidence_threshold` | Detection confidence level (0-1) | 0.6 |
| `pose_confidence_threshold` | Pose detection confidence (0-1) | 0.7 |

## 📁 Project Structure

```
smart-ai-cctv-system/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── download_models.py      # Model downloader script
├── zones/
│   └── config.json        # Configuration file
├── models/                # Pre-trained models directory
├── screenshots/           # Captured evidence images
│   ├── weapons/
│   ├── fire/
│   └── loitering/
└── src/
    ├── __init__.py
    ├── detector.py        # Detection models (weapon, fire, pose)
    ├── tracker.py         # Person tracking and loitering detection
    ├── alerts.py          # Desktop notification system
    └── utils.py           # Helper functions
```

## 🎓 Model Details

### Weapon Detection
- **Model**: YOLOv8 (Firearms Detection)
- **Source**: Hugging Face (Subh775/Firearm_Detection_Yolov8n)
- **Input**: Video frames
- **Output**: Bounding boxes for detected weapons

### Fire Detection
- **Model**: YOLOv8 (Fire Detection)
- **Source**: Hugging Face (SalahALHaismawi/yolov26-fire-detection)
- **Input**: Video frames
- **Output**: Bounding boxes for fire/smoke regions

### Fight Detection
- **Model**: MediaPipe Pose
- **Detection**: Aggressive pose patterns
- **Input**: Video frames
- **Output**: Pose landmarks and fight classification

### Person Tracking
- **Model**: YOLOv8 (Person detection)
- **Tracking**: Simple ID assignment
- **Zone Monitoring**: Polygon-based zone detection

## 📊 Output & Logs

- **Console Output**: Real-time detection logs and alerts
- **Screenshots**: Saved to `screenshots/` folder with timestamps
- **Desktop Notifications**: System popups for critical alerts

## 🎮 Controls

- **Q Key**: Quit the application
- **ESC Key**: Exit (on some systems)

## 🔍 Troubleshooting

### No Detections
1. Ensure video source is accessible
2. Adjust `confidence_threshold` in config.json (lower = more detections)
3. Verify model files are downloaded: `python download_models.py`

### Performance Issues
1. Use GPU acceleration if available
2. Lower video resolution
3. Disable unused detection features (`--no-weapon`, `--no-fire`, etc.)
4. Comment out screenshot saving in main.py

### Model Download Errors
1. Check internet connection
2. Use `--no-hf` flag to skip automatic downloads
3. Manually download models from Hugging Face

## 📝 Notes

- **MediaPipe Version**: Requires `mediapipe<0.10.31` for Solutions API compatibility
- **Performance**: GPU recommended for real-time processing (60+ FPS)
- **Accuracy**: Detection accuracy depends on video quality and lighting conditions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💼 Author

**O7ja** - [GitHub Profile](https://github.com/O7ja)

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- MediaPipe by Google
- OpenCV for computer vision
- Hugging Face for model hosting

## 📞 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**