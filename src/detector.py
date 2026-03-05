"""
YOLOv8/YOLOv10 weapon detection, fire detection, and MediaPipe fight/pose detection.
Supports Hugging Face Hub model loading.
"""
import cv2
import numpy as np
from pathlib import Path
import warnings

# Optional imports - fail gracefully if not installed
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_hub_download = None


class Detector:
    """
    Handles YOLOv8/YOLOv10 (weapons, fire) and MediaPipe (fight poses).
    Supports loading models from Hugging Face Hub.
    """

    def __init__(
        self,
        weapon_model_path=None,
        fire_model_path=None,
        person_model_path=None,
        confidence_threshold=0.6,
        fire_confidence_threshold=0.5,
        pose_confidence=0.7,
        use_hf_models=True,
    ):
        self.confidence_threshold = confidence_threshold
        self.fire_confidence_threshold = fire_confidence_threshold
        self.pose_confidence = pose_confidence
        self.weapon_model = None
        self.fire_model = None
        self.person_model = None
        self.pose_detector = None

        # Initialize YOLO models
        if YOLO is not None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)

            # Load weapon detection model
            self.weapon_model = self._load_model(
                weapon_model_path,
                model_dir / "weapon_detection.pt",
                hf_repo_id="Subh775/Firearm_Detection_Yolov8n",
                hf_filename="weights/best.pt",  # Correct path: weights/best.pt
                use_hf=use_hf_models,
                model_type="weapon",
            )

            # Load fire detection model (using publicly accessible model)
            self.fire_model = self._load_model(
                fire_model_path,
                model_dir / "fire_detection.pt",
                hf_repo_id="SalahALHaismawi/yolov26-fire-detection",
                hf_filename="best.pt",  # Public model, no gating required
                use_hf=use_hf_models,
                model_type="fire",
            )

            # Load person detection model (fallback to COCO YOLOv8n)
            self.person_model = self._load_model(
                person_model_path,
                model_dir / "person_detection.pt",
                hf_repo_id=None,
                hf_filename=None,
                use_hf=False,
                model_type="person",
            )
            if self.person_model is None:
                try:
                    self.person_model = YOLO("yolov8n.pt")
                    print("Using default YOLOv8n for person detection")
                except Exception as e:
                    print(f"Warning: Could not load person detection model: {e}")

        # Initialize MediaPipe Pose
        if mp is not None:
            try:
                # Try the old Solutions API (mediapipe < 0.10.31)
                if hasattr(mp, 'solutions'):
                    self.mp_pose = mp.solutions.pose
                    self.pose_detector = self.mp_pose.Pose(
                        min_detection_confidence=self.pose_confidence,
                        min_tracking_confidence=self.pose_confidence,
                    )
                else:
                    # MediaPipe 0.10.31+ removed solutions API
                    print("Warning: MediaPipe Solutions API not available.")
                    print("  Fight detection disabled. To enable:")
                    print("  Option 1: Use MediaPipe Tasks API (requires code migration)")
                    print("  Option 2: Install older version: pip install 'mediapipe<0.10.31'")
                    self.pose_detector = None
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe: {e}")
                self.pose_detector = None

    def _load_model(self, user_path, local_path, hf_repo_id=None, hf_filename=None, use_hf=True, model_type="unknown"):
        """Load a YOLO model from various sources."""
        if YOLO is None:
            return None

        # Try user-provided path first
        if user_path and Path(user_path).exists():
            try:
                print(f"Loading {model_type} model from: {user_path}")
                return YOLO(str(user_path))
            except Exception as e:
                print(f"Warning: Failed to load {model_type} model from {user_path}: {e}")

        # Try local models directory
        if local_path.exists():
            try:
                print(f"Loading {model_type} model from: {local_path}")
                return YOLO(str(local_path))
            except Exception as e:
                print(f"Warning: Failed to load {model_type} model from {local_path}: {e}")

        # Try Hugging Face Hub
        if use_hf and HF_AVAILABLE and hf_repo_id and hf_filename:
            try:
                print(f"Downloading {model_type} model from Hugging Face: {hf_repo_id}")
                model_path = hf_hub_download(
                    repo_id=hf_repo_id,
                    filename=hf_filename,
                    cache_dir=str(local_path.parent),
                )
                print(f"Loaded {model_type} model from Hugging Face: {model_path}")
                return YOLO(model_path)
            except Exception as e:
                print(f"Warning: Failed to download {model_type} model from Hugging Face: {e}")
                # Try alternative: download to local path
                try:
                    if not local_path.exists():
                        model_path = hf_hub_download(
                            repo_id=hf_repo_id,
                            filename=hf_filename,
                            local_dir=str(local_path.parent),
                        )
                        return YOLO(model_path)
                except Exception:
                    pass

        return None

    def detect_persons(self, frame):
        """
        Detect persons (COCO class 0) using YOLO. Returns list of (x1,y1,x2,y2,conf,0).
        Uses dedicated person model if available, else falls back to weapon model.
        """
        model = self.person_model or self.weapon_model
        if model is None:
            return []
        try:
            results = model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                cls_id = int(box.cls[0])
                # COCO class 0 is person, but check model names if available
                cls_name = results.names.get(cls_id, "").lower()
                if cls_id == 0 or "person" in cls_name:
                    conf = float(box.conf[0])
                    if conf >= 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((x1, y1, x2, y2, conf, 0))
            return detections
        except Exception as e:
            print(f"Error detecting persons: {e}")
            return []

    def detect_weapons(self, frame):
        """
        Run weapon detection on frame. Returns list of detections.
        Format: [(x1,y1,x2,y2,conf,cls_id), ...]
        """
        if self.weapon_model is None:
            return []
        try:
            results = self.weapon_model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    detections.append((x1, y1, x2, y2, conf, cls_id))
            return detections
        except Exception as e:
            print(f"Error detecting weapons: {e}")
            return []

    def detect_fire(self, frame):
        """
        Run fire/smoke detection on frame. Returns list of detections.
        Format: [(x1,y1,x2,y2,conf,cls_id), ...]
        cls_id: 0=fire, 1=smoke (may vary by model)
        """
        if self.fire_model is None:
            return []
        try:
            results = self.fire_model(frame, verbose=False)[0]
            detections = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= self.fire_confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    detections.append((x1, y1, x2, y2, conf, cls_id))
            return detections
        except Exception as e:
            print(f"Error detecting fire: {e}")
            return []

    def detect_pose(self, frame):
        """
        Run MediaPipe pose detection. Returns pose landmarks if found.
        """
        if self.pose_detector is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb)
        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def is_fight_pose(self, landmarks, frame_shape):
        """
        Simple heuristic for fight-like poses based on arm and body positions.
        Returns True if pose suggests fighting/aggressive motion.
        """
        if landmarks is None or len(landmarks) < 33:
            return False
        h, w = frame_shape[:2]

        # MediaPipe indices: 11=left shoulder, 12=right shoulder, 13=left elbow, 14=right elbow
        # 15=left wrist, 16=right wrist
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Check visibility
        if any(l.visibility < self.pose_confidence for l in [left_wrist, right_wrist, left_elbow, right_elbow]):
            return False

        # Arms extended outward (like punching) - wrists far from shoulders
        lw_x, lw_y = left_wrist.x * w, left_wrist.y * h
        rw_x, rw_y = right_wrist.x * w, right_wrist.y * h
        ls_x, ls_y = left_shoulder.x * w, left_shoulder.y * h
        rs_x, rs_y = right_shoulder.x * w, right_shoulder.y * h

        dist_l = np.sqrt((lw_x - ls_x) ** 2 + (lw_y - ls_y) ** 2)
        dist_r = np.sqrt((rw_x - rs_x) ** 2 + (rw_y - rs_y) ** 2)

        # Extended arms (threshold based on typical arm length relative to frame)
        arm_threshold = 0.15 * min(h, w)
        return dist_l > arm_threshold and dist_r > arm_threshold

    def release(self):
        """Release resources."""
        if self.pose_detector is not None:
            self.pose_detector.close()
        # YOLO models don't need explicit cleanup, but we can set them to None
        self.weapon_model = None
        self.fire_model = None
        self.person_model = None
