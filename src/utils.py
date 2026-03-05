"""
Helper functions: drawing boxes, calculating distances, zone utilities.
"""
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


def draw_boxes(frame, detections, labels=None, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on frame for detections.
    detections: list of (x1, y1, x2, y2, confidence, class_id)
    """
    for det in detections:
        if len(det) >= 5:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4] if len(det) > 4 else 0
            cls_id = int(det[5]) if len(det) > 5 else 0
            label = labels[cls_id] if labels and cls_id < len(labels) else f"Class {cls_id}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label_text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def calculate_distance(p1, p2):
    """Euclidean distance between two points (x, y)."""
    return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def point_in_polygon(point, polygon):
    """
    Check if point (x, y) is inside polygon.
    polygon: list of [x, y] points.
    Uses OpenCV pointPolygonTest.
    """
    pts = np.array(polygon, dtype=np.int32)
    result = cv2.pointPolygonTest(pts, (point[0], point[1]), False)
    return result >= 0


def draw_zone_polygon(frame, polygon, color=(0, 255, 255), alpha=0.3):
    """Draw a semi-transparent zone polygon on the frame."""
    overlay = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, 2)
    return frame


def save_screenshot(frame, folder, prefix=""):
    """Save a screenshot with timestamp. Returns path or None."""
    base = Path(__file__).resolve().parent.parent
    folder_path = base / "screenshots" / folder
    folder_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}{timestamp}.jpg"
    filepath = folder_path / filename
    if cv2.imwrite(str(filepath), frame):
        return str(filepath)
    return None
