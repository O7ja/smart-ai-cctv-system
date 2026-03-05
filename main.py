"""
Smart AI CCTV System - Entry point.
Weapon detection, fight pose detection, and loitering alerts with desktop notifications.
"""
import json
import argparse
from pathlib import Path

import cv2

from src.detector import Detector
from src.tracker import Tracker
from src.alerts import AlertManager
from src.utils import draw_boxes, draw_zone_polygon, save_screenshot


def load_config():
    config_path = Path(__file__).parent / "zones" / "config.json"
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Smart AI CCTV System")
    parser.add_argument("--source", default=0, help="Video source: 0 for webcam, or path to video file")
    parser.add_argument("--weapon-model", default=None, help="Path to weapon detection .pt model")
    parser.add_argument("--fire-model", default=None, help="Path to fire detection .pt model")
    parser.add_argument("--no-weapon", action="store_true", help="Disable weapon detection")
    parser.add_argument("--no-fire", action="store_true", help="Disable fire detection")
    parser.add_argument("--no-fight", action="store_true", help="Disable fight detection")
    parser.add_argument("--no-loiter", action="store_true", help="Disable loitering detection")
    parser.add_argument("--no-hf", action="store_true", help="Disable Hugging Face model downloads")
    args = parser.parse_args()

    config = load_config()
    weapon_cfg = config.get("weapon_detection", {})
    fire_cfg = config.get("fire_detection", {})
    fight_cfg = config.get("fight_detection", {})
    zones = config.get("loitering_zones", [])

    detector = Detector(
        weapon_model_path=args.weapon_model,
        fire_model_path=args.fire_model,
        confidence_threshold=weapon_cfg.get("confidence_threshold", 0.6),
        fire_confidence_threshold=fire_cfg.get("confidence_threshold", 0.5),
        pose_confidence=fight_cfg.get("pose_confidence_threshold", 0.7),
        use_hf_models=not args.no_hf,
    )
    tracker = Tracker(zones)
    alerts = AlertManager()

    weapon_enabled = weapon_cfg.get("enabled", True) and not args.no_weapon
    fire_enabled = fire_cfg.get("enabled", True) and not args.no_fire
    fight_enabled = fight_cfg.get("enabled", True) and not args.no_fight
    loiter_enabled = not args.no_loiter

    source = args.source
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Could not open video source.")
        return

    print("Smart CCTV running. Press 'q' to quit.")
    print("Weapon:", "ON" if weapon_enabled and detector.weapon_model else "OFF")
    print("Fire:", "ON" if fire_enabled and detector.fire_model else "OFF")
    print("Fight:", "ON" if fight_enabled and detector.pose_detector else "OFF")
    print("Loitering:", "ON" if loiter_enabled else "OFF")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw zones
        for zone in zones:
            pts = zone.get("points", [])
            draw_zone_polygon(frame, pts)

        # Weapon detection
        if weapon_enabled and detector.weapon_model:
            weapon_dets = detector.detect_weapons(frame)
            if weapon_dets:
                # alerts.notify_weapon("Weapon detected in frame!")
                # path = save_screenshot(frame, "weapons", "weapon_")
                # if path:
                #     print(f"Screenshot saved: {path}")
                draw_boxes(frame, weapon_dets, color=(0, 0, 255), labels=["Gun", "Weapon"])

        # Fire detection
        if fire_enabled and detector.fire_model:
            fire_dets = detector.detect_fire(frame)
            if fire_dets:
                # alerts.notify_fire("Fire or smoke detected in frame!")
                # path = save_screenshot(frame, "fire", "fire_")
                # if path:
                #     print(f"Screenshot saved: {path}")
                cv2.putText(frame, "FIRE DETECTED", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)
            draw_boxes(frame, fire_dets, color=(0, 165, 255), labels=["Fire", "Smoke"])

        # Fight detection
        if fight_enabled and detector.pose_detector:
            landmarks = detector.detect_pose(frame)
            if detector.is_fight_pose(landmarks, frame.shape):
                # alerts.notify_fight("Aggressive pose detected!")
                # path = save_screenshot(frame, "fights", "fight_")
                # if path:
                #     print(f"Screenshot saved: {path}")
                cv2.putText(frame, "FIGHT DETECTED", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Loitering - use person detections
        if loiter_enabled and zones and detector.weapon_model:
            person_dets = detector.detect_persons(frame)
            track_ids = list(range(len(person_dets)))
            loiter_alerts = tracker.update(person_dets, track_ids)
            for alert in loiter_alerts:
                zone_name = next((z.get("name", alert["zone_id"]) for z in zones if z["id"] == alert["zone_id"]), alert["zone_id"])
                # alerts.notify_loitering(zone_name, alert["seconds"])
                # path = save_screenshot(frame, "loitering", "loiter_")
                # if path:
                #     print(f"Screenshot saved: {path}")

        cv2.imshow("Smart CCTV", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()


if __name__ == "__main__":
    main()
