"""
Loitering detection and person ID tracking within zones.
"""
import time
from collections import defaultdict
from .utils import point_in_polygon


class PersonRecord:
    """Tracks a person's presence in a zone."""

    def __init__(self, track_id, center, zone_id):
        self.track_id = track_id
        self.center = center
        self.zone_id = zone_id
        self.first_seen = time.time()
        self.last_seen = time.time()

    def update(self, center):
        self.center = center
        self.last_seen = time.time()


class Tracker:
    """Tracks people in zones and detects loitering."""

    def __init__(self, zones_config):
        """
        zones_config: list of dicts with 'id', 'points', 'threshold_seconds'.
        """
        self.zones = zones_config
        self.zone_occupants = defaultdict(dict)  # zone_id -> {track_id: PersonRecord}
        self.loitering_alerts = set()  # (zone_id, track_id) to avoid duplicate alerts

    def get_center(self, bbox):
        """Get center point of bbox (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def find_zone(self, center):
        """Return zone_id if center is inside any zone, else None."""
        for zone in self.zones:
            points = zone.get("points", [])
            if point_in_polygon(center, points):
                return zone.get("id")
        return None

    def update(self, detections, track_ids=None):
        """
        Update tracking with new detections.
        detections: list of (x1,y1,x2,y2,...)
        track_ids: optional list of track IDs, same length as detections.
        Returns list of loitering alerts: [{"zone_id": ..., "track_id": ..., "seconds": ...}, ...]
        """
        now = time.time()
        alerts = []
        seen = set()

        if track_ids is None:
            track_ids = list(range(len(detections)))

        for i, det in enumerate(detections):
            if len(det) < 4:
                continue
            center = self.get_center(det)
            zone_id = self.find_zone(center)
            if zone_id is None:
                continue

            tid = track_ids[i] if i < len(track_ids) else i
            seen.add((zone_id, tid))

            zone_data = self.zone_occupants[zone_id]
            zone_config = next((z for z in self.zones if z.get("id") == zone_id), {})
            threshold = zone_config.get("threshold_seconds", 30)

            if tid in zone_data:
                zone_data[tid].update(center)
            else:
                zone_data[tid] = PersonRecord(tid, center, zone_id)

            elapsed = now - zone_data[tid].first_seen
            if elapsed >= threshold and (zone_id, tid) not in self.loitering_alerts:
                self.loitering_alerts.add((zone_id, tid))
                alerts.append({
                    "zone_id": zone_id,
                    "track_id": tid,
                    "seconds": int(elapsed),
                })

        # Cleanup people who left zones
        for zone_id, occupants in list(self.zone_occupants.items()):
            for tid in list(occupants.keys()):
                if (zone_id, tid) not in seen:
                    del occupants[tid]
                    self.loitering_alerts.discard((zone_id, tid))

        return alerts
