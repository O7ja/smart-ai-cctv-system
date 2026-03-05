"""
Desktop notifications using Plyer.
"""
import os

try:
    from plyer import notification
    PLYER_AVAILABLE = True
except ImportError:
    PLYER_AVAILABLE = False


class AlertManager:
    """Sends desktop notifications for threats."""

    def __init__(self, app_name="Smart CCTV"):
        self.app_name = app_name
        self.enabled = PLYER_AVAILABLE

    def notify_weapon(self, message="Weapon detected!"):
        """Notify about weapon detection."""
        if self.enabled:
            try:
                notification.notify(
                    title=f"{self.app_name} - Weapon Alert",
                    message=message,
                    app_name=self.app_name,
                    timeout=10,
                )
            except Exception:
                pass

    def notify_loitering(self, zone_name, seconds):
        """Notify about loitering detection."""
        if self.enabled:
            try:
                notification.notify(
                    title=f"{self.app_name} - Loitering Alert",
                    message=f"Person loitering in {zone_name} for {seconds}s",
                    app_name=self.app_name,
                    timeout=10,
                )
            except Exception:
                pass

    def notify_fight(self, message="Fight/aggressive behavior detected!"):
        """Notify about fight detection."""
        if self.enabled:
            try:
                notification.notify(
                    title=f"{self.app_name} - Fight Alert",
                    message=message,
                    app_name=self.app_name,
                    timeout=10,
                )
            except Exception:
                pass

    def notify_fire(self, message="Fire or smoke detected!"):
        """Notify about fire/smoke detection."""
        if self.enabled:
            try:
                notification.notify(
                    title=f"{self.app_name} - Fire Alert",
                    message=message,
                    app_name=self.app_name,
                    timeout=10,
                )
            except Exception:
                pass
