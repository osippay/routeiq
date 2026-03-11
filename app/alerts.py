"""
RouteIQ — Universal alert dispatcher.
Supports: telegram, webhook, email, slack, none.
Never raises — all errors are logged silently.
"""
from __future__ import annotations

import logging
import os
import smtplib
import time
from email.mime.text import MIMEText

import requests

logger = logging.getLogger(__name__)


class AlertManager:
    """Dispatches alerts to the configured channel."""

    def __init__(self) -> None:
        self._channel = os.getenv("ALERT_CHANNEL", "none").lower().strip()

    def send(self, message: str) -> None:
        """Send alert to configured channel. Never raises."""
        try:
            handler = {
                "telegram": self._send_telegram,
                "webhook": self._send_webhook,
                "email": self._send_email,
                "slack": self._send_slack,
            }.get(self._channel)

            if handler:
                handler(message)
            else:
                logger.info("[RouteIQ Alert] %s", message)
        except Exception as e:
            logger.error("Alert dispatch failed (channel=%s): %s", self._channel, e)

    def _send_telegram(self, message: str) -> None:
        token = os.getenv("ALERT_TELEGRAM_TOKEN", "")
        chat_id = os.getenv("ALERT_TELEGRAM_CHAT_ID", "")
        if not token or not chat_id:
            logger.warning("Telegram alert: missing TOKEN or CHAT_ID")
            return
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=5,
        )
        if r.ok:
            logger.info("Telegram alert sent")
        else:
            logger.warning("Telegram alert failed: %s", r.status_code)

    def _send_webhook(self, message: str) -> None:
        url = os.getenv("ALERT_WEBHOOK_URL", "")
        if not url:
            logger.warning("Webhook alert: missing ALERT_WEBHOOK_URL")
            return
        r = requests.post(url, json={"text": message, "ts": time.time()}, timeout=5)
        if r.ok:
            logger.info("Webhook alert sent")
        else:
            logger.warning("Webhook alert failed: %s", r.status_code)

    def _send_email(self, message: str) -> None:
        host = os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com")
        port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        user = os.getenv("ALERT_SMTP_USER", "")
        password = os.getenv("ALERT_SMTP_PASS", "")
        to_addr = os.getenv("ALERT_EMAIL", "")
        if not all([user, password, to_addr]):
            logger.warning("Email alert: missing SMTP env vars")
            return
        msg = MIMEText(message)
        msg["Subject"] = "RouteIQ Alert"
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP(host, port, timeout=5) as smtp:
            smtp.starttls()
            smtp.login(user, password)
            smtp.send_message(msg)
        logger.info("Email alert sent to %s", to_addr)

    def _send_slack(self, message: str) -> None:
        url = os.getenv("ALERT_SLACK_WEBHOOK_URL", "")
        if not url:
            logger.warning("Slack alert: missing ALERT_SLACK_WEBHOOK_URL")
            return
        r = requests.post(url, json={"text": message}, timeout=5)
        if r.ok:
            logger.info("Slack alert sent")
        else:
            logger.warning("Slack alert failed: %s", r.status_code)


def send_alert(message: str) -> None:
    """Module-level shortcut."""
    AlertManager().send(message)
