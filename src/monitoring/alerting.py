"""
Alerting helpers for monitoring events.
"""

import json
import logging
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)

SEVERITY_ORDER = {
    "low": 0,
    "medium": 1,
    "high": 2
}


def should_send_alert(alerts: List[Dict], min_severity: str) -> bool:
    if not alerts:
        return False

    min_level = SEVERITY_ORDER.get(min_severity.lower(), 1)
    for alert in alerts:
        severity = alert.get("severity", "low").lower()
        if SEVERITY_ORDER.get(severity, 0) >= min_level:
            return True

    return False


def send_webhook(url: str, payload: Dict, timeout_seconds: int = 5) -> bool:
    try:
        response = requests.post(
            url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout_seconds
        )
        response.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Webhook send failed: %s", e)
        return False