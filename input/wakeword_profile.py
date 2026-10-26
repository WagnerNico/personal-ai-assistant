"""Read optional, model-specific settings from a personal evaluation."""

import hashlib
import json
import math
from pathlib import Path


def load_wake_profile(model_path):
    """Return a verified personal preset, or None when missing or outdated.

    Profiles are runtime threshold proposals, not training data or acceptance
    evidence. Applying one remains an explicit action in the desktop settings.
    """
    model_path = Path(model_path)
    try:
        profile = json.loads(model_path.with_name("personal_calibration.json").read_text(encoding="utf-8"))
        if profile["version"] != 1 or profile["model_sha256"] != hashlib.sha256(model_path.read_bytes()).hexdigest():
            return None
        threshold = float(profile["threshold"])
        positive_min = float(profile["minimum_positive_confirmed_peak"])
        negative_max = float(profile["maximum_negative_confirmed_peak"])
        negative_seconds = float(profile["negative_audio_seconds"])
        positives = profile["positive_recordings"]
        if (not all(math.isfinite(value) for value in (threshold, positive_min, negative_max, negative_seconds))
                or not 0 <= negative_max < threshold <= positive_min <= 1
                or not isinstance(positives, int) or isinstance(positives, bool) or positives < 1
                or negative_seconds <= 0):
            return None
        return {**profile, "threshold": threshold,
                "minimum_positive_confirmed_peak": positive_min,
                "maximum_negative_confirmed_peak": negative_max,
                "negative_audio_seconds": negative_seconds}
    except (OSError, ValueError, TypeError, KeyError):
        return None
