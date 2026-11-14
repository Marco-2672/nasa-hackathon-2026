# numeric helpers shared by engine.py and engine_api.py
# demo thresholds, tuned for the day-47 alex scenario

from __future__ import annotations

# per-metric status thresholds, values are fractional change vs baseline
STATUS_THRESHOLDS = {
    "resting_heart_rate":  {"amber_high": 0.15, "red_high": 0.30},
    "sleep":               {"amber_low": -0.20, "red_low": -0.40},
    "spo2":                {"amber_low": -0.03, "red_low": -0.06},
    "o2_consumption_rest": {"amber_high": 0.20, "red_high": 0.50},
    # no red_low: zero minutes must return amber
    "exercise":            {"amber_low": -0.50},
}

# trend threshold
TREND_THRESHOLD = 0.07


def change_pct(latest: float, baseline: float) -> int:
    # percent change as int, unrounded floats get dropped by number guard
    if baseline == 0:
        return 0
    return round((latest - baseline) / baseline * 100)


def trend_3d(latest: float, prior: float | None) -> str:
    # compare latest to reading three days earlier
    if prior is None or prior == 0:
        return "stable"
    pct = (latest - prior) / abs(prior)
    if pct > TREND_THRESHOLD:
        return "rising"
    if pct < -TREND_THRESHOLD:
        return "falling"
    return "stable"


def status_for(metric: str, latest: float, baseline: float) -> str:
    # green / amber / red for one metric, unknown metric returns green
    if baseline == 0:
        return "green"
    pct = (latest - baseline) / baseline
    t = STATUS_THRESHOLDS.get(metric, {})
    if "red_high" in t and pct >= t["red_high"]:
        return "red"
    if "red_low" in t and pct <= t["red_low"]:
        return "red"
    if "amber_high" in t and pct >= t["amber_high"]:
        return "amber"
    if "amber_low" in t and pct <= t["amber_low"]:
        return "amber"
    return "green"


def round_spoken(metric: str, value: float) -> float:
    # sleep keeps one decimal, everything else rounds to whole units
    if metric in ("sleep", "mission_dose_msv", "dose_rate_msv_per_day", "co2_mmhg"):
        return round(value, 1)
    return round(value)