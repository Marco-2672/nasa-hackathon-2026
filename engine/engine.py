# derives alerts from readings and writes them to alerts table
# run: python -m engine.engine
# reads demo clock, scans each crew against metrics thresholds, writes alerts
# idempotent for the current day: re-running replaces that day's alerts

from __future__ import annotations

import json
import sys
from pathlib import Path

from .metrics import change_pct, status_for
from .schema import DEFAULT_DB, connect, init_db

LOOKBACK_DAYS = 3


def _daily_reading(con, crew_id: str, metric: str, day: int):
    # max-minutes reading that day, sleep returns 07:00, exercise returns 23:00
    row = con.execute(
        "SELECT value FROM readings WHERE crew_id=? AND metric=? AND mission_day=? "
        "ORDER BY minutes DESC LIMIT 1",
        (crew_id, metric, day),
    ).fetchone()
    return None if row is None else row["value"]


def _clock(con):
    row = con.execute("SELECT value FROM meta WHERE key='clock_minutes'").fetchone()
    return None if row is None else int(row[0])


def _baseline(con, crew_id: str, metric: str):
    row = con.execute(
        "SELECT value FROM baselines WHERE crew_id=? AND metric=?",
        (crew_id, metric),
    ).fetchone()
    return None if row is None else row["value"]


def _alerts_for_crew(con, crew_id: str, today: int) -> list[dict]:
    # generate every alert this crew qualifies for on `today`
    out: list[dict] = []

    # sleep_short: 3 consecutive amber-low nights
    sleep_now = _daily_reading(con, crew_id, "sleep", today)
    sleep_base = _baseline(con, crew_id, "sleep")
    if sleep_now is not None and sleep_base:
        if status_for("sleep", sleep_now, sleep_base) == "amber":
            since = today
            for d in range(today - 1, today - LOOKBACK_DAYS, -1):
                v = _daily_reading(con, crew_id, "sleep", d)
                if v is not None and status_for("sleep", v, sleep_base) == "amber":
                    since = d
                else:
                    break
            out.append({
                "level": "amber", "code": "SLEEP_SHORT",
                "message": f"Sleep {sleep_now:.1f} h, {abs(change_pct(sleep_now, sleep_base))}% below baseline",
                "since_mission_day": since,
                "protocol_ids": ["fatigue_recovery"],
            })

    # exercise_missed: yesterday below 50% of baseline
    ex_now = _daily_reading(con, crew_id, "exercise", today - 1)
    ex_base = _baseline(con, crew_id, "exercise")
    if ex_now is not None and ex_base and ex_now < ex_base * 0.5:
        since = today - 1
        for d in range(today - 2, today - LOOKBACK_DAYS, -1):
            v = _daily_reading(con, crew_id, "exercise", d)
            if v is not None and v < ex_base * 0.5:
                since = d
            else:
                break
        out.append({
            "level": "amber", "code": "EXERCISE_MISSED",
            "message": f"Exercise {int(ex_now)} of {int(ex_base)} minutes/day baseline",
            "since_mission_day": since,
            "protocol_ids": ["fatigue_recovery"],
        })

    # rhr_rising: today >= 15% above baseline, count consecutive above-baseline days
    rhr_now = _daily_reading(con, crew_id, "resting_heart_rate", today)
    rhr_base = _baseline(con, crew_id, "resting_heart_rate")
    if rhr_now is not None and rhr_base:
        pct = (rhr_now - rhr_base) / rhr_base
        if pct >= 0.15:
            n_days = 1
            for d in range(today - 1, today - LOOKBACK_DAYS, -1):
                v = _daily_reading(con, crew_id, "resting_heart_rate", d)
                if v is not None and v > rhr_base:
                    n_days += 1
                else:
                    break
            out.append({
                "level": "amber", "code": "RHR_RISING",
                "message": f"Resting heart rate {change_pct(rhr_now, rhr_base)}% above baseline for {n_days} days",
                "since_mission_day": today - n_days + 1,
                "protocol_ids": ["fatigue_recovery"],
            })

    return out


def derive_alerts(db_path: Path = DEFAULT_DB) -> int:
    # clear today's alerts and regenerate; returns number written
    con = connect(db_path)
    try:
        init_db(con)
        clock = _clock(con)
        if clock is None:
            print("error: no clock_minutes in meta; run python -m engine.sim first")
            return 0
        today = clock // 1440
        # keep emergencies: they come from the crew, not from readings
        con.execute("DELETE FROM alerts WHERE created_day = ? AND code != 'EMERGENCY'", (today,))
        active_events = [p for p in (json.loads(r["payload_json"])
                                     for r in con.execute("SELECT payload_json FROM scheduled_events"))
                         if p["active_from_minutes"] <= clock < p["active_to_minutes"]]
        idx = 0
        for crew_id in ("alex", "sam"):
            event_alerts = [{"level": p["level"], "code": p["code"], "message": p["message"],
                             "since_mission_day": today, "protocol_ids": p["protocol_ids"]} for p in active_events]
            for a in _alerts_for_crew(con, crew_id, today) + event_alerts:
                idx += 1
                con.execute(
                    "INSERT INTO alerts (alert_id, crew_id, level, code, message, "
                    "since_mission_day, created_day, protocol_ids_json, resolved_day) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"A{today}-{idx:02d}", crew_id, a["level"], a["code"], a["message"],
                     a["since_mission_day"], today, json.dumps(a["protocol_ids"]), None),
                )
        con.commit()
        return idx
    finally:
        con.close()


def main() -> int:
    n = derive_alerts()
    print(f"wrote {n} alerts to {DEFAULT_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())