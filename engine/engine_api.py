# contract surface, called from voice/
# every function returns json-serialisable output, never raises to caller
# on error returns {"error": "<short reason>"}
# demo clock lives in meta.clock_minutes; set_clock() updates it

from __future__ import annotations

import json
import random
import statistics
import time as _time
from pathlib import Path

from .engine import current_co2, derive_alerts, injected_value
from .metrics import CO2_LIMIT_MMHG, CO2_SMAC_MMHG, change_pct, round_spoken, status_for, trend_3d
from .schema import DEFAULT_DB, connect, init_db

# order and units match tool_examples.get_vitals
METRIC_ORDER = [
    "resting_heart_rate", "sleep", "spo2", "o2_consumption_rest", "exercise",
]
METRIC_UNITS = {
    "resting_heart_rate": "bpm",
    "sleep": "hours",
    "spo2": "%",
    "o2_consumption_rest": "mL/min",
    "exercise": "minutes/day",
}

PROTOCOLS_PATH = Path(__file__).resolve().parent / "protocols.json"

# 3-minute pvt lapse threshold; classic 10-min pvt uses 500ms, this is tighter
PVT_LAPSE_MS = 400


def _clock(con) -> dict | None:
    # return mission_day, minute_of_day, minutes, time, or None if no clock
    row = con.execute("SELECT value FROM meta WHERE key='clock_minutes'").fetchone()
    if row is None:
        return None
    minutes = int(row[0])
    return {
        "minutes": minutes,
        "mission_day": minutes // 1440,
        "minute_of_day": minutes % 1440,
        "time": f"{(minutes % 1440) // 60:02d}:{minutes % 60:02d}",
    }


def _latest(con, crew_id: str, metric: str, up_to_minutes: int):
    # most recent reading at or before up_to_minutes
    return con.execute(
        "SELECT value, minutes FROM readings WHERE crew_id=? AND metric=? AND minutes<=? "
        "ORDER BY minutes DESC LIMIT 1",
        (crew_id, metric, up_to_minutes),
    ).fetchone()


def _three_days_earlier(con, crew_id: str, metric: str, up_to_minutes: int):
    # most recent reading at or before (up_to_minutes - 3*1440)
    return con.execute(
        "SELECT value FROM readings WHERE crew_id=? AND metric=? AND minutes<=? "
        "ORDER BY minutes DESC LIMIT 1",
        (crew_id, metric, up_to_minutes - 3 * 1440),
    ).fetchone()


def _baseline(con, crew_id: str, metric: str):
    return con.execute(
        "SELECT value, unit FROM baselines WHERE crew_id=? AND metric=?",
        (crew_id, metric),
    ).fetchone()


def _err(e: Exception) -> dict:
    return {"error": str(e)[:60]}


def _load_protocols() -> list[dict]:
    return json.loads(PROTOCOLS_PATH.read_text())


def get_vitals(crew_id: str, hours: int = 24) -> dict:
    # latest reading per metric with baseline, change_pct, trend, status
    # hours is accepted for contract compatibility, sim is dense enough that
    # the most recent reading is always within hours
    try:
        con = connect(DEFAULT_DB)
        clock = _clock(con)
        if clock is None:
            return {"error": "clock not set"}
        metrics = []
        for m in METRIC_ORDER:
            latest_row = _latest(con, crew_id, m, clock["minutes"])
            base_row = _baseline(con, crew_id, m)
            if latest_row is None:
                return {"error": f"no {m} reading for {crew_id}"}
            if base_row is None:
                return {"error": f"no baseline for {crew_id}/{m}"}
            latest = latest_row["value"]
            over = injected_value(con, crew_id, m, clock["minutes"])
            if over is not None:
                latest = over
            baseline = base_row["value"]
            prior_row = _three_days_earlier(con, crew_id, m, clock["minutes"])
            prior = None if prior_row is None else prior_row["value"]
            metrics.append({
                "metric": m,
                "unit": base_row["unit"],
                "latest": round_spoken(m, latest),
                "baseline": round_spoken(m, baseline),
                "change": round_spoken(m, latest - baseline),  # the model never has to subtract
                "change_pct": change_pct(latest, baseline),
                "trend_3d": trend_3d(latest, prior),
                "status": status_for(m, latest, baseline),
            })
        return {
            "crew_id": crew_id,
            "mission_day": clock["mission_day"],
            "time": clock["time"],
            "metrics": metrics,
        }
    except Exception as e:
        return _err(e)


def get_environment() -> dict:
    # radiation + cabin at current clock, plus currently-active events
    try:
        con = connect(DEFAULT_DB)
        clock = _clock(con)
        if clock is None:
            return {"error": "clock not set"}
        env = con.execute(
            "SELECT o2_pct, co2_mmhg, temp_c, dose_rate_msv_per_day, mission_dose_msv "
            "FROM environment WHERE mission_day=? AND minutes<=? ORDER BY minutes DESC LIMIT 1",
            (clock["mission_day"], clock["minutes"]),
        ).fetchone()
        if env is None:
            return {"error": "no environment sample at clock"}

        rad_status = "green"
        events = []
        for row in con.execute("SELECT name, payload_json FROM scheduled_events"):
            p = json.loads(row["payload_json"])
            if p["active_from_minutes"] <= clock["minutes"] < p["active_to_minutes"]:
                rad_status = "red"
                events.append({
                    "name": row["name"],
                    "level": p["level"],
                    "code": p["code"],
                    "message": p["message"],
                    "protocol_ids": p["protocol_ids"],
                })

        co2 = current_co2(con, clock["minutes"])
        cabin_status = "red" if co2 >= CO2_SMAC_MMHG else "amber" if co2 > CO2_LIMIT_MMHG else "green"
        return {
            "mission_day": clock["mission_day"],
            "time": clock["time"],
            "radiation": {
                "dose_rate_msv_per_day": round(env["dose_rate_msv_per_day"], 1),
                "mission_dose_msv": round(env["mission_dose_msv"], 1),
                "status": rad_status,
            },
            "cabin": {
                "o2_pct": int(env["o2_pct"]),
                "co2_mmhg": round(co2, 1),
                "temp_c": int(env["temp_c"]),
                "status": cabin_status,
            },
            "events": events,
        }
    except Exception as e:
        return _err(e)


def get_alerts(crew_id: str | None = None, active_only: bool = True) -> dict:
    # active alerts, active_only filters on resolved_day is null
    # nothing resolves in the demo so active_only is a no-op
    try:
        con = connect(DEFAULT_DB)
        sql = "SELECT * FROM alerts"
        params: list = []
        where: list[str] = []
        clock = _clock(con)
        if clock is not None:  # only today's alerts, so a clock jump never shows stale ones
            where.append("created_day = ?")
            params.append(clock["mission_day"])
        if crew_id is not None:
            where.append("crew_id = ?")
            params.append(crew_id)
        if active_only:
            where.append("resolved_day IS NULL")
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY alert_id"
        rows = con.execute(sql, params).fetchall()
        return {"alerts": [
            {
                "alert_id": r["alert_id"],
                "crew_id": r["crew_id"],
                "level": r["level"],
                "code": r["code"],
                "message": r["message"],
                "since_mission_day": r["since_mission_day"],
                "protocol_ids": json.loads(r["protocol_ids_json"]),
            }
            for r in rows
        ]}
    except Exception as e:
        return _err(e)


def log_symptom(crew_id: str, description: str, severity_0_10: int, transcript: str) -> dict:
    # append a symptom report, returns assigned log_id and clock time
    try:
        con = connect(DEFAULT_DB)
        clock = _clock(con)
        if clock is None:
            return {"error": "clock not set"}
        # next log id: highest existing number plus one
        row = con.execute(
            "SELECT log_id FROM symptoms ORDER BY log_id DESC LIMIT 1"
        ).fetchone()
        n = 1 if row is None else int(row["log_id"].split("-")[1]) + 1
        log_id = f"L-{n:04d}"
        con.execute(
            "INSERT INTO symptoms (log_id, crew_id, mission_day, time, minutes, description, severity, transcript) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (log_id, crew_id, clock["mission_day"], clock["time"], clock["minutes"],
             description, int(severity_0_10), transcript),
        )
        con.commit()
        return {"ok": True, "log_id": log_id,
                "mission_day": clock["mission_day"], "time": clock["time"]}
    except Exception as e:
        return _err(e)


CHECKIN_QUESTIONS = [
    {"id": "sleep_quality", "text": "How well did you sleep, from 1, very badly, to 5, very well?", "min": 1, "max": 5},
    {"id": "sleepiness_kss", "text": "How sleepy do you feel right now, from 1, extremely alert, to 9, very sleepy and fighting to stay awake?", "min": 1, "max": 9},
    {"id": "fatigue_sp", "text": "How tired do you feel right now, from 1, fully alert, to 7, completely exhausted?", "min": 1, "max": 7},
    {"id": "mood", "text": "How is your mood, from 1, very low, to 5, very good?", "min": 1, "max": 5},
    {"id": "stress", "text": "How stressed do you feel, from 1, not at all, to 5, extremely?", "min": 1, "max": 5},
]


def start_checkin(crew_id: str) -> dict:
    # create (or reset) a check-in row and return the questions
    try:
        con = connect(DEFAULT_DB)
        clock = _clock(con)
        if clock is None:
            return {"error": "clock not set"}
        checkin_id = f"C{clock['mission_day']}-{crew_id}"
        con.execute(
            "INSERT OR REPLACE INTO checkins (checkin_id, crew_id, mission_day, time, minutes, answers_json, summary, completed) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (checkin_id, crew_id, clock["mission_day"], clock["time"], clock["minutes"],
             "{}", "", 0),
        )
        con.commit()
        return {"checkin_id": checkin_id, "questions": CHECKIN_QUESTIONS}
    except Exception as e:
        return _err(e)


def submit_checkin(checkin_id: str, answers: dict) -> dict:
    # score answers vs baselines, store summary, return flagged results
    try:
        con = connect(DEFAULT_DB)
        row = con.execute("SELECT * FROM checkins WHERE checkin_id=?", (checkin_id,)).fetchone()
        if row is None:
            return {"error": f"unknown checkin {checkin_id}"}
        crew_id = row["crew_id"]
        results = []
        flags = []
        for q in CHECKIN_QUESTIONS:
            qid = q["id"]
            if qid not in answers:
                continue
            v = int(answers[qid])
            base_row = _baseline(con, crew_id, qid)
            if base_row is None:
                continue
            base = base_row["value"]
            # worse means higher for fatigue/sleepiness/stress, lower for mood/sleep_quality
            higher_is_worse = qid in ("fatigue_sp", "sleepiness_kss", "stress")
            worse = (v >= base + 1) if higher_is_worse else (v <= base - 1)
            if worse:
                results.append({
                    "id": qid, "value": v, "baseline": int(base),
                    "flag": "higher than usual" if higher_is_worse else "lower than usual",
                })
                flags.append(_flag_text(qid, v, base, crew_id, con, row["mission_day"]))
        summary = _summary_text(answers, con, crew_id)
        con.execute(
            "UPDATE checkins SET answers_json=?, summary=?, completed=1 WHERE checkin_id=?",
            (json.dumps(answers), summary, checkin_id),
        )
        con.commit()
        return {"ok": True, "results": results, "flags": flags}
    except Exception as e:
        return _err(e)


def _flag_text(qid: str, value: int, baseline: float, crew_id: str, con, today: int) -> str:
    # human-readable flag; "N days running" only when stored check-ins on the days before back it up
    label = {"fatigue_sp": "Fatigue", "sleepiness_kss": "Sleepiness",
             "stress": "Stress", "mood": "Mood", "sleep_quality": "Sleep quality"}[qid]
    higher_is_worse = qid in ("fatigue_sp", "sleepiness_kss", "stress")
    text = f"{label} {'above' if higher_is_worse else 'below'} personal baseline"
    run, expected = 1, today - 1
    for r in con.execute("SELECT mission_day, answers_json FROM checkins WHERE crew_id=? AND completed=1 "
                         "AND mission_day<? ORDER BY mission_day DESC", (crew_id, today)):
        v = json.loads(r["answers_json"]).get(qid)
        if r["mission_day"] != expected or v is None:
            break
        if not ((v >= baseline + 1) if higher_is_worse else (v <= baseline - 1)):
            break
        run, expected = run + 1, expected - 1
    if run >= 2:
        text += f" {run} days running"
    return text


def _summary_text(answers: dict, con, crew_id: str) -> str:
    # compact line for the crew log, mentions fatigue against this person's own baseline
    f = answers.get("fatigue_sp")
    if f is None:
        return "Check-in submitted"
    b = _baseline(con, crew_id, "fatigue_sp")
    return f"Fatigue {f} of 7" + (f", baseline {int(b['value'])}" if b else "")


def get_protocol(protocol_id: str) -> dict:
    # return one protocol card as read out, fixed text, sourced
    try:
        for p in _load_protocols():
            if p["protocol_id"] == protocol_id:
                return {k: p[k] for k in ("protocol_id", "title", "steps", "escalate_if", "source")}
        return {"error": f"unknown protocol_id {protocol_id}"}
    except Exception as e:
        return _err(e)


def get_crew_log(crew_id: str, days: int = 7) -> dict:
    # last n days of symptoms and check-ins, chronological
    try:
        con = connect(DEFAULT_DB)
        clock = _clock(con)
        if clock is None:
            return {"error": "clock not set"}
        since_day = clock["mission_day"] - days

        entries = []
        for s in con.execute(
            "SELECT mission_day, time, minutes, description, severity FROM symptoms "
            "WHERE crew_id=? AND mission_day>=? ORDER BY minutes",
            (crew_id, since_day),
        ):
            entries.append({
                "mission_day": s["mission_day"], "time": s["time"],
                "type": "symptom", "text": s["description"], "severity": int(s["severity"]),
            })
        for c in con.execute(
            "SELECT mission_day, time, minutes, summary FROM checkins "
            "WHERE crew_id=? AND completed=1 AND mission_day>=? ORDER BY minutes",
            (crew_id, since_day),
        ):
            entries.append({
                "mission_day": c["mission_day"], "time": c["time"],
                "type": "checkin", "summary": c["summary"],
            })
        entries.sort(key=lambda e: (e["mission_day"], e["time"]))
        return {"crew_id": crew_id, "entries": entries}
    except Exception as e:
        return _err(e)


def get_crew_profile(crew_id: str) -> dict:
    # short profile to inject into the model's system prompt
    # summary_text is written by code, never by the ai
    try:
        con = connect(DEFAULT_DB)
        crew_row = con.execute("SELECT name, role FROM crew WHERE crew_id=?", (crew_id,)).fetchone()
        if crew_row is None:
            return {"error": f"unknown crew_id {crew_id}"}
        b = lambda m: _baseline(con, crew_id, m)
        rhr = b("resting_heart_rate")
        slp = b("sleep")
        spo2 = b("spo2")
        baselines = {
            "resting_heart_rate_bpm": int(rhr["value"]),
            "sleep_hours": round(slp["value"], 1),
            "spo2_pct": int(spo2["value"]),
        }
        clock = _clock(con)
        today = clock["mission_day"] if clock else 47

        # short-sleep count in last 7 days
        short_nights = 0
        for d in range(today - 6, today + 1):
            v = con.execute(
                "SELECT value FROM readings WHERE crew_id=? AND metric='sleep' AND mission_day=? "
                "ORDER BY minutes DESC LIMIT 1", (crew_id, d),
            ).fetchone()
            if v is not None and v["value"] < slp["value"] * 0.8:
                short_nights += 1

        # rhr alert pattern for today
        rhr_alert = con.execute(
            "SELECT * FROM alerts WHERE crew_id=? AND code='RHR_RISING' "
            "AND created_day=? ORDER BY alert_id LIMIT 1",
            (crew_id, today),
        ).fetchone()

        # most recent symptom in last 7 days
        sym = con.execute(
            "SELECT mission_day, description, severity FROM symptoms "
            "WHERE crew_id=? AND mission_day>=? ORDER BY minutes DESC LIMIT 1",
            (crew_id, today - 6),
        ).fetchone()

        # most recent day with zero exercise
        zero_day = None
        for d in range(today - 1, today - 7, -1):
            v = con.execute(
                "SELECT value FROM readings WHERE crew_id=? AND metric='exercise' AND mission_day=? "
                "ORDER BY minutes DESC LIMIT 1", (crew_id, d),
            ).fetchone()
            if v is not None and v["value"] == 0:
                zero_day = d
                break

        parts = []
        if short_nights:
            parts.append(f"short sleep {short_nights} nights")
        if rhr_alert is not None:
            parts.append(f"resting heart rate rising since day {rhr_alert['since_mission_day']}")
        if sym is not None:
            parts.append(f"{sym['description']} logged day {sym['mission_day']}, severity {sym['severity']}")
        if zero_day is not None:
            parts.append(f"no exercise day {zero_day}")
        last7 = "; ".join(parts) if parts else "no notable events"

        active = con.execute(
            "SELECT level, code, since_mission_day FROM alerts WHERE crew_id=? AND created_day=?",
            (crew_id, today),
        ).fetchall()
        alert_bits = [
            f"{r['level']}, {_alert_description(r['code'], r['since_mission_day'])}"
            for r in active
        ]
        alerts_text = ", ".join(alert_bits) if alert_bits else "none"

        summary = (
            f"{crew_row['name']}, {crew_row['role']}. "
            f"Personal baselines: resting heart rate {baselines['resting_heart_rate_bpm']} bpm, "
            f"sleep {baselines['sleep_hours']} h, SpO2 {baselines['spo2_pct']}%. "
            f"Last 7 days: {last7}. Active alerts: {alerts_text}."
        )
        return {
            "crew_id": crew_id,
            "name": crew_row["name"],
            "role": crew_row["role"],
            "baselines": baselines,
            "notes": [],
            "summary_text": summary,
        }
    except Exception as e:
        return _err(e)


def _alert_description(code: str, since_day: int) -> str:
    return {
        "RHR_RISING": f"resting heart rate rising since day {since_day}",
        "SLEEP_SHORT": f"short sleep since day {since_day}",
        "EXERCISE_MISSED": f"missed exercise since day {since_day}",
        "EMERGENCY": "emergency logged",
        "SPE_SHELTER": "solar particle event, shelter protocol",
        "CO2_HIGH": f"cabin CO2 above NASA's {CO2_LIMIT_MMHG:g} mmHg limit",
        "SPO2_LOW": "oxygen saturation below baseline",
    }.get(code, code.lower().replace("_", " "))


def raise_emergency(crew_id: str, trigger_phrase: str, transcript: str, protocol_id: str = "general") -> dict:
    # write red alert + emergency log, called from phrase check
    try:
        con = connect(DEFAULT_DB)
        clock = _clock(con)
        if clock is None:
            return {"error": "clock not set"}
        # next alarm id
        row = con.execute(
            "SELECT alarm_id FROM emergencies ORDER BY alarm_id DESC LIMIT 1"
        ).fetchone()
        n = 1 if row is None else int(row["alarm_id"].split("-")[1]) + 1
        alarm_id = f"E-{n:03d}"
        con.execute(
            "INSERT INTO emergencies (alarm_id, crew_id, mission_day, time, minutes, trigger_phrase, transcript) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (alarm_id, crew_id, clock["mission_day"], clock["time"], clock["minutes"],
             trigger_phrase, transcript),
        )
        # also a red alert so dashboard and get_alerts see it
        alert_id = f"A{clock['mission_day']}-EM"
        con.execute(
            "INSERT OR REPLACE INTO alerts (alert_id, crew_id, level, code, message, "
            "since_mission_day, created_day, protocol_ids_json, resolved_day) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (alert_id, crew_id, "red", "EMERGENCY",
             f"Emergency: {trigger_phrase}", clock["mission_day"], clock["mission_day"],
             json.dumps([protocol_id]), None),
        )
        con.commit()
        return {"ok": True, "alarm_id": alarm_id}
    except Exception as e:
        return _err(e)


def record_pvt(crew_id: str, reaction_times_ms: list[int]) -> dict:
    # score a 3-minute pvt, 400ms lapse threshold (tighter than classic 500ms)
    try:
        if not reaction_times_ms:
            return {"error": "no reaction times provided"}
        con = connect(DEFAULT_DB)
        row = con.execute("SELECT value FROM meta WHERE key='pvt_baseline_ms'").fetchone()
        baseline = 265 if row is None else int(row[0])
        median_rt = int(statistics.median(reaction_times_ms))
        lapses = sum(1 for rt in reaction_times_ms if rt >= PVT_LAPSE_MS)
        pct = change_pct(median_rt, baseline)
        if pct >= 30:
            status = "red"
        elif pct >= 15:
            status = "amber"
        else:
            status = "green"
        return {
            "median_rt_ms": median_rt,
            "lapses": lapses,
            "baseline_median_ms": baseline,
            "change_pct": pct,
            "status": status,
        }
    except Exception as e:
        return _err(e)


def list_protocols() -> list:
    # put this in the system prompt so the model knows which cards exist
    try:
        return [{"protocol_id": p["protocol_id"], "title": p["title"]}
                for p in _load_protocols()]
    except Exception:
        return []


def set_clock(mission_day: int, time: str) -> dict:
    # demo control: jump the clock, used to script the video
    try:
        hh, mm = time.split(":")
        minutes = int(mission_day) * 1440 + int(hh) * 60 + int(mm)
        con = connect(DEFAULT_DB)
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('clock_minutes', ?)",
                    (str(minutes),))
        con.commit()
        derive_alerts(DEFAULT_DB)  # alerts always match the clock after a jump
        return {"ok": True, "mission_day": int(mission_day), "time": time}
    except Exception as e:
        return _err(e)


def reset_demo() -> dict:
    # demo control: back to the seeded state between takes, without rebuilding the database file
    # (other processes keep working): clears everything written during a take, restores the seeded
    # crew log, resets the clock to day 47 07:30 and recalculates alerts
    try:
        from .sim import seed_logs
        con = connect(DEFAULT_DB)
        init_db(con)
        for table in ("symptoms", "checkins", "emergencies", "alerts", "injected", "conversation"):
            con.execute(f"DELETE FROM {table}")
        con.execute("DELETE FROM meta WHERE key='ptt'")
        # tells the voice loop to forget the conversation too, so a retake starts clean
        con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('demo_reset', ?)", (str(_time.time()),))
        seed_logs(con)
        con.commit()
        con.close()
        return set_clock(47, "07:30")
    except Exception as e:
        return _err(e)


def trigger_event(name: str) -> dict:
    # demo control: force a scheduled event to be active now by jumping the clock
    try:
        con = connect(DEFAULT_DB)
        row = con.execute(
            "SELECT mission_day, time FROM scheduled_events WHERE name=? "
            "ORDER BY minutes LIMIT 1", (name,),
        ).fetchone()
        if row is None:
            return {"error": f"unknown event {name}"}
        return set_clock(row["mission_day"], row["time"])
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- live faults (demo controls)
# Faults the dashboard can inject at the current mission time. They override the simulated readings
# until reset_demo(). Values are chosen to cross the alert thresholds, not taken from a real incident.
FAULTS = {
    "co2_scrubber_fault":   {"label": "CO2 scrubber fault", "crew_id": None, "metric": "co2_mmhg", "value": 4.2},
    "sam_spo2_drop":        {"label": "Sam: blood oxygen drop", "crew_id": "sam", "metric": "spo2", "value": 91},
    "solar_particle_event": {"label": "Solar particle event"},
}


def list_faults() -> list:
    return [{"fault": k, "label": v["label"]} for k, v in FAULTS.items()]


def active_faults() -> list:
    try:
        con = connect(DEFAULT_DB)
        init_db(con)
        kinds = [r["kind"] for r in con.execute("SELECT DISTINCT kind FROM injected")]
        env = get_environment()
        if any(e.get("code") == "SPE_SHELTER" for e in env.get("events", [])):
            kinds.append("solar_particle_event")
        return kinds
    except Exception:
        return []


def inject_fault(kind: str) -> dict:
    # kind "surprise" picks a fault at random (not seeded on purpose: nobody knows what is coming)
    try:
        if kind == "surprise":
            fresh = [k for k in FAULTS if k not in active_faults()] or list(FAULTS)
            kind = random.SystemRandom().choice(fresh)
        f = FAULTS.get(kind)
        if f is None:
            return {"error": f"unknown fault {kind}"}
        if kind == "solar_particle_event":
            res = trigger_event("solar_particle_event")
            if "error" in res:
                return res
        else:
            con = connect(DEFAULT_DB)
            init_db(con)
            clock = _clock(con)
            if clock is None:
                return {"error": "clock not set"}
            con.execute("INSERT INTO injected (kind, crew_id, metric, value, from_minutes) VALUES (?, ?, ?, ?, ?)",
                        (kind, f["crew_id"], f["metric"], f["value"], clock["minutes"]))
            con.commit()
            con.close()
            derive_alerts(DEFAULT_DB)
        return {"ok": True, "fault": kind, "label": f["label"]}
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- chart series (dashboard)
def get_series(crew_id: str, metric: str, hours: int = 48) -> dict:
    # readings up to the mission clock, x = hours before now, live faults applied
    try:
        con = connect(DEFAULT_DB)
        init_db(con)
        clock = _clock(con)["minutes"]
        rows = con.execute("SELECT minutes, value FROM readings WHERE crew_id=? AND metric=? AND minutes BETWEEN ? AND ? "
                           "ORDER BY minutes", (crew_id, metric, clock - hours * 60, clock)).fetchall()
        f = con.execute("SELECT value, from_minutes FROM injected WHERE crew_id=? AND metric=? "
                        "ORDER BY from_minutes DESC LIMIT 1", (crew_id, metric)).fetchone()
        pts = [{"hours": round((r["minutes"] - clock) / 60, 2),
                "value": f["value"] if f and r["minutes"] >= f["from_minutes"] else r["value"]} for r in rows]
        if f and f["from_minutes"] <= clock and (not pts or pts[-1]["hours"] != 0.0):
            pts.append({"hours": 0.0, "value": f["value"]})
        return {"crew_id": crew_id, "metric": metric, "points": pts}
    except Exception as e:
        return _err(e)


def get_co2_series(hours: int = 24) -> dict:
    try:
        con = connect(DEFAULT_DB)
        init_db(con)
        clock = _clock(con)["minutes"]
        rows = con.execute("SELECT minutes, co2_mmhg FROM environment WHERE minutes BETWEEN ? AND ? ORDER BY minutes",
                           (clock - hours * 60, clock)).fetchall()
        f = con.execute("SELECT value, from_minutes FROM injected WHERE crew_id IS NULL AND metric='co2_mmhg' "
                        "ORDER BY from_minutes DESC LIMIT 1").fetchone()
        pts = [{"hours": round((r["minutes"] - clock) / 60, 2),
                "value": f["value"] if f and r["minutes"] >= f["from_minutes"] else r["co2_mmhg"]} for r in rows]
        if f and f["from_minutes"] <= clock and (not pts or pts[-1]["hours"] != 0.0):
            pts.append({"hours": 0.0, "value": f["value"]})
        return {"points": pts, "limit_mmhg": CO2_LIMIT_MMHG}
    except Exception as e:
        return _err(e)


# ---------------------------------------------------------------- comms log + Bento status (voice loop <-> dashboard)
def log_conversation(speaker: str, text: str, crew_id: str | None = None, kind: str = "") -> dict:
    try:
        con = connect(DEFAULT_DB)
        init_db(con)
        clock = _clock(con) or {"mission_day": None, "time": None}
        con.execute("INSERT INTO conversation (at, mission_day, time, speaker, crew_id, text, kind) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (_time.time(), clock["mission_day"], clock["time"], speaker, crew_id, text, kind))
        con.commit()
        return {"ok": True}
    except Exception as e:
        return _err(e)


def get_conversation(limit: int = 14) -> dict:
    try:
        con = connect(DEFAULT_DB)
        init_db(con)
        rows = con.execute("SELECT * FROM conversation ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return {"lines": [dict(r) for r in reversed(rows)]}
    except Exception as e:
        return _err(e)


def _set_meta(key: str, value: str) -> None:
    con = connect(DEFAULT_DB)
    con.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    con.commit()
    con.close()


def _get_meta(key: str):
    row = connect(DEFAULT_DB).execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_status(state: str) -> dict:
    # ready / listening / thinking / speaking, written by the voice loop
    try:
        _set_meta("bento_state", state)
        _set_meta("bento_heartbeat", str(_time.time()))
        return {"ok": True}
    except Exception as e:
        return _err(e)


def heartbeat() -> dict:
    try:
        _set_meta("bento_heartbeat", str(_time.time()))
        return {"ok": True}
    except Exception as e:
        return _err(e)


def get_status() -> dict:
    # online = the voice loop has checked in within the last 10 seconds
    try:
        beat = float(_get_meta("bento_heartbeat") or 0)
        return {"state": _get_meta("bento_state") or "offline", "online": _time.time() - beat < 10}
    except Exception as e:
        return _err(e)


def request_talk(action: str) -> dict:
    # the dashboard's Talk / Stop button; the voice loop picks it up with take_talk_request()
    try:
        _set_meta("ptt", action)
        return {"ok": True}
    except Exception as e:
        return _err(e)


def get_demo_reset() -> dict:
    # when Reset demo was last pressed (0 = never); the voice loop clears its memory when this changes
    try:
        return {"at": float(_get_meta("demo_reset") or 0)}
    except Exception as e:
        return _err(e)


def take_talk_request() -> dict:
    try:
        req = _get_meta("ptt") or ""
        if req:
            con = connect(DEFAULT_DB)
            con.execute("DELETE FROM meta WHERE key='ptt'")
            con.commit()
            con.close()
        return {"request": req}
    except Exception as e:
        return _err(e)
