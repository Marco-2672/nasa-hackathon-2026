# deterministic simulator + database seeder
# run: python -m engine.sim
# deletes data/mission.db and rebuilds; every value is seeded
# does not write alerts; engine.py does that

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

from .schema import DEFAULT_DB, connect, init_db

# scenario constants
SEED = 20261114
FIRST_DAY = 30
LAST_DAY = 50
STEP_MIN = 15
SLEEP_MIN = 7 * 60
EXERCISE_MIN = 23 * 60

# radiation: cruise is curiosity rad transit, spe is simulated
CRUISE_DOSE_RATE = 1.8
SPE_DOSE_RATE = 20.0
SPE_DAY = 47
SPE_START_MIN = 10 * 60
SPE_END_MIN = 16 * 60
SPE_EXTRA_DOSE = 4.5

# demo start: day 47 at 07:30, matching contract
DEMO_CLOCK = 47 * 1440 + 7 * 60 + 30

CREW = [
    {"crew_id": "alex", "name": "Alex", "role": "commander"},
    {"crew_id": "sam",  "name": "Sam",  "role": "mission specialist"},
]

# baselines stored exactly, fatigue_sp used by submit_checkin
BASELINES = {
    "alex": {
        "resting_heart_rate":  (58,  "bpm"),
        "sleep":               (7.2, "hours"),
        "spo2":                (98,  "%"),
        "o2_consumption_rest": (250, "mL/min"),
        "exercise":            (90,  "minutes/day"),
        "sleep_quality":       (4,   "score"),
        "sleepiness_kss":      (3,   "score"),
        "fatigue_sp":          (3,   "score"),
        "mood":                (4,   "score"),
        "stress":              (2,   "score"),
    },
    "sam": {
        "resting_heart_rate":  (55,  "bpm"),
        "sleep":               (7.5, "hours"),
        "spo2":                (98,  "%"),
        "o2_consumption_rest": (235, "mL/min"),
        "exercise":            (80,  "minutes/day"),
        "sleep_quality":       (4,   "score"),
        "sleepiness_kss":      (3,   "score"),
        "fatigue_sp":          (2,   "score"),
        "mood":                (4,   "score"),
        "stress":              (2,   "score"),
    },
}

# scripted alex values, keyed by mission day
ALEX_RHR = {45: 64, 46: 66, 47: 68, 48: 68, 49: 67, 50: 66}
ALEX_SLEEP = {45: 4.8, 46: 4.9, 47: 5.1}
ALEX_EXERCISE = {46: 0}


def _minutes(day: int, minute_of_day: int) -> int:
    return day * 1440 + minute_of_day


def _hhmm(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}:{minute_of_day % 60:02d}"


def _vitals_today(crew_id: str, day: int, rng: random.Random):
    # (rhr, spo2, o2_rest) for one crew-day
    if crew_id == "alex":
        if day == 47:
            return 68, 97, 260
        rhr = ALEX_RHR.get(day, 58 + rng.choice([-1, 0, 0, 0, 1]))
        return rhr, rng.choice([97, 98, 98, 98, 99]), 250 + rng.choice([-5, -3, 0, 0, 3, 5])
    rhr = 55 + rng.choice([-1, 0, 0, 0, 1])
    return rhr, rng.choice([97, 98, 98, 98, 99]), 235 + rng.choice([-5, -3, 0, 0, 3, 5])


def _sleep(crew_id: str, day: int, rng: random.Random) -> float:
    if crew_id == "alex":
        if day in ALEX_SLEEP:
            return ALEX_SLEEP[day]
        return round(7.2 + rng.uniform(-0.3, 0.3), 1)
    return round(7.5 + rng.uniform(-0.3, 0.3), 1)


def _exercise(crew_id: str, day: int, rng: random.Random) -> int:
    if crew_id == "alex":
        if day in ALEX_EXERCISE:
            return ALEX_EXERCISE[day]
        return rng.choice([75, 85, 90, 90, 95, 100])
    return rng.choice([65, 70, 75, 80, 85])


def _dose_rate(day: int, mod: int) -> float:
    if day == SPE_DAY and SPE_START_MIN <= mod < SPE_END_MIN:
        return SPE_DOSE_RATE
    return CRUISE_DOSE_RATE


def _mission_dose(day: int, mod: int) -> float:
    # daily grain so day 47 at 07:30 reads exactly 84.6
    base = round(CRUISE_DOSE_RATE * day, 1)
    if day < SPE_DAY:
        extra = 0.0
    elif day == SPE_DAY:
        if mod <= SPE_START_MIN:
            extra = 0.0
        elif mod >= SPE_END_MIN:
            extra = SPE_EXTRA_DOSE
        else:
            extra = round(SPE_EXTRA_DOSE * (mod - SPE_START_MIN) / (SPE_END_MIN - SPE_START_MIN), 1)
    else:
        extra = SPE_EXTRA_DOSE
    return round(base + extra, 1)


def _seed_static(con: sqlite3.Connection) -> None:
    # rows that don't depend on the day loop
    con.executemany(
        "INSERT INTO crew (crew_id, name, role) VALUES (?, ?, ?)",
        [(c["crew_id"], c["name"], c["role"]) for c in CREW],
    )
    con.executemany(
        "INSERT INTO baselines (crew_id, metric, unit, value) VALUES (?, ?, ?, ?)",
        [(cid, m, u, v) for cid, d in BASELINES.items() for m, (v, u) in d.items()],
    )

    # spe scheduled event, active_from/to live inside payload_json
    con.execute(
        "INSERT INTO scheduled_events (event_id, mission_day, time, minutes, name, payload_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            f"EV-SPE-{SPE_DAY}",
            SPE_DAY,
            _hhmm(SPE_START_MIN),
            _minutes(SPE_DAY, SPE_START_MIN),
            "solar_particle_event",
            json.dumps({
                "level": "red",
                "code": "SPE_SHELTER",
                "message": "Solar particle event: radiation dose rate elevated. Shelter protocol required.",
                "protocol_ids": ["radiation_shelter"],
                "dose_rate_msv_per_day": SPE_DOSE_RATE,
                "active_from_minutes": _minutes(SPE_DAY, SPE_START_MIN),
                "active_to_minutes":   _minutes(SPE_DAY, SPE_END_MIN),
            }),
        ),
    )

    # meta: seed, clock, pvt baseline
    con.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [
            ("seed", str(SEED)),
            ("first_day", str(FIRST_DAY)),
            ("last_day", str(LAST_DAY)),
            ("vitals_step_min", str(STEP_MIN)),
            ("clock_minutes", str(DEMO_CLOCK)),
            ("pvt_baseline_ms", "265"),
            ("scenario", "spaceapps-2026-bento-demo"),
        ],
    )

    seed_logs(con)


def seed_logs(con: sqlite3.Connection) -> None:
    # the seeded crew-log rows; also used by engine_api.reset_demo() between demo takes
    # seeded symptom: must match tool_examples.get_crew_log entry 0
    con.execute(
        "INSERT INTO symptoms (log_id, crew_id, mission_day, time, minutes, description, severity, transcript) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("L-0111", "alex", 46, "21:10", _minutes(46, 21 * 60 + 10),
         "mild headache", 3, "I have a mild headache."),
    )

    # seeded checkin: must match tool_examples.get_crew_log entry 1
    con.execute(
        "INSERT INTO checkins (checkin_id, crew_id, mission_day, time, minutes, answers_json, summary, completed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("C47-alex", "alex", 47, "07:20", _minutes(47, 7 * 60 + 20),
         json.dumps({"sleep_quality": 3, "sleepiness_kss": 6, "fatigue_sp": 5,
                     "mood": 3, "stress": 3}),
         "Fatigue 5 of 7, baseline 3", 1),
    )


def build(db_path: Path) -> None:
    # delete db file (and wal/shm siblings) then rebuild deterministically
    rng = random.Random(SEED)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        if p.exists():
            p.unlink()

    con = connect(db_path)
    try:
        init_db(con)
        _seed_static(con)

        # readings: every crew, every day 30-50
        readings: list[tuple] = []
        for crew in CREW:
            cid = crew["crew_id"]
            for day in range(FIRST_DAY, LAST_DAY + 1):
                rhr, spo2, o2 = _vitals_today(cid, day, rng)
                for mod in range(0, 1440, STEP_MIN):
                    t, m = _hhmm(mod), _minutes(day, mod)
                    readings.append((cid, day, t, m, "resting_heart_rate", rhr, "bpm"))
                    readings.append((cid, day, t, m, "spo2", spo2, "%"))
                    readings.append((cid, day, t, m, "o2_consumption_rest", o2, "mL/min"))
                readings.append((cid, day, _hhmm(SLEEP_MIN), _minutes(day, SLEEP_MIN),
                                 "sleep", _sleep(cid, day, rng), "hours"))
                readings.append((cid, day, _hhmm(EXERCISE_MIN), _minutes(day, EXERCISE_MIN),
                                 "exercise", _exercise(cid, day, rng), "minutes/day"))
        con.executemany(
            "INSERT INTO readings (crew_id, mission_day, time, minutes, metric, value, unit) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            readings,
        )

        # environment: cabin constant, radiation varies across the spe
        environment: list[tuple] = []
        for day in range(FIRST_DAY, LAST_DAY + 1):
            for mod in range(0, 1440, STEP_MIN):
                environment.append((
                    day, _hhmm(mod), _minutes(day, mod),
                    21, 3.2, 22,
                    _dose_rate(day, mod),
                    _mission_dose(day, mod),
                ))
        con.executemany(
            "INSERT INTO environment (mission_day, time, minutes, o2_pct, co2_mmhg, temp_c, "
            "dose_rate_msv_per_day, mission_dose_msv) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            environment,
        )
        con.commit()
    finally:
        con.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed data/mission.db with the demo scenario.")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    args = ap.parse_args(argv)

    build(Path(args.db))

    con = connect(Path(args.db))
    try:
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("crew", "baselines", "readings", "environment",
                            "scheduled_events", "symptoms", "checkins")}
        clock = con.execute("SELECT value FROM meta WHERE key='clock_minutes'").fetchone()[0]
    finally:
        con.close()

    print(f"seeded {args.db}  (clock_minutes={clock}, i.e. day 47 07:30)")
    for k, v in counts.items():
        print(f"  {k:17s} {v}")
    print("next: python -m engine.engine")
    return 0


if __name__ == "__main__":
    sys.exit(main())