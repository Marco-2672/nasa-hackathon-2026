# db schema and connection helper shared by every engine module
# seeded tables (written by sim.py) + app tables (written by runtime)

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "mission.db"

# tables that exist from seed time onwards
_SCHEMA_SEEDED = """
-- one row per crew member
CREATE TABLE IF NOT EXISTS crew (
    crew_id TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    role    TEXT NOT NULL
);

-- personal baselines; stored exactly so get_vitals doesn't drift
CREATE TABLE IF NOT EXISTS baselines (
    crew_id TEXT NOT NULL,
    metric  TEXT NOT NULL,
    unit    TEXT NOT NULL,
    value   REAL NOT NULL,
    PRIMARY KEY (crew_id, metric)
);

-- time series; minutes = mission_day*1440 + hh*60 + mm
CREATE TABLE IF NOT EXISTS readings (
    reading_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    crew_id     TEXT NOT NULL,
    mission_day INTEGER NOT NULL,
    time        TEXT NOT NULL,
    minutes     INTEGER NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    unit        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_readings_lookup ON readings (crew_id, metric, minutes);

-- cabin + radiation; cabin values pinned so contract example reproduces
CREATE TABLE IF NOT EXISTS environment (
    mission_day           INTEGER NOT NULL,
    time                  TEXT NOT NULL,
    minutes               INTEGER NOT NULL,
    o2_pct                INTEGER NOT NULL,
    co2_mmhg              REAL NOT NULL,
    temp_c                INTEGER NOT NULL,
    dose_rate_msv_per_day REAL NOT NULL,
    mission_dose_msv      REAL NOT NULL,
    PRIMARY KEY (mission_day, minutes)
);

-- scripted events, payload_json carries active window
CREATE TABLE IF NOT EXISTS scheduled_events (
    event_id     TEXT PRIMARY KEY,
    mission_day  INTEGER NOT NULL,
    time         TEXT NOT NULL,
    minutes      INTEGER NOT NULL,
    name         TEXT NOT NULL,
    payload_json TEXT NOT NULL
);

-- key/value store: clock_minutes, seed, scenario name, pvt_baseline_ms
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# tables the running system writes to
_SCHEMA_APP = """
-- symptom reports; log_id format L-NNNN
CREATE TABLE IF NOT EXISTS symptoms (
    log_id      TEXT PRIMARY KEY,
    crew_id     TEXT NOT NULL,
    mission_day INTEGER NOT NULL,
    time        TEXT NOT NULL,
    minutes     INTEGER NOT NULL,
    description TEXT NOT NULL,
    severity    INTEGER NOT NULL,
    transcript  TEXT NOT NULL
);

-- daily self-report; checkin_id format C{day}-{crew}
CREATE TABLE IF NOT EXISTS checkins (
    checkin_id   TEXT PRIMARY KEY,
    crew_id      TEXT NOT NULL,
    mission_day  INTEGER NOT NULL,
    time         TEXT NOT NULL,
    minutes      INTEGER NOT NULL,
    answers_json TEXT NOT NULL,
    summary      TEXT NOT NULL,
    completed    INTEGER NOT NULL DEFAULT 1
);

-- emergencies caught by pre-ai phrase check; alarm_id format E-NNN
CREATE TABLE IF NOT EXISTS emergencies (
    alarm_id       TEXT PRIMARY KEY,
    crew_id        TEXT NOT NULL,
    mission_day    INTEGER NOT NULL,
    time           TEXT NOT NULL,
    minutes        INTEGER NOT NULL,
    trigger_phrase TEXT NOT NULL,
    transcript     TEXT NOT NULL
);

-- derived alerts; alert_id format A{day}-NN
CREATE TABLE IF NOT EXISTS alerts (
    alert_id          TEXT PRIMARY KEY,
    crew_id           TEXT NOT NULL,
    level             TEXT NOT NULL,
    code              TEXT NOT NULL,
    message           TEXT NOT NULL,
    since_mission_day INTEGER NOT NULL,
    created_day       INTEGER NOT NULL,
    protocol_ids_json TEXT NOT NULL,
    resolved_day      INTEGER
);
"""


def init_db(con: sqlite3.Connection) -> None:
    # create every table if missing, safe to call repeatedly
    con.executescript(_SCHEMA_SEEDED)
    con.executescript(_SCHEMA_APP)
    con.commit()


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    # WAL so engine can write while dashboard reads, busy_timeout avoids lock errors
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con