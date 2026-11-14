# Contract: engine_api.py

Agreed 2026-11-14. Don't change a name, argument or output field without telling the other person. Real example outputs live in tool_examples.json.


Plain Python functions returning JSON-serialisable dicts. Marco wraps the AI-facing ones as tools for the model.

Conventions:
- Mission time is reported as `"mission_day"` (int) and `"time"` (`"HH:MM"` string).
- Crew ids are lowercase first names. Demo crew: `"alex"` (the scenario Marco's tests already use, centred on mission day 47) and `"sam"`.
- Alert levels: `"green"`, `"amber"`, `"red"`.
- If Marco shares the TOOLS list from his `llm_eval.py` and its names or arguments differ from the ones below, align with his and update `contract.md`.

### AI-facing functions

`get_vitals(crew_id: str, hours: int = 24) -> dict`
```json
{"crew_id": "alex", "mission_day": 47, "time": "07:30",
 "metrics": [
  {"metric": "resting_heart_rate", "unit": "bpm", "latest": 68, "baseline": 58, "change_pct": 17, "trend_3d": "rising", "status": "amber"},
  {"metric": "sleep", "unit": "hours", "latest": 5.1, "baseline": 7.2, "change_pct": -29, "trend_3d": "falling", "status": "amber"},
  {"metric": "spo2", "unit": "%", "latest": 97, "baseline": 98, "change_pct": -1, "trend_3d": "stable", "status": "green"},
  {"metric": "o2_consumption_rest", "unit": "mL/min", "latest": 260, "baseline": 250, "change_pct": 4, "trend_3d": "stable", "status": "green"},
  {"metric": "exercise", "unit": "minutes/day", "latest": 0, "baseline": 90, "change_pct": -100, "trend_3d": "falling", "status": "amber"}
 ]}
```

`get_environment() -> dict`
```json
{"mission_day": 47, "time": "07:30",
 "radiation": {"dose_rate_msv_per_day": 1.8, "mission_dose_msv": 84.6, "status": "green"},
 "cabin": {"o2_pct": 21, "co2_mmhg": 3.2, "temp_c": 22, "status": "green"},
 "events": []}
```

`get_alerts(crew_id: str | None = None, active_only: bool = True) -> dict`
```json
{"alerts": [
  {"alert_id": "A47-03", "crew_id": "alex", "level": "amber", "code": "RHR_RISING",
   "message": "Resting heart rate 17% above baseline for 3 days",
   "since_mission_day": 45, "protocol_ids": ["fatigue_recovery"]}
 ]}
```

`log_symptom(crew_id: str, description: str, severity_0_10: int, transcript: str) -> dict`
```json
{"ok": true, "log_id": "L-0112", "mission_day": 47, "time": "07:34"}
```

`start_checkin(crew_id: str) -> dict`
```json
{"checkin_id": "C47-alex", "questions": [
  {"id": "sleep_quality", "text": "How well did you sleep, from 1, very badly, to 5, very well?", "min": 1, "max": 5},
  {"id": "sleepiness_kss", "text": "Karolinska Sleepiness Scale: 1 is extremely alert, 9 is very sleepy and fighting to stay awake.", "min": 1, "max": 9},
  {"id": "fatigue_sp", "text": "Samn-Perelli fatigue: 1 is fully alert, 7 is completely exhausted.", "min": 1, "max": 7},
  {"id": "mood", "text": "How is your mood, from 1, very low, to 5, very good?", "min": 1, "max": 5},
  {"id": "stress", "text": "How stressed do you feel, from 1, not at all, to 5, extremely?", "min": 1, "max": 5}
 ]}
```

`submit_checkin(checkin_id: str, answers: dict) -> dict`
```json
{"ok": true,
 "results": [{"id": "fatigue_sp", "value": 5, "baseline": 3, "flag": "higher than usual"}],
 "flags": ["Fatigue above personal baseline 2 days running"]}
```

`get_protocol(protocol_id: str) -> dict`
```json
{"protocol_id": "fatigue_recovery", "title": "...", "steps": ["...", "..."],
 "escalate_if": ["..."], "source": "<the published first-aid or NASA source this text came from>"}
```

`get_crew_log(crew_id: str, days: int = 7) -> dict`
```json
{"crew_id": "alex", "entries": [
  {"mission_day": 46, "time": "21:10", "type": "symptom", "text": "mild headache", "severity": 3},
  {"mission_day": 47, "time": "07:20", "type": "checkin", "summary": "Fatigue 5 of 7, baseline 3"}
 ]}
```

### Code-facing functions (used by Marco's code or the dashboard, not shown to the AI)

- `get_crew_profile(crew_id) -> dict`
  - Returns `crew_id`, `name`, `role`, `baselines`, `notes`, and a `summary_text` of 120 words or fewer built by code (no AI).
  - Example `summary_text`: "Alex, commander. Personal baselines: resting heart rate 58 bpm, sleep 7.2 h, SpO2 98%. Last 7 days: short sleep 3 nights; resting heart rate rising since day 45; mild headache logged day 46, severity 3; no exercise day 46. Active alerts: amber, resting heart rate rising."
- `raise_emergency(crew_id, trigger_phrase, transcript) -> {"ok": true, "alarm_id": "E-001"}`
  - Writes a red alert and logs the transcript. The dashboard shows a red banner.
- `record_pvt(crew_id, reaction_times_ms: list[int]) -> dict`
  - Returns `median_rt_ms`, `lapses`, `baseline_median_ms`, `change_pct`, `status`.
  - State the lapse threshold in code. The classic 10-minute PVT counts reaction times of 500 ms or more as lapses; pick and document a threshold for a 3-minute version.
- `list_protocols() -> [{"protocol_id": ..., "title": ...}]`
  - Marco puts this list in the system prompt so the AI knows which cards exist.
- Demo controls: `set_clock(mission_day, time)` and `trigger_event(name)`.
  - Needed to script the demo video, for example: jump to day 47, 07:30; fire the solar particle event now.

