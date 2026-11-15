# bento dashboard; run from repo root: streamlit run dashboard/app.py
# reads through engine_api
# live panel auto-refreshes every 2s; sidebar buttons fire demo events

from __future__ import annotations

import sys
from pathlib import Path

# ensure engine is importable when launched via `streamlit run dashboard/app.py`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from engine import engine_api as api
from engine.schema import connect

st.set_page_config(page_title="Bento — Crew Health", page_icon="🛰", layout="wide")


# --- helpers ---

LEVEL_EMOJI = {"green": "🟢", "amber": "🟡", "red": "🔴"}
LEVEL_BG = {"green": "#0f3d2e", "amber": "#3d320f", "red": "#4a1414"}
LEVEL_BORDER = {"green": "#3cb371", "amber": "#e0a800", "red": "#dc3545"}


def safe_call(fn, *args, **kwargs):
    # return (result, error); error is None on success
    result = fn(*args, **kwargs)
    if isinstance(result, dict) and "error" in result:
        return None, result["error"]
    return result, None


def banner_for(crew_id: str) -> tuple[str, str]:
    # priority: red alert > active event > amber alerts > nominal
    alerts_res, _ = safe_call(api.get_alerts, crew_id=crew_id, active_only=True)
    alerts = (alerts_res or {}).get("alerts", [])
    reds = [a for a in alerts if a["level"] == "red"]
    ambers = [a for a in alerts if a["level"] == "amber"]

    env_res, _ = safe_call(api.get_environment)
    events = (env_res or {}).get("events", [])
    rad_status = (env_res or {}).get("radiation", {}).get("status", "green")

    if reds:
        return "red", f"EMERGENCY — {reds[0]['message']}"
    if events:
        e = events[0]
        action = e["protocol_ids"][0] if e.get("protocol_ids") else "act"
        return "red", f"{e['message']}  ·  protocol: {action}"
    if rad_status == "red":
        return "red", "Radiation dose rate elevated"
    if ambers:
        codes = ", ".join(sorted({a["code"] for a in ambers}))
        return "amber", f"{len(ambers)} amber alert{'s' if len(ambers) != 1 else ''}: {codes}"
    return "green", "All systems nominal"


def vitals_df(crew_id: str) -> pd.DataFrame:
    data, err = safe_call(api.get_vitals, crew_id)
    if err:
        return pd.DataFrame({"error": [err]})
    rows = []
    for m in data["metrics"]:
        rows.append({
            "metric": m["metric"].replace("_", " "),
            "latest": m["latest"],
            "baseline": m["baseline"],
            "change %": m["change_pct"],
            "trend (3d)": m["trend_3d"],
            "unit": m["unit"],
            "status": LEVEL_EMOJI.get(m["status"], m["status"]),
        })
    return pd.DataFrame(rows)


def environment_df() -> pd.DataFrame:
    env, err = safe_call(api.get_environment)
    if err:
        return pd.DataFrame({"error": [err]})
    rows = [
        {"reading": "dose rate", "value": env["radiation"]["dose_rate_msv_per_day"],
         "unit": "mSv/day", "status": LEVEL_EMOJI.get(env["radiation"]["status"], "")},
        {"reading": "mission dose", "value": env["radiation"]["mission_dose_msv"],
         "unit": "mSv", "status": LEVEL_EMOJI.get(env["radiation"]["status"], "")},
        {"reading": "cabin O2", "value": env["cabin"]["o2_pct"],
         "unit": "%", "status": LEVEL_EMOJI.get(env["cabin"]["status"], "")},
        {"reading": "cabin CO2", "value": env["cabin"]["co2_mmhg"],
         "unit": "mmHg", "status": LEVEL_EMOJI.get(env["cabin"]["status"], "")},
        {"reading": "cabin temp", "value": env["cabin"]["temp_c"],
         "unit": "C", "status": LEVEL_EMOJI.get(env["cabin"]["status"], "")},
    ]
    return pd.DataFrame(rows)


def crew_log_df(crew_id: str) -> pd.DataFrame:
    log, err = safe_call(api.get_crew_log, crew_id, days=7)
    if err:
        return pd.DataFrame({"error": [err]})
    rows = []
    for e in log["entries"]:
        detail = f"{e['text']} (severity {e['severity']})" if e["type"] == "symptom" else e["summary"]
        rows.append({"day": e["mission_day"], "time": e["time"], "type": e["type"], "detail": detail})
    return pd.DataFrame(rows)


def clear_demo_state() -> None:
    # reset clock to day 47 07:30, delete all alerts and emergencies
    con = connect()
    try:
        con.execute("DELETE FROM alerts")
        con.execute("DELETE FROM emergencies")
        con.commit()
    finally:
        con.close()
    api.set_clock(47, "07:30")


# --- sidebar ----

with st.sidebar:
    st.header("Bento")
    st.caption("crew health companion · local only")
    st.divider()

    crew_id = st.radio("Crew", ["alex", "sam"], horizontal=True)

    st.divider()
    st.subheader("Demo controls")
    if st.button("Trigger solar particle event"):
        api.trigger_event("solar_particle_event")
        st.rerun()
    if st.button("Raise test emergency"):
        api.raise_emergency(crew_id, "test", "this is a test")
        st.rerun()
    if st.button("Reset demo"):
        clear_demo_state()
        st.rerun()

    st.divider()
    st.caption("auto-refresh: 2s")
    st.caption("source: engine/engine_api.py")


# --- live panel ---

@st.fragment(run_every="2s")
def live(crew_id: str) -> None:
    env, err = safe_call(api.get_environment)
    if err:
        st.error(f"environment: {err}")
        return
    st.markdown(f"#### Mission Day {env['mission_day']} · {env['time']}")

    level, text = banner_for(crew_id)
    st.markdown(
        f"<div style='background:{LEVEL_BG[level]};"
        f"border-left:6px solid {LEVEL_BORDER[level]};"
        f"padding:14px 18px;border-radius:6px;font-size:18px;margin-bottom:12px;'>"
        f"{LEVEL_EMOJI[level]} {text}</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([2, 1])
    with left:
        st.subheader(f"Vitals — {crew_id}")
        v = vitals_df(crew_id)
        if "error" in v.columns:
            st.error(v["error"].iloc[0])
        else:
            st.dataframe(v, use_container_width=True, hide_index=True)
    with right:
        st.subheader("Environment")
        e = environment_df()
        if "error" in e.columns:
            st.error(e["error"].iloc[0])
        else:
            st.dataframe(e, use_container_width=True, hide_index=True)

    st.subheader(f"Crew log — {crew_id} (last 7 days)")
    log = crew_log_df(crew_id)
    if "error" in log.columns:
        st.error(log["error"].iloc[0])
    elif log.empty:
        st.caption("no entries")
    else:
        st.dataframe(log, use_container_width=True, hide_index=True)


live(crew_id)