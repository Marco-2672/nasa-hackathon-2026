# Bento mission-control dashboard. Run from the repo root:  streamlit run dashboard/app.py
# Reads everything through engine_api. Refreshes every 1.5 s. New alerts pop up and red alerts flash.
# The control panel injects live faults, resets the demo and holds the Talk button for the voice loop.

from __future__ import annotations

import html
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from engine import engine_api as api

st.set_page_config(page_title="BENTO · Crew Health", page_icon="🛰", layout="wide", initial_sidebar_state="collapsed")

CYAN, GREEN, AMBER, RED, DIM = "#00e5ff", "#2ee59d", "#ffb020", "#ff3b5c", "#7d8ba6"
LEVEL_COLOR = {"green": GREEN, "amber": AMBER, "red": RED}
METRIC_NAMES = {"resting_heart_rate": "RESTING HR", "sleep": "SLEEP", "spo2": "SpO₂",
                "o2_consumption_rest": "O₂ USE · REST", "exercise": "EXERCISE"}
UNITS = {"bpm": "bpm", "hours": "h", "%": "%", "mL/min": "mL/min", "minutes/day": "min"}
POPUP_SECONDS = 9

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {{ --cyan:{CYAN}; --green:{GREEN}; --amber:{AMBER}; --red:{RED}; --dim:{DIM}; }}
.stApp {{
  background:
    radial-gradient(1px 1px at 12% 18%, #ffffff55 50%, transparent 51%),
    radial-gradient(1px 1px at 68% 32%, #ffffff44 50%, transparent 51%),
    radial-gradient(1px 1px at 82% 74%, #ffffff33 50%, transparent 51%),
    radial-gradient(1px 1px at 35% 85%, #ffffff44 50%, transparent 51%),
    radial-gradient(1200px 600px at 80% -10%, #0b2a4a 0%, transparent 60%),
    radial-gradient(900px 500px at -10% 110%, #1a0f3a 0%, transparent 60%),
    #04060c;
  color: #d8e3f0;
}}
header[data-testid="stHeader"], #MainMenu, footer {{ visibility: hidden; height: 0; }}
.block-container {{ padding-top: 0.8rem; padding-bottom: 0.5rem; max-width: 100%; }}
* {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
.orb {{ font-family: 'Orbitron', 'JetBrains Mono', monospace; letter-spacing: .14em; text-transform: uppercase; }}
.panel {{ background: rgba(9,16,30,.82); border: 1px solid rgba(0,229,255,.22); border-radius: 10px;
          padding: 12px 14px; box-shadow: inset 0 0 24px rgba(0,229,255,.06), 0 0 18px rgba(0,0,0,.4); margin-bottom: 10px; }}
.ptitle {{ font-family: 'Orbitron', monospace; font-size: 12px; letter-spacing: .22em; color: var(--cyan);
           text-transform: uppercase; margin-bottom: 8px; display:flex; justify-content:space-between; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between; padding: 10px 16px; margin-bottom: 10px;
           border-bottom: 1px solid rgba(0,229,255,.25); background: linear-gradient(90deg, rgba(0,229,255,.08), transparent 60%); }}
.brand {{ font-family:'Orbitron'; font-size: 26px; font-weight:700; letter-spacing:.3em; color:#fff; }}
.brand span {{ color: var(--cyan); }}
.sub {{ font-size: 11px; color: var(--dim); letter-spacing: .18em; }}
.clock {{ font-family:'Orbitron'; font-size: 22px; color:#fff; letter-spacing:.2em; text-align:center; }}
.clock small {{ display:block; font-size: 10px; color: var(--dim); letter-spacing:.3em; }}
.pill {{ display:inline-flex; align-items:center; gap:10px; padding: 8px 16px; border-radius: 999px;
         border: 1px solid currentColor; font-family:'Orbitron'; font-size: 13px; letter-spacing:.2em; }}
.dot {{ width:12px; height:12px; border-radius:50%; background: currentColor; box-shadow: 0 0 10px currentColor; }}
.pulse {{ animation: pulse 1s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity:1; transform:scale(1); }} 50% {{ opacity:.3; transform:scale(.7); }} }}
.bars {{ display:inline-flex; gap:3px; align-items:flex-end; height:14px; }}
.bars i {{ width:3px; background: currentColor; animation: eq .8s ease-in-out infinite; }}
.bars i:nth-child(2) {{ animation-delay:.15s }} .bars i:nth-child(3) {{ animation-delay:.3s }} .bars i:nth-child(4) {{ animation-delay:.45s }}
@keyframes eq {{ 0%,100% {{ height:3px }} 50% {{ height:14px }} }}
.banner {{ border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; font-family:'Orbitron'; letter-spacing:.12em;
           font-size: 17px; display:flex; gap:14px; align-items:center; }}
.banner .msg {{ font-family:'JetBrains Mono'; letter-spacing:0; font-size: 16px; color:#fff; }}
.b-red {{ background: rgba(255,59,92,.18); border: 2px solid var(--red); color: var(--red); animation: flash 1s steps(2) infinite; }}
.b-amber {{ background: rgba(255,176,32,.12); border: 1px solid var(--amber); color: var(--amber); animation: glow 2.2s ease-in-out infinite; }}
.b-green {{ background: rgba(46,229,157,.08); border: 1px solid rgba(46,229,157,.5); color: var(--green); }}
@keyframes flash {{ 0% {{ background: rgba(255,59,92,.45); box-shadow: 0 0 40px rgba(255,59,92,.7); }} 100% {{ background: rgba(255,59,92,.08); box-shadow:none; }} }}
@keyframes glow {{ 0%,100% {{ box-shadow: 0 0 0 rgba(255,176,32,0); }} 50% {{ box-shadow: 0 0 22px rgba(255,176,32,.45); }} }}
.red-edge {{ position: fixed; inset: 0; pointer-events: none; z-index: 9990; box-shadow: inset 0 0 90px rgba(255,59,92,.65);
             animation: edge 1s ease-in-out infinite alternate; }}
@keyframes edge {{ from {{ opacity:.25 }} to {{ opacity:1 }} }}
.popups {{ position: fixed; top: 86px; right: 22px; z-index: 9995; display:flex; flex-direction:column; gap:10px; width: 420px; }}
.popup {{ background: rgba(8,12,24,.96); border-left: 6px solid; border-radius: 8px; padding: 14px 16px;
          box-shadow: 0 10px 40px rgba(0,0,0,.6); animation: slidein .45s cubic-bezier(.2,.9,.3,1.2); }}
.popup .k {{ font-family:'Orbitron'; font-size: 12px; letter-spacing:.2em; }}
.popup .t {{ color:#fff; font-size: 15px; margin-top: 6px; }}
.popup.red {{ border-color: var(--red); animation: slidein .45s cubic-bezier(.2,.9,.3,1.2), flash 1s steps(2) infinite; }}
.popup.amber {{ border-color: var(--amber); }}
@keyframes slidein {{ from {{ transform: translateX(120%); opacity:0 }} to {{ transform:none; opacity:1 }} }}
.crewhead {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom: 8px; }}
.crewname {{ font-family:'Orbitron'; font-size: 18px; color:#fff; letter-spacing:.2em; }}
.role {{ color: var(--dim); font-size: 11px; letter-spacing:.15em; }}
.chip {{ font-family:'Orbitron'; font-size: 11px; letter-spacing:.18em; padding: 3px 10px; border-radius: 4px; border:1px solid currentColor; }}
.tiles {{ display:grid; grid-template-columns: repeat(5, 1fr); gap: 8px; }}
.tile {{ background: rgba(0,229,255,.04); border:1px solid rgba(125,139,166,.25); border-top: 3px solid; border-radius: 6px; padding: 8px; }}
.tile .n {{ font-size: 10px; color: var(--dim); letter-spacing:.14em; }}
.tile .v {{ font-family:'Orbitron'; font-size: 24px; color:#fff; margin: 4px 0 2px; }}
.tile .v small {{ font-family:'JetBrains Mono'; font-size: 11px; color: var(--dim); margin-left: 3px; }}
.tile .d {{ font-size: 11px; }}
.tile.red {{ animation: flash 1s steps(2) infinite; }}
.env {{ display:grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.comms {{ font-size: 13px; line-height: 1.55; max-height: 300px; overflow-y: auto; }}
.comms .row {{ display:flex; gap:10px; padding: 3px 0; border-bottom: 1px dashed rgba(125,139,166,.15); }}
.comms .ts {{ color: var(--dim); min-width: 70px; }}
.comms .who {{ min-width: 92px; font-family:'Orbitron'; font-size: 11px; letter-spacing:.14em; padding-top: 2px; }}
.c-crew {{ color: var(--cyan); }} .c-bento {{ color:#e9f2ff; }} .c-alert {{ color: var(--amber); }}
.c-emergency {{ color: var(--red); }} .c-briefing {{ color: #8ef5d0; }}
div[data-testid="stButton"] > button {{ width: 100%; background: rgba(0,229,255,.06); color: var(--cyan);
   border: 1px solid rgba(0,229,255,.55); border-radius: 6px; font-family: 'Orbitron', monospace; letter-spacing: .14em;
   font-size: 12px; padding: 10px 6px; transition: all .15s; }}
div[data-testid="stButton"] > button:hover {{ box-shadow: 0 0 18px rgba(0,229,255,.45); color:#fff; border-color: var(--cyan); }}
div[data-testid="stButton"] > button:disabled {{ opacity: .35; }}
.st-key-talk button {{ font-size: 18px !important; padding: 18px 6px !important; background: rgba(46,229,157,.12) !important;
   color: var(--green) !important; border: 2px solid var(--green) !important; }}
.st-key-stop button {{ font-size: 18px !important; padding: 18px 6px !important; background: rgba(255,59,92,.22) !important;
   color: #fff !important; border: 2px solid var(--red) !important; animation: flash 1s steps(2) infinite; }}
.st-key-f_surprise button {{ color: #c79bff !important; border-color: #c79bff !important; background: rgba(199,155,255,.08) !important; }}
.st-key-reset button, .st-key-test_em button {{ color: var(--dim) !important; border-color: rgba(125,139,166,.5) !important; }}
</style>
""", unsafe_allow_html=True)


def ok(res, default=None):
    return default if (res is None or (isinstance(res, dict) and "error" in res)) else res


def worst(levels):
    return "red" if "red" in levels else "amber" if "amber" in levels else "green"


def chart_vitals(crew_id: str):
    hr = ok(api.get_series(crew_id, "resting_heart_rate", 48), {"points": []})["points"]
    sp = ok(api.get_series(crew_id, "spo2", 48), {"points": []})["points"]
    df = pd.DataFrame([{"h": p["hours"], "v": p["value"], "s": "Resting HR (bpm)"} for p in hr] +
                      [{"h": p["hours"], "v": p["value"], "s": "SpO₂ (%)"} for p in sp])
    if df.empty:
        return None
    base = alt.Chart(df).encode(x=alt.X("h:Q", title="hours before now", scale=alt.Scale(domain=[-48, 0])))
    hr_line = base.transform_filter(alt.datum.s == "Resting HR (bpm)").mark_line(color=CYAN, strokeWidth=2, interpolate="step-after").encode(
        y=alt.Y("v:Q", title="HR bpm", scale=alt.Scale(zero=False)))
    sp_line = base.transform_filter(alt.datum.s == "SpO₂ (%)").mark_line(color="#ff6ad5", strokeWidth=2, interpolate="step-after").encode(
        y=alt.Y("v:Q", title="SpO₂ %", scale=alt.Scale(domain=[85, 100])))
    return style(alt.layer(hr_line, sp_line).resolve_scale(y="independent").properties(height=120))


def chart_co2():
    co2 = ok(api.get_co2_series(24), {"points": [], "limit_mmhg": 3.0})
    df = pd.DataFrame([{"h": p["hours"], "v": p["value"]} for p in co2["points"]])
    if df.empty:
        return None
    line = alt.Chart(df).mark_area(line={"color": AMBER}, color=alt.Gradient(
        gradient="linear", stops=[alt.GradientStop(color="rgba(255,176,32,.35)", offset=0),
                                  alt.GradientStop(color="rgba(255,176,32,0)", offset=1)], x1=1, x2=1, y1=0, y2=1),
        interpolate="step-after").encode(x=alt.X("h:Q", title="hours before now", scale=alt.Scale(domain=[-24, 0])),
                                         y=alt.Y("v:Q", title="CO₂ mmHg", scale=alt.Scale(domain=[0, 6])))
    rule = alt.Chart(pd.DataFrame({"y": [co2["limit_mmhg"]]})).mark_rule(color=RED, strokeDash=[6, 4]).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"y": [co2["limit_mmhg"]], "t": ["NASA LIMIT 3 mmHg"]})).mark_text(
        color=RED, align="left", dx=4, dy=-7, fontSize=10).encode(y="y:Q", x=alt.value(4), text="t:N")
    return style(alt.layer(line, rule, label).properties(height=140))


def show_chart(chart):
    # width="stretch" on newer Streamlit, use_container_width on older versions
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def style(chart):
    return chart.configure(background="transparent").configure_view(stroke=None).configure_axis(
        labelColor=DIM, titleColor=DIM, gridColor="rgba(125,139,166,.12)", domainColor="rgba(125,139,166,.3)",
        labelFont="JetBrains Mono", titleFont="JetBrains Mono", titleFontSize=10, labelFontSize=10)


# ---------------------------------------------------------------- control actions (run before the refresh)
def act_talk():
    api.request_talk("start")


def act_stop():
    api.request_talk("stop")


def act_fault(kind):
    api.inject_fault(kind)


def act_reset():
    api.reset_demo()
    st.session_state.pop("seen_alerts", None)


def act_test_emergency():
    api.raise_emergency("alex", "test alarm from the control panel", "test", "general")


@st.fragment(run_every=1.5)
def live():
    env = ok(api.get_environment(), {})
    alerts = ok(api.get_alerts(), {"alerts": []})["alerts"]
    status = ok(api.get_status(), {"state": "offline", "online": False})
    day, clock = env.get("mission_day", 0), env.get("time", "--:--")

    # pop-ups for alerts seen for the first time (existing ones at page load are not popped)
    now = time.time()
    keys = {(a["crew_id"], a["code"], a["alert_id"].split("-")[0]): a for a in alerts}
    seen = st.session_state.setdefault("seen_alerts", {k: 0.0 for k in keys})
    for k in keys:
        seen.setdefault(k, now)
    for k in [k for k in seen if k not in keys]:
        del seen[k]
    fresh, grouped = [], set()
    for k, first in seen.items():
        a = keys[k]
        if now - first < POPUP_SECONDS and (a["code"], a["message"]) not in grouped:
            grouped.add((a["code"], a["message"]))
            crews = sorted({x["crew_id"] for x in alerts if x["code"] == a["code"] and x["message"] == a["message"]})
            fresh.append((a, "ALL CREW" if len(crews) > 1 else a["crew_id"].upper()))
    red_active = any(a["level"] == "red" for a in alerts)

    # top bar
    state = status["state"] if status.get("online") else "offline"
    s_color = {"ready": GREEN, "listening": RED, "thinking": AMBER, "speaking": CYAN}.get(state, DIM)
    s_icon = ("<span class='bars'><i></i><i></i><i></i><i></i></span>" if state == "speaking"
              else f"<span class='dot {'pulse' if state in ('listening', 'thinking') else ''}'></span>")
    st.markdown(f"""
    <div class='topbar'>
      <div><div class='brand'>BENTO<span>//</span>CREW HEALTH</div>
           <div class='sub'>ON-BOARD AI · RUNS LOCALLY · NO EARTH LINK NEEDED · MARS TRANSIT</div></div>
      <div class='clock'><small>MISSION ELAPSED</small>DAY {day:03d} · {clock}</div>
      <div style='text-align:right'><span class='pill' style='color:{s_color}'>{s_icon} BENTO {state.upper()}</span>
           <div class='sub' style='margin-top:6px'>EARTH ONE-WAY DELAY 3–22 MIN</div></div>
    </div>""", unsafe_allow_html=True)

    # alert banner + flashing edge + pop-ups
    top = sorted(alerts, key=lambda a: {"red": 0, "amber": 1}.get(a["level"], 2))
    if top and top[0]["level"] == "red":
        a = top[0]
        st.markdown(f"<div class='banner b-red'>⚠ PRIORITY ALERT <span class='msg'>{html.escape(a['message'])}</span></div>",
                    unsafe_allow_html=True)
    elif top:
        n = len({a['code'] for a in top})
        st.markdown(f"<div class='banner b-amber'>◆ {n} ACTIVE ALERT{'S' if n != 1 else ''} "
                    f"<span class='msg'>{html.escape(' · '.join(sorted({a['message'] for a in top})[:3]))}</span></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown("<div class='banner b-green'>● ALL SYSTEMS NOMINAL</div>", unsafe_allow_html=True)
    if red_active:
        st.markdown("<div class='red-edge'></div>", unsafe_allow_html=True)
    if fresh:
        cards = "".join(f"<div class='popup {a['level']}'><div class='k' style='color:{LEVEL_COLOR.get(a['level'], CYAN)}'>"
                        f"{'⚠ PRIORITY ALERT' if a['level'] == 'red' else '◆ NEW ALERT'} · {who}</div>"
                        f"<div class='t'>{html.escape(a['message'])}</div></div>" for a, who in fresh)
        st.markdown(f"<div class='popups'>{cards}</div>", unsafe_allow_html=True)

    # crew + environment
    left, right = st.columns([2.3, 1])
    with left:
        cols = st.columns(2)
        for col, crew_id in zip(cols, ("alex", "sam")):
            vit = ok(api.get_vitals(crew_id), {"metrics": []})
            prof = ok(api.get_crew_profile(crew_id), {})
            crew_alerts = [a for a in alerts if a["crew_id"] == crew_id]
            lvl = worst([m["status"] for m in vit["metrics"]] + [a["level"] for a in crew_alerts])
            tiles = ""
            for m in vit["metrics"]:
                c = LEVEL_COLOR.get(m["status"], DIM)
                arrow = "▲" if m["change"] > 0 else "▼" if m["change"] < 0 else "■"
                tiles += (f"<div class='tile {m['status'] if m['status'] == 'red' else ''}' style='border-top-color:{c}'>"
                          f"<div class='n'>{METRIC_NAMES.get(m['metric'], m['metric'])}</div>"
                          f"<div class='v'>{m['latest']:g}<small>{UNITS.get(m['unit'], m['unit'])}</small></div>"
                          f"<div class='d' style='color:{c}'>{arrow} {m['change_pct']:+d}% · base {m['baseline']:g}</div></div>")
            with col:
                st.markdown(f"""<div class='panel'><div class='crewhead'>
                    <div><span class='crewname'>{html.escape(prof.get('name', crew_id)).upper()}</span>
                    <span class='role'> · {html.escape(prof.get('role', '')).upper()}</span></div>
                    <span class='chip' style='color:{LEVEL_COLOR[lvl]}'>{lvl.upper()}</span></div>
                    <div class='tiles'>{tiles}</div></div>""", unsafe_allow_html=True)
                ch = chart_vitals(crew_id)
                if ch is not None:
                    show_chart(ch)
    with right:
        rad, cab = env.get("radiation", {}), env.get("cabin", {})
        rc, cc = LEVEL_COLOR.get(rad.get("status"), DIM), LEVEL_COLOR.get(cab.get("status"), DIM)
        st.markdown(f"""<div class='panel'><div class='ptitle'><span>Environment</span><span>cabin + space weather</span></div>
          <div class='env'>
            <div class='tile {'red' if rad.get('status') == 'red' else ''}' style='border-top-color:{rc}'><div class='n'>DOSE RATE</div>
                 <div class='v'>{rad.get('dose_rate_msv_per_day', 0):g}<small>mSv/day</small></div></div>
            <div class='tile' style='border-top-color:{CYAN}'><div class='n'>MISSION DOSE</div>
                 <div class='v'>{rad.get('mission_dose_msv', 0):g}<small>mSv</small></div></div>
            <div class='tile {'red' if cab.get('status') == 'red' else ''}' style='border-top-color:{cc}'><div class='n'>CABIN CO₂</div>
                 <div class='v'>{cab.get('co2_mmhg', 0):g}<small>mmHg</small></div></div>
            <div class='tile' style='border-top-color:{GREEN}'><div class='n'>CABIN O₂ · TEMP</div>
                 <div class='v'>{cab.get('o2_pct', 0)}<small>%</small> {cab.get('temp_c', 0)}<small>°C</small></div></div>
          </div></div>""", unsafe_allow_html=True)
        ch = chart_co2()
        if ch is not None:
            show_chart(ch)

    # comms log + control panel
    lc, rc_ = st.columns([1.7, 1])
    with lc:
        lines = ok(api.get_conversation(14), {"lines": []})["lines"]
        rows = ""
        for l in lines:
            kind = l["kind"] if l["speaker"] == "bento" and l["kind"] in ("alert", "emergency", "briefing") else l["speaker"]
            who = (l["crew_id"] or "crew").upper() if l["speaker"] == "crew" else {
                "alert": "BENTO ⚠", "emergency": "BENTO ✚", "briefing": "BENTO ◇"}.get(kind, "BENTO")
            rows += (f"<div class='row'><span class='ts'>{l['mission_day'] or ''}·{l['time'] or ''}</span>"
                     f"<span class='who c-{kind}'>{who}</span><span class='c-{kind}'>{html.escape(l['text'])}</span></div>")
        st.markdown(f"<div class='panel'><div class='ptitle'><span>Comms log</span><span>voice · transcribed on board</span></div>"
                    f"<div class='comms'>{rows or '<span style=color:#7d8ba6>Waiting for the crew…</span>'}</div></div>",
                    unsafe_allow_html=True)
    with rc_:
        st.markdown("<div class='ptitle' style='margin-top:4px'><span>Control panel</span><span>live</span></div>",
                    unsafe_allow_html=True)
        if state == "listening":
            st.button("■  STOP  · SEND TO BENTO", key="stop", on_click=act_stop)
        else:
            st.button("◉  TALK TO BENTO", key="talk", on_click=act_talk, disabled=not status.get("online"))
        active = set(ok(api.active_faults(), []) or [])
        st.markdown("<div class='sub' style='margin:8px 0 4px'>INJECT A LIVE FAULT</div>", unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        faults = ok(api.list_faults(), []) or []
        for i, f in enumerate(faults):
            with (f1 if i % 2 == 0 else f2):
                st.button(f["label"].upper(), key=f"f_{f['fault']}", on_click=act_fault, args=(f["fault"],),
                          disabled=f["fault"] in active)
        with (f1 if len(faults) % 2 == 0 else f2):
            st.button("✦ SURPRISE FAULT", key="f_surprise", on_click=act_fault, args=("surprise",),
                      disabled=len(active) >= len(faults))
        r1, r2 = st.columns(2)
        with r1:
            st.button("↺ RESET DEMO", key="reset", on_click=act_reset)
        with r2:
            st.button("TEST ALARM", key="test_em", on_click=act_test_emergency)


live()
