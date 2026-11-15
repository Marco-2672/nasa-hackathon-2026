"""Integration check: the real engine driven through the voice code, without audio or the model.

Run from the repo root, with nothing else running (it rebuilds data/mission.db before each check
and once more at the end, so the demo state is clean afterwards):

    python check_integration.py

Each check names its owner: [voice] = Marco, [engine] = partner, [content] = protocol cards (Sunday).
"""
import os
import sys
import types

os.environ["ENGINE"] = "real"

# Stand-ins for the audio libraries: this checks logic, so it loads no speech models and opens no audio device.
import numpy as np  # noqa: E402


class _Stream:
    def __init__(self, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def write(self, a): pass


class _TTS:
    def __init__(self, **k): pass
    def __call__(self, text, voice): yield text, "", np.zeros(10, np.float32)


_sd = types.ModuleType("sounddevice"); _sd.OutputStream = _sd.InputStream = _Stream
_fw = types.ModuleType("faster_whisper"); _fw.WhisperModel = lambda *a, **k: None
_ko = types.ModuleType("kokoro"); _ko.KPipeline = _TTS
sys.modules.update({"sounddevice": _sd, "faster_whisper": _fw, "kokoro": _ko})

from engine import engine_api as api, sim  # noqa: E402
from engine.engine import derive_alerts  # noqa: E402
from voice import llm, safety, voice_loop as vl  # noqa: E402
from voice.tools import run_tool  # noqa: E402

CREW = ["alex", "sam"]
WORST = {"sleep_quality": "min", "sleepiness_kss": "max", "fatigue_sp": "max", "mood": "min", "stress": "max"}


def reseed():
    sim.build(api.DEFAULT_DB)
    derive_alerts(api.DEFAULT_DB)


def run_checkin(crew, pick):
    """Drive a full check-in through the voice code. pick(question) -> spoken answer."""
    st = run_tool("start_checkin", {"crew_id": crew})
    if "error" in st:
        return None, st["error"]
    state = {"checkin_id": st["checkin_id"], "questions": st["questions"], "i": 0, "answers": {}}
    lines, active = [], True
    for q in st["questions"]:
        lines, active = vl.checkin_answer(state, pick(q))
    return lines, ("still active after all questions" if active else "")


# ---------------------------------------------------------------- checks: return (ok, detail)
def prompt_builds():
    bad = []
    for c in CREW:
        _, facts = llm.build_system_prompt(c, run_tool)
        if "Personal baselines" not in facts:
            bad.append(f"{c}: {facts[:80]!r}")
    return not bad, "; ".join(bad)


def ai_tools_no_errors():
    calls = [("get_vitals", {"crew_id": "alex"}), ("get_environment", {}), ("get_alerts", {}),
             ("get_crew_log", {"crew_id": "alex"}), ("start_checkin", {"crew_id": "alex"})]
    calls += [("get_protocol", {"protocol_id": p["protocol_id"]}) for p in run_tool("list_protocols", {})]
    errs = [f"{n}: {r['error']}" for n, a in calls if isinstance(r := run_tool(n, a), dict) and "error" in r]
    return not errs, "; ".join(errs)


def messy_model_args():
    errs = []
    for n, a in [("get_vitals", {"crew_id": "Alex"}), ("get_environment", {"crew_id": "alex"}),
                 ("get_alerts", {"crew_id": "ALEX", "active_only": True}), ("get_crew_log", {"crew_id": "alex", "days": "7"})]:
        r = run_tool(n, a)
        if isinstance(r, dict) and "error" in r:
            errs.append(f"{n}({a}): {r['error']}")
    return not errs, "; ".join(errs)


def guard_accepts_real_numbers():
    vit = run_tool("get_vitals", {"crew_id": "alex"})
    _, facts = llm.build_system_prompt("alex", run_tool)
    sentences = []
    for m in vit["metrics"][:3]:
        word = "above" if m["change_pct"] >= 0 else "below"
        sentences.append(f"Your {m['metric'].replace('_', ' ')} is {m['latest']} {m['unit']}, "
                         f"{abs(m['change_pct'])}% {word} your baseline of {m['baseline']}.")
    _, dropped = safety.filter_reply(" ".join(sentences), safety.allowed_numbers(facts, vit))
    return not dropped, "; ".join(f"{s!r}: {why}" for s, why in dropped)


def checkin_end_to_end():
    lines, err = run_checkin("alex", lambda q: str(q["min"]))
    if err:
        return False, err
    log = run_tool("get_crew_log", {"crew_id": "alex"})
    today = run_tool("get_vitals", {"crew_id": "alex"})["mission_day"]
    logged = any(e["type"] == "checkin" and e["mission_day"] == today for e in log.get("entries", []))
    return "check-in done" in lines[0] and logged, f"spoken={lines}, logged today={logged}"


def checkin_questions_are_questions():
    qs = run_tool("start_checkin", {"crew_id": "alex"})["questions"]
    bad = [q["id"] for q in qs if not q["text"].rstrip().endswith("?")]
    return not bad, f"these are read out word for word but aren't questions: {bad}"


def worst_answers_get_flagged():
    bad = []
    for c in CREW:
        lines, err = run_checkin(c, lambda q: str(q[WORST.get(q["id"], "max")]))
        if err or len(lines) < 2 or "Nothing stands out" in lines[1]:
            bad.append(f"{c}: {lines or err}")
    return not bad, "worst possible answers produced no flag -> " + "; ".join(bad)


def flags_backed_by_data():
    lines, _ = run_checkin("alex", lambda q: str(q[WORST.get(q["id"], "max")]))
    claims = [l for l in (lines or []) if "days running" in l]
    return not claims, f"spoken as fact with no earlier check-in in the data: {claims}"


def emergency_survives_engine_rerun():
    run_tool("raise_emergency", {"crew_id": "alex", "trigger_phrase": "bleeding everywhere", "transcript": "x"})
    derive_alerts(api.DEFAULT_DB)
    red = [a for a in run_tool("get_alerts", {})["alerts"] if a["level"] == "red"]
    return bool(red), "the red EMERGENCY alert is deleted when python -m engine.engine runs again"


def emergency_has_right_protocol():
    run_tool("raise_emergency", {"crew_id": "alex", "trigger_phrase": "bleeding everywhere", "transcript": "x",
                                 "protocol_id": "severe_bleeding"})
    red = [a for a in run_tool("get_alerts", {})["alerts"] if a["level"] == "red"]
    ok = bool(red) and red[0]["protocol_ids"] == ["severe_bleeding"]
    return ok, f"alert protocol_ids={red[0]['protocol_ids'] if red else None}; raise_emergency should accept protocol_id"


def solar_event_is_an_alert():
    api.trigger_event("solar_particle_event")
    alerts = run_tool("get_alerts", {})["alerts"]
    red = [a["code"] for a in alerts if a["level"] == "red"]
    return bool(red), f"after trigger_event the alerts are {[a['code'] for a in alerts]}: the companion can't announce it"


def clock_jump_consistent():
    api.set_clock(48, "07:30")
    stale = [a["alert_id"] for a in run_tool("get_alerts", {"crew_id": "alex"})["alerts"] if not a["alert_id"].startswith("A48")]
    profile_alerts = run_tool("get_crew_profile", {"crew_id": "alex"})["summary_text"].split("Active alerts:")[1].strip()
    return not stale, f"after set_clock(48): get_alerts still returns day-47 alerts {stale}, crew file says '{profile_alerts}'"


def demo_reset_restores_state():
    api.log_symptom("alex", "headache", 4, "take 1")
    st = api.start_checkin("alex"); api.submit_checkin(st["checkin_id"], {"fatigue_sp": 6})
    api.raise_emergency("alex", "test", "take 1"); api.trigger_event("solar_particle_event")
    api.reset_demo()
    log = run_tool("get_crew_log", {"crew_id": "alex"})["entries"]
    alerts = run_tool("get_alerts", {})["alerts"]
    clock = run_tool("get_vitals", {"crew_id": "alex"})
    ok = (len(log) == 2 and log[0]["text"] == "mild headache" and log[1]["summary"] == "Fatigue 5 of 7, baseline 3"
          and not [a for a in alerts if a["level"] == "red"] and len(alerts) == 3
          and (clock["mission_day"], clock["time"]) == (47, "07:30"))
    return ok, f"after reset_demo: log={log}, alerts={[a['code'] for a in alerts]}, clock={clock.get('mission_day')} {clock.get('time')}"


def solar_event_reannounced_after_reset():
    seen = {}
    vl.new_alert_lines(seen)
    api.trigger_event("solar_particle_event")
    first, _ = vl.new_alert_lines(seen)
    api.reset_demo()
    vl.new_alert_lines(seen)
    api.trigger_event("solar_particle_event")
    second, red = vl.new_alert_lines(seen)
    amber_repeat, _ = (api.reset_demo(), vl.new_alert_lines(seen))[1]
    return bool(first) and bool(second) and red and not amber_repeat, \
        f"take 1 announced={bool(first)}, take 2 announced={bool(second)}, amber alerts repeated={amber_repeat}"


def live_faults_alert_and_reset():
    api.inject_fault("co2_scrubber_fault")
    env = run_tool("get_environment", {})
    co2 = [a for a in run_tool("get_alerts", {})["alerts"] if a["code"] == "CO2_HIGH"]
    api.inject_fault("sam_spo2_drop")
    sp = [a for a in run_tool("get_alerts", {"crew_id": "sam"})["alerts"] if a["code"] == "SPO2_LOW"]
    crew_file = run_tool("get_crew_profile", {"crew_id": "alex"})["summary_text"]
    api.reset_demo()
    clean = not [a for a in run_tool("get_alerts", {})["alerts"] if a["code"] in ("CO2_HIGH", "SPO2_LOW")]
    ok = (env["cabin"]["co2_mmhg"] == 4.2 and env["cabin"]["status"] == "amber" and len(co2) == 2
          and sp and sp[0]["level"] == "red" and "CO2" in crew_file and clean
          and run_tool("get_environment", {})["cabin"]["co2_mmhg"] == 2.5)
    return ok, f"cabin={env['cabin']}, co2 alerts={len(co2)}, sam spo2={sp}, crew file has CO2={'CO2' in crew_file}, reset clean={clean}"


def severity_not_asked_twice():
    """Scripted model that asks for a severity the crew member already gave: code must remind it once."""
    reseed()
    script = [{"content": "I'll log your headache. What is the severity on a scale of 0 to 10?"},
              {"content": "", "tool_calls": [{"id": "1", "function": {"name": "log_symptom", "arguments":
                  '{"crew_id": "alex", "description": "headache", "severity_0_10": 3}'}}]},
              {"content": "I've logged your headache at 3."}]
    real_chat, llm.chat = llm.chat, lambda messages, tools=None, max_tokens=None: script.pop(0)
    try:
        _, _ = llm.build_system_prompt("alex", run_tool)
        msgs = [{"role": "user", "content": "I've had a headache for an hour, about a 3 out of 10."}]
        reply, results = llm.run_turn(msgs, run_tool, "alex", msgs[0]["content"])
    finally:
        llm.chat = real_chat
    logged = [e for e in run_tool("get_crew_log", {"crew_id": "alex", "days": 1}).get("entries", [])
              if e.get("type") == "symptom" and e.get("mission_day") == 47 and "headache" in str(e.get("text"))]
    reseed()
    return bool(logged) and not script, f"reply={reply!r}, logged={logged}"


def card_ids_spoken_as_names():
    llm.build_system_prompt("alex", run_tool)
    out = llm.speakable("Follow the co2high card, then the radiation_shelter and `breathing_difficulty` cards.")
    ok = "_" not in out and "co2high" not in out.lower() and "radiation shelter" in out and "breathing difficulty" in out
    return ok, out


def comms_and_status_logged():
    vl.say_lines(["Test line for the dashboard."], "briefing", "alex")
    vl.set_state("thinking")
    lines = run_tool("get_conversation", {})["lines"]
    status = run_tool("get_status", {})
    ok = bool(lines) and lines[-1]["text"] == "Test line for the dashboard." and lines[-1]["kind"] == "briefing" \
        and status["state"] == "thinking" and status["online"]
    return ok, f"last comms line={lines[-1] if lines else None}, status={status}"


def protocol_cards_ready():
    for f in api.list_faults():  # every fault's alert must point at a real card
        api.inject_fault(f["fault"])
    cards = {p["protocol_id"] for p in run_tool("list_protocols", {})}
    needed = {safety.protocol_for(k) for k in safety.EMERGENCY_PATTERNS} | {"radiation_shelter"}
    for a in run_tool("get_alerts", {})["alerts"]:
        needed |= set(a["protocol_ids"])
    missing = sorted(needed - cards)
    placeholder = sorted(c for c in cards & needed if "PLACEHOLDER" in str(run_tool("get_protocol", {"protocol_id": c})))
    return not missing and not placeholder, f"missing cards {missing}; still placeholder {placeholder}"


CHECKS = [
    ("voice", prompt_builds), ("voice", ai_tools_no_errors), ("voice", messy_model_args),
    ("voice", guard_accepts_real_numbers), ("voice", checkin_end_to_end),
    ("engine", checkin_questions_are_questions), ("engine", worst_answers_get_flagged),
    ("engine", flags_backed_by_data), ("engine", emergency_survives_engine_rerun),
    ("engine", emergency_has_right_protocol), ("engine", solar_event_is_an_alert),
    ("engine", clock_jump_consistent), ("engine", demo_reset_restores_state),
    ("voice", solar_event_reannounced_after_reset), ("engine", live_faults_alert_and_reset),
    ("voice", comms_and_status_logged), ("voice", severity_not_asked_twice), ("voice", card_ids_spoken_as_names),
    ("content", protocol_cards_ready),
]


def main():
    failed = 0
    for owner, check in CHECKS:
        reseed()
        try:
            ok, detail = check()
        except Exception as e:
            ok, detail = False, f"crashed: {type(e).__name__}: {e}"
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} [{owner:7}] {check.__name__}" + ("" if ok else f"\n       {detail}"))
    reseed()
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} passed. data/mission.db reset to day 47 07:30.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
