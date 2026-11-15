"""Push-to-talk voice loop.

mic -> faster-whisper -> emergency check -> llama-server + tools -> number guard -> Kokoro -> speaker

Run from the repo root:  python -m voice.voice_loop
Talk: press Enter (or the dashboard's Talk button) to start recording, again to stop.
Type a crew id (alex, sam) to switch speaker. q quits.
"""
import queue
import re
import select
import sys
import threading
import time
import urllib.error
import warnings

warnings.filterwarnings("ignore", category=UserWarning)    # keep the terminal clean for the demo video
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline

from voice import llm, safety
from voice.tools import run_tool

SR_IN = 16000
SR_OUT = 24000         # Kokoro outputs 24 kHz
VOICE = "bm_george"    # the voice you picked
LANG = VOICE[0]        # Kokoro's language code is the voice's first letter: a = American, b = British
CREW = ["alex", "sam"]
LEAD_IN_S = 0.1        # silence before each sentence; raise to 0.3 if the first word is still clipped
HISTORY_TURNS = 4      # past exchanges kept; tool messages are not kept, to save context
ALERT_POLL_S = 3       # how often to look for new engine alerts while waiting at the prompt
RED_CARD_STEPS = 2     # a red alert reads this many protocol steps word for word, then the AI briefs the crew
MAX_RECORD_S = 30      # recording stops by itself after this long

print("Loading speech models...")
stt = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
tts = KPipeline(lang_code=LANG, repo_id="hexgrad/Kokoro-82M")


class Speaker:
    """Plays audio on a background thread, so the next sentence is synthesised while this one plays.

    One output stream stays open for the whole session and is fed silence when idle. Opening a new
    stream per sentence (sd.play) lets the sound system put the device to sleep between sentences,
    and waking it clips the first word."""

    def __init__(self):
        self.q = queue.Queue()
        self.lead_in = np.zeros(int(SR_OUT * LEAD_IN_S), np.float32)
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        idle = np.zeros((int(SR_OUT * 0.05), 1), np.float32)
        with sd.OutputStream(samplerate=SR_OUT, channels=1, dtype="float32") as stream:
            while True:
                try:
                    audio = self.q.get(timeout=0.05)
                except queue.Empty:
                    stream.write(idle)  # keep the device awake
                    continue
                stream.write(audio.reshape(-1, 1))
                self.q.task_done()

    def say(self, text):
        for _, _, audio in tts(text, voice=VOICE):
            a = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
            self.q.put(np.concatenate([self.lead_in, a.astype(np.float32)]))

    def alarm(self):
        t = np.linspace(0, 0.25, int(SR_OUT * 0.25), endpoint=False)
        beep = (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        gap = np.zeros(int(SR_OUT * 0.1), np.float32)
        self.q.put(np.concatenate([beep, gap] * 3))

    def wait(self):
        self.q.join()


speaker = Speaker()


# ---------------------------------------------------------------- dashboard links (comms log, status, Talk button)
def log_line(speaker_name, text, crew=None, kind=""):
    run_tool("log_conversation", {"speaker": speaker_name, "text": text, "crew_id": crew, "kind": kind})


def set_state(state):
    run_tool("set_status", {"state": state})


def talk_request():
    res = run_tool("take_talk_request", {})
    return res.get("request", "") if isinstance(res, dict) else ""


def wait_for_command(prompt):
    """Wait for a typed line, or for the dashboard's Talk button (returns "" = start talking)."""
    print(prompt, end="", flush=True)
    last_beat = 0.0
    while True:
        ready, _, _ = select.select([sys.stdin], [], [], 0.2)
        if ready:
            return sys.stdin.readline().strip().lower()
        if time.time() - last_beat > 3:
            run_tool("heartbeat", {})
            last_beat = time.time()
        if talk_request() == "start":
            print("(Talk button)")
            return ""


def record_ptt():
    chunks = []

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    set_state("listening")
    print("  Recording... press Enter (or the dashboard's Stop button) to stop.")
    t0 = time.time()
    with sd.InputStream(samplerate=SR_IN, channels=1, dtype="float32", callback=cb):
        while time.time() - t0 < MAX_RECORD_S:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.readline()
                break
            if talk_request() == "stop":
                break
    set_state("thinking")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def transcribe(audio):
    segments, _ = stt.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def handle_emergency(crew, kind, phrase, transcript):
    """Plain code only: no language model on this path."""
    print(f"\n  !!! EMERGENCY ({kind}): heard '{phrase}'. Bypassing the AI.")
    card_id = safety.protocol_for(kind)
    run_tool("raise_emergency", {"crew_id": crew, "trigger_phrase": phrase, "transcript": transcript,
                                 "protocol_id": card_id})
    speaker.alarm()
    lines = [f"Emergency alarm raised for {crew.title()}. The crew has been alerted."]
    card = run_tool("get_protocol", {"protocol_id": card_id})
    if isinstance(card, dict) and card.get("steps"):
        lines += [card.get("title", "")] + list(card["steps"])
    else:
        print(f"  (no protocol card found for '{card_id}')")
    say_lines([line for line in lines if line], "emergency", crew)


CANCEL_RE = re.compile(r"\b(stop|cancel|pause|later|interrupt|skip)\b", re.I)


def checkin_question(state):
    q = state["questions"][state["i"]]
    return f"Question {state['i'] + 1} of {len(state['questions'])}. {q['text']}"


def checkin_answer(state, text):
    """The check-in runs in plain code: questions read exactly as written, answers parsed, then
    submitted. The model is not involved, so it cannot invent questions or interpret answers.
    Returns (lines_to_speak, still_active)."""
    if CANCEL_RE.search(text):
        return ["Check-in paused. Ask for your check-in again when you're ready."], False
    q = state["questions"][state["i"]]
    valid = {n for n in safety.user_numbers(text) if q["min"] <= n <= q["max"]}
    if len(valid) != 1:
        return [f"Please answer with one number from {q['min']} to {q['max']}."], True
    state["answers"][q["id"]] = int(valid.pop())
    state["i"] += 1
    if state["i"] < len(state["questions"]):
        return [checkin_question(state)], True
    result = run_tool("submit_checkin", {"checkin_id": state["checkin_id"], "answers": state["answers"]})
    if not isinstance(result, dict) or result.get("error"):
        print(f"  [warn] submit_checkin failed: {result}")
        return ["I couldn't save your check-in. Please let your crewmate know."], False
    flags = [f if f.endswith(".") else f + "." for f in result.get("flags") or []]
    return ["Thanks, that's your check-in done."] + (flags or ["Nothing stands out against your usual answers."]), False


def say_lines(lines, tag, crew=None):
    set_state("speaking")
    for line in lines:
        print(f"  {llm.NAME} ({tag}): {line}")
        log_line("bento", line, crew, tag)
        speaker.say(line)


# ---------------------------------------------------------------- speak first on new alerts
idle = threading.Event()          # set only while waiting at the prompt, so alerts never interrupt a turn
turn_lock = threading.Lock()      # one speaker at a time: a briefing and a crew turn never overlap
announce_lock = threading.Lock()
announcements = []                # alerts spoken since the last turn, added to the conversation history
prompt_text = [""]
session = {"crew": CREW[0], "history": []}   # shared with the alert watcher


def _alert_key(a):
    # crew + code + day, not alert_id: ids are renumbered whenever the engine recalculates
    return a.get("crew_id"), a.get("code"), str(a.get("alert_id", "")).split("-")[0]


def new_alert_groups(seen):
    """Alerts not seen before, grouped so a crew-wide event is announced once: [(level, code, message, items)]."""
    res = run_tool("get_alerts", {})
    if not isinstance(res, dict) or "alerts" not in res:
        return []
    current = {_alert_key(a): a.get("level") for a in res["alerts"]}
    for key in [k for k, level in seen.items() if level == "red" and k not in current]:
        del seen[key]  # a red alert that has cleared (e.g. demo reset) can be announced again
    fresh = [a for a in res["alerts"] if a.get("code") != "EMERGENCY" and _alert_key(a) not in seen]
    seen.update({_alert_key(a): a.get("level") for a in fresh})
    groups = {}
    for a in fresh:  # a crew-wide event arrives as one alert per crew member: announce it once
        groups.setdefault((a.get("level"), a.get("code"), a.get("message", "")), []).append(a)
    return [(level, code, message, items) for (level, code, message), items in groups.items()]


def _who(items):
    return "All crew" if len(items) > 1 else str(items[0].get("crew_id", "")).title()


def fixed_alert_lines(groups):
    """The plain-code part of an announcement. Red: headline + the first protocol steps word for word."""
    lines, red = [], False
    for level, code, message, items in groups:
        msg = message if message.endswith(".") else message + "."
        if level == "red":
            red = True
            lines.append(f"{_who(items)}, priority alert. {msg}")
            ids = items[0].get("protocol_ids") or []
            card = run_tool("get_protocol", {"protocol_id": ids[0]}) if ids else None
            if isinstance(card, dict) and card.get("steps"):
                lines += list(card["steps"])[:RED_CARD_STEPS]
        else:
            lines.append(f"{_who(items)}, new alert. {msg}")
    return [line for line in lines if line], red


def new_alert_lines(seen):
    """Plain-code announcement for alerts not seen before (no model). Used by the tests."""
    return fixed_alert_lines(new_alert_groups(seen))


def ai_briefing(groups, crew, history):
    """The AI reacts to the new situation: checks related readings, connects them, points to the card.
    Same guardrails as a crew turn: tool gates, number guard, speculation and promise filters."""
    alert_text = "; ".join(f"{level} alert for {_who(items)}: {message}" for level, code, message, items in groups)
    target = groups[0][3][0].get("crew_id") if len(groups) == 1 and len(groups[0][3]) == 1 else crew
    note = (f"(Ship alert system, not the crew speaking.) New alert: {alert_text}. "
            "Without being asked, brief the crew in at most 3 short sentences: check the related readings with your "
            "tools, say what changed against baseline or limits, and name the protocol card to follow. "
            "Only mention readings that relate to this alert. "
            "Do not repeat the alert word for word and do not log symptoms.")
    system, facts = llm.build_system_prompt(target, run_tool)
    messages = [{"role": "system", "content": system}] + history[-2 * HISTORY_TURNS:] + [{"role": "user", "content": note}]
    reply, results = llm.run_turn(messages, run_tool, target, note)
    earlier = [m["content"] for m in history if m["role"] == "assistant"]
    return safety.filter_reply(reply, safety.allowed_numbers(facts, alert_text, *results, *earlier))


def respond_to_alerts(seen, crew, history):
    """Speak first on new alerts: code part (alarm, headline, first steps), then the AI briefing.
    Returns (fixed_lines, red, briefing_lines, briefing_dropped)."""
    groups = new_alert_groups(seen)
    if not groups:
        return [], False, [], []
    fixed, red = fixed_alert_lines(groups)
    print()
    if red:
        speaker.alarm()
    say_lines(fixed, "alert")
    try:
        briefing, dropped = ai_briefing(groups, crew, history)
    except Exception as e:  # model server down: the code part has already been spoken
        print(f"  [warn] AI briefing failed: {e}")
        briefing, dropped = [], []
    for sentence, why in dropped:
        print(f"  [guard] dropped: {sentence!r} ({why})")
    say_lines(briefing, "briefing")
    return fixed, red, briefing, dropped


def alert_watcher(seen):
    while True:
        time.sleep(ALERT_POLL_S)
        if not idle.is_set():
            continue
        with turn_lock:
            if not idle.is_set():
                continue
            fixed, red, briefing, _ = respond_to_alerts(seen, session["crew"], list(session["history"]))
            if not fixed:
                continue
            speaker.wait()
            set_state("ready")
            with announce_lock:
                announcements.append(" ".join(fixed + briefing))
            print(prompt_text[0].lstrip("\n"), end="", flush=True)


def warm_up():
    """First calls are slow (model loading, caches). Do them now, not in the first demo exchange."""
    print("Warming up...")
    # Real speech is needed to warm the transcriber (silence never reaches its decoder),
    # so synthesise a phrase and resample it from 24 kHz to 16 kHz.
    clips = [a.cpu().numpy() if hasattr(a, "cpu") else np.asarray(a)
             for _, _, a in tts("Hello, how am I doing today?", voice=VOICE)]
    speech = np.concatenate(clips).astype(np.float32)
    speech16 = np.interp(np.arange(0, len(speech), SR_OUT / SR_IN), np.arange(len(speech)), speech).astype(np.float32)
    segments, _ = stt.transcribe(speech16, language="en", beam_size=1, vad_filter=True)
    list(segments)
    llm.warm_up(run_tool, CREW[0])


def take_turn(crew, checkin):
    """Record, transcribe and answer one crew turn. Returns the check-in state."""
    history = session["history"]
    audio = record_ptt()
    t0 = time.time()
    text = transcribe(audio)
    t_stt = time.time() - t0
    if not text:
        print("  (Heard nothing. Press Enter or Talk, speak, then press Enter or Stop.)")
        return checkin
    print(f"  You: {text}   (speech-to-text {t_stt:.1f} s)")
    log_line("crew", text, crew)

    hit = safety.check_emergency(text)
    if hit:
        handle_emergency(crew, hit[0], hit[1], text)
        return None

    if checkin:
        lines, active = checkin_answer(checkin, text)
        say_lines(lines, "check-in", crew)
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(lines)}]
        return checkin if active else None

    system, facts = llm.build_system_prompt(crew, run_tool)
    messages = ([{"role": "system", "content": system}]
                + history[-2 * HISTORY_TURNS:]
                + [{"role": "user", "content": text}])
    t1 = time.time()
    try:
        reply, results = llm.run_turn(messages, run_tool, crew, text)
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"  [error] model server not reachable ({e}). Start it with: "
              "nohup ~/spaceapps/start_llm.sh > ~/spaceapps/llama.log 2>&1 &")
        return checkin
    t_llm = time.time() - t1

    started = next((r for r in results if isinstance(r, dict) and r.get("checkin_id") and r.get("questions")), None)
    if started:
        checkin = {"checkin_id": started["checkin_id"], "questions": started["questions"], "i": 0, "answers": {}}
        lines = ["Let's do your check-in. Answer each question with a number.", checkin_question(checkin)]
        say_lines(lines, "check-in", crew)
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(lines)}]
        return checkin

    if llm.LAST_SEVERITY >= 10:
        print("\n  !!! EMERGENCY (severity 10 reported to the model). Raising the alarm.")
        run_tool("raise_emergency", {"crew_id": crew, "trigger_phrase": "severity 10 report", "transcript": text,
                                     "protocol_id": safety.DEFAULT_EMERGENCY_CARD})
        speaker.alarm()

    # Trusted numbers: this turn's tool results, the crew file, the user's words, and earlier
    # replies (they already passed the guard, so their numbers came from tool data).
    earlier = [m["content"] for m in history if m["role"] == "assistant"]
    allowed = safety.allowed_numbers(facts, text, *results, *earlier)
    spoken, dropped = safety.filter_reply(reply, allowed)
    for sentence, why in dropped:
        print(f"  [guard] dropped: {sentence!r} ({why})")
    set_state("speaking")
    t_first = None
    for sentence in spoken:
        log_line("bento", sentence, crew, "reply")
        speaker.say(sentence)
        if t_first is None:
            t_first = time.time() - t0

    print(f"  {llm.NAME}: {' '.join(spoken)}")
    print(f"  timings: speech-to-text {t_stt:.1f} s | model {t_llm:.1f} s | "
          f"first audio {t_first:.1f} s after you stopped")
    history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(spoken)}]
    return checkin


def main():
    warm_up()
    checkin = None
    seen = {}
    new_alert_groups(seen)  # alerts that already exist at startup count as known
    threading.Thread(target=alert_watcher, args=(seen,), daemon=True).start()
    set_state("ready")
    print("Ready. Enter (or the dashboard's Talk button) = start/stop recording. "
          "Type a crew id to switch speaker, q to quit.")
    while True:
        crew = session["crew"]
        prompt_text[0] = f"\n[{crew}] Enter to talk > "
        idle.set()
        cmd = wait_for_command(prompt_text[0])
        idle.clear()
        with turn_lock:  # if an alert briefing is under way, it finishes first
            speaker.wait()
            with announce_lock:  # the model should know what was just announced
                for a in announcements:
                    session["history"] += [
                        {"role": "user", "content": "(The ship's alert system has just announced something.)"},
                        {"role": "assistant", "content": a}]
                announcements.clear()
            if cmd == "q":
                break
            if cmd in CREW:
                session["crew"], session["history"], checkin = cmd, [], None
                print(f"  Now speaking as {cmd}.")
                continue
            checkin = take_turn(crew, checkin)
            speaker.wait()
            set_state("ready")
    set_state("offline")


if __name__ == "__main__":
    main()
