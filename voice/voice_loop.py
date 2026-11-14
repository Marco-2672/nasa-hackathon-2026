"""Push-to-talk voice loop.

mic -> faster-whisper -> emergency check -> llama-server + tools -> number guard -> Kokoro -> speaker

Run from the repo root:  python -m voice.voice_loop
Keys: Enter to start recording, Enter to stop. Type a crew id (alex, sam) to switch speaker. q quits.
"""
import queue
import threading
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from kokoro import KPipeline

from voice import llm, safety
from voice.tools import run_tool

SR_IN = 16000
SR_OUT = 24000         # Kokoro outputs 24 kHz
LANG = "b"             # "a" American, "b" British: match your TTS test
VOICE = "bm_george"     # set to the voice you picked
CREW = ["alex", "sam"]
HISTORY_TURNS = 4      # past exchanges kept; tool messages are not kept, to save context
FALLBACK = "I can't confirm those numbers right now. Please check the dashboard."

print("Loading speech models...")
stt = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=8)
tts = KPipeline(lang_code=LANG)


class Speaker:
    """Plays audio on a background thread, so the next sentence is synthesised while this one plays."""

    def __init__(self):
        self.q = queue.Queue()
        threading.Thread(target=self._play, daemon=True).start()

    def _play(self):
        while True:
            audio = self.q.get()
            sd.play(audio, SR_OUT)
            sd.wait()
            self.q.task_done()

    def say(self, text):
        for _, _, audio in tts(text, voice=VOICE):
            a = audio.cpu().numpy() if hasattr(audio, "cpu") else np.asarray(audio)
            self.q.put(a.astype(np.float32))

    def alarm(self):
        t = np.linspace(0, 0.25, int(SR_OUT * 0.25), endpoint=False)
        beep = (0.4 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)
        gap = np.zeros(int(SR_OUT * 0.1), np.float32)
        self.q.put(np.concatenate([beep, gap] * 3))

    def wait(self):
        self.q.join()


speaker = Speaker()


def record_ptt():
    chunks = []

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=SR_IN, channels=1, dtype="float32", callback=cb):
        input("  Recording... press Enter to stop.")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.float32)


def transcribe(audio):
    segments, _ = stt.transcribe(audio, language="en", beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def handle_emergency(crew, kind, phrase, transcript):
    """Plain code only: no language model on this path."""
    print(f"\n  !!! EMERGENCY ({kind}): heard '{phrase}'. Bypassing the AI.")
    run_tool("raise_emergency", {"crew_id": crew, "trigger_phrase": phrase, "transcript": transcript})
    speaker.alarm()
    speaker.say(f"Emergency alarm raised for {crew.title()}. The crew has been alerted.")
    card = run_tool("get_protocol", {"protocol_id": kind})
    if isinstance(card, dict) and card.get("steps"):
        speaker.say(card.get("title", ""))
        for step in card["steps"]:
            speaker.say(step)


def main():
    crew, history = CREW[0], []
    print("Ready. Enter = start/stop recording. Type a crew id to switch speaker, q to quit.")
    while True:
        cmd = input(f"\n[{crew}] Enter to talk > ").strip().lower()
        if cmd == "q":
            break
        if cmd in CREW:
            crew, history = cmd, []
            print(f"  Now speaking as {crew}.")
            continue

        audio = record_ptt()
        t0 = time.time()
        text = transcribe(audio)
        t_stt = time.time() - t0
        print(f"  You: {text}   (speech-to-text {t_stt:.1f} s)")
        if not text:
            continue

        hit = safety.check_emergency(text)
        if hit:
            handle_emergency(crew, hit[0], hit[1], text)
            speaker.wait()
            continue

        system, facts = llm.build_system_prompt(crew, run_tool)
        messages = ([{"role": "system", "content": system}]
                    + history[-2 * HISTORY_TURNS:]
                    + [{"role": "user", "content": text}])
        t1 = time.time()
        reply, results = llm.run_turn(messages, run_tool, crew, text)
        t_llm = time.time() - t1

        allowed = safety.allowed_numbers(facts, text, *results)
        spoken, t_first = [], None
        for sentence in safety.split_sentences(reply):
            ok, why = safety.guard_sentence(sentence, allowed)
            if not ok:
                print(f"  [guard] dropped: {sentence!r} ({why})")
                continue
            speaker.say(sentence)
            spoken.append(sentence)
            if t_first is None:
                t_first = time.time() - t0
        if not spoken:
            speaker.say(FALLBACK)
            spoken = [FALLBACK]
            t_first = time.time() - t0

        print(f"  Companion: {' '.join(spoken)}")
        print(f"  timings: speech-to-text {t_stt:.1f} s | model {t_llm:.1f} s | "
              f"first audio {t_first:.1f} s after you stopped")
        history += [{"role": "user", "content": text}, {"role": "assistant", "content": " ".join(spoken)}]
        speaker.wait()


if __name__ == "__main__":
    main()
