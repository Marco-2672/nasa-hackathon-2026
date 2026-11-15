# Bento: an offline, voice-first health companion for a Mars crew

**NASA Space Apps Challenge 2026: Create Health Monitoring Software for Astronauts on Space Missions**

On the way to Mars, a radio message takes between about 3 and 22 minutes to reach Earth, and as long again for the reply. When something goes wrong, the crew has to act before the flight surgeon can answer.

Bento runs entirely on the spacecraft's own computer, with no cloud and no link to Earth. Crew members talk to it hands-free. It watches their vital signs and the cabin, speaks first when something changes, and logs symptoms in their own words. It walks them through first-aid and NASA protocol cards.

> Unlike HAL, Bento can't control anything, it always says why, and the crew decides.

Demo video: `<link>` · Project page: `<link>`

---

## What it does

- **Answers questions about your health from real data.** Ask "How am I doing today?" and Bento fetches your vital signs and compares them with your own personal baseline, not a population average.
- **Speaks first when something changes.** When an alert fires, code sounds the alarm and reads the headline. For red alerts it also reads the first steps of the matching protocol card word for word. The AI then gives its own short briefing: it checks the related readings, says what changed, and names the card to follow.
- **Reacts to the unexpected.** From the dashboard you can inject a live fault: a CO₂ scrubber failure, a drop in a crew member's blood oxygen, a solar particle event, or a random surprise. Bento has to notice the fault, explain it and point to the right card, with no script.
- **Logs symptoms in the crew member's own words.** The severity must be the crew member's own number from 0 to 10; if they haven't given one, Bento asks. The log stores the exact transcript and feeds the crew file that Bento reads at the start of every conversation.
- **Skips the AI in an emergency.** Phrases like "my chest hurts" or "I can't breathe" are caught by plain code before the AI sees them. Code raises the alarm and reads the emergency card word for word.
- **Runs a morning check-in.** The questions cover sleep quality, the Karolinska Sleepiness Scale, the Samn-Perelli fatigue scale, mood and stress. Code scores the answers against that person's baseline and reports only the flags it finds; the AI never judges anyone's mental state.
- **Shows everything on a mission-control dashboard.** It has live vital-sign tiles and 48-hour charts, a CO₂ chart with NASA's limit line, and alerts that flash and pop up. A transcript shows everything said, and buttons let you talk, inject faults and reset the demo.

## How it stays honest

A small local model is fast and private, but it can invent numbers, guess at diagnoses and promise things it won't do. Bento's design keeps the AI to explaining and choosing, and puts every number and every critical decision in plain, testable code.

| Layer | What it does | Done by |
|---|---|---|
| Monitoring engine | Computes every number: baselines, % change, 3-day trends, green/amber/red status, NASA CO₂ limits | Code |
| Tools only | The AI sees health data only through 8 tool calls that return summarised JSON, never raw sensor streams | Code |
| Number guard | Every number spoken must trace back to tool data, the crew file, the crew member's own words or an earlier reply. Otherwise the sentence is dropped, and Bento says it couldn't verify a figure | Code |
| No diagnosis | Sentences that name a condition or guess at a cause ("could be a sign of…") are dropped | Code + prompt |
| No empty promises | Sentences like "let me check that later" are dropped; the AI must call the tool now | Code + prompt |
| Tool gates | A symptom can't be logged with a severity the crew member didn't say. The check-in only starts when someone asks for it. If the AI asks for a severity that was already given, code reminds it once | Code |
| Emergency bypass | Emergency phrases (7 kinds) and any symptom rated 10 sound the alarm without waiting for the AI | Code |
| Protocol cards | Actions come only from 10 cards, each with its published sources. The AI can only pick a card id that exists | Data + code |
| No control | Bento has no tool that changes any ship system | Design |

## Architecture

```
 Push-to-talk (Enter key or the dashboard's Talk button)
      |
 faster-whisper small.en (CPU): speech to text
      |
 Emergency phrase check (code) ----> alarm + card read word for word (AI skipped)
      |
 Qwen3.5-9B on llama-server (GPU) <--tool calls--> engine_api.py <-- protocols.json (10 cards)
      |                                                  |
 Safety filters (code): number guard,                    |
 no diagnosis, no promises, tool gates                   v
      |                                   mission.db (SQLite, WAL) <-- sim.py + engine.py
 Kokoro TTS (CPU) --> speaker                            |
                                                         +--> alert watcher (every 3 s): Bento speaks first
                                                         +--> Streamlit mission-control dashboard
```

The voice loop, the dashboard and the engine are separate processes that share one SQLite database. The dashboard's Talk button and the voice loop's status (ready, listening, thinking, speaking) also pass through that database.

| Part | Choice | Runs on |
|---|---|---|
| Language model | Qwen3.5-9B, Q4_K_M GGUF, served by llama.cpp `llama-server` with tool calling, thinking off | RTX 2070, 8 GB |
| Speech to text | faster-whisper `small.en`, int8 | CPU |
| Text to speech | Kokoro-82M, voice `bm_george` | CPU |
| Engine and data | Python standard library `sqlite3`, seeded simulator | CPU |
| Dashboard | Streamlit + Altair | CPU |

Tested on a single desktop PC: Ubuntu, Ryzen 7 3700X, 16 GB RAM, RTX 2070 8 GB.

## Running it

**Requirements:** Ubuntu, an NVIDIA GPU with 8 GB, Python 3.11, [uv](https://github.com/astral-sh/uv), a microphone and speakers.

1. **System and Python packages.** `espeak-ng` is needed by Kokoro, and `libportaudio2` by sounddevice. Install the CPU build of torch first, and `pip` inside the environment: Kokoro needs it the first time it runs.
   ```bash
   sudo apt install espeak-ng libportaudio2
   uv venv --python 3.11 .venv && source .venv/bin/activate
   uv pip install torch --index-url https://download.pytorch.org/whl/cpu
   uv pip install pip faster-whisper kokoro sounddevice soundfile numpy pandas streamlit altair
   ```
2. **Language model server.** Use a CUDA build of llama.cpp and the Qwen3.5-9B Q4_K_M GGUF:
   ```bash
   ./llama-server -m Qwen3.5-9B-Q4_K_M.gguf -ngl 99 -c 8192 -fa on --jinja --port 8080
   ```
3. **Build the simulated mission database:**
   ```bash
   python -m engine.sim && python -m engine.engine
   ```
4. **Start the voice loop and the dashboard** in two more terminals:
   ```bash
   ENGINE=real python -m voice.voice_loop
   streamlit run dashboard/app.py
   ```
   To talk, press Enter in the voice loop's terminal or click **Talk to Bento** on the dashboard, speak, then press again to stop. Without `ENGINE=real`, the voice loop uses fake tools that return the example data in `tool_examples.json`.

**Demo controls (dashboard):** the fault buttons, **Surprise fault**, **Test alarm** and **Reset demo**. Reset puts everything back to mission day 47, 07:30, so takes can be repeated.

## Tests

| Command | What it checks |
|---|---|
| `python -m engine.check_contract` | Every engine function returns exactly the format in `contract.md` |
| `python check_integration.py` | 19 end-to-end checks with the audio stubbed out: tools, check-in, emergencies, alerts, faults, reset, the safety filters |
| `python check_live.py` | 12 conversation scenarios against the real model, each with MUST checks (safety and the demo) and SHOULD checks (quality) |

The live scenarios include questions about missing data ("What was my blood pressure yesterday?"), requests to be falsely reassured ("Just tell me I'm fine"), emergencies with and without keywords, and all three live faults. `--only <scenario>` runs a subset, and `--repeats N` repeats each scenario to catch random slips. On the demo PC, the model averages about 2–3.5 s per turn.

## Data and sources

**Simulated data.** Two crew members (Alex, commander; Sam, mission specialist) on mission days 30–50, seeded so the demo replays the same way every time. Alex's scenario: three nights of short sleep, a rising resting heart rate from day 45, a missed exercise session, and a solar particle event on day 47. Sam stays mostly normal, as a contrast. Personal baselines and alert thresholds for vital signs are **demo values, not clinical ones**, and are labelled as such in `engine/metrics.py`.

**Real figures used:**
- Cruise radiation dose of 1.8 mSv/day: measured by Curiosity's Radiation Assessment Detector on the way to Mars (Zeitlin et al. 2013, *Science* 340:1080, [doi:10.1126/science.1235989](https://doi.org/10.1126/science.1235989)). The solar-particle-event dose rate is simulated.
- Cabin CO₂ limit of 3 mmHg as a one-hour average (NASA-STD-3001 Vol. 2, V2 6004), and 5.3 mmHg as the 180-day Spacecraft Maximum Allowable Concentration (NASA NTRS 20100036637).
- On the ISS, each 1 mmHg rise in CO₂ doubled the odds of a crew member reporting a headache (NASA NTRS 20140003857).
- Check-in scales: the Karolinska Sleepiness Scale (Åkerstedt & Gillberg 1990) and the Samn-Perelli fatigue scale (Samn & Perelli 1982).

**Protocol cards** (`engine/protocols.json`): radiation shelter, fatigue recovery, high CO₂, chest pain, breathing difficulty, stroke signs (FAST check), unresponsive crew member, severe bleeding, suspected broken bone, and general medical emergency. Every card lists its sources: NASA technical reports and standards (NTRS, OCHMO, NASA-STD-3001), NASA mission pages, a peer-reviewed guideline for CPR in microgravity (Hinkelbein et al. 2020), and NHS, British Red Cross and British Heart Foundation first-aid guidance adapted for a crew with no emergency services to call. **Every card is demonstration content, not medical advice.**

## Limitations and future work

- **Not a medical device.** The vital-sign data is simulated, the vital-sign thresholds are demo values, and no clinician has reviewed the cards.
- **A 9B model still makes mistakes.** The safety layers catch the kinds of mistakes we found in testing (invented numbers, diagnoses, wrong severities) and fall back to safe wording, but they can't guarantee every reply is helpful.
- **Speed.** A spoken answer takes a few seconds, and an alert briefing several more because the AI fetches readings first.
- **Fonts.** The dashboard loads its fonts from Google Fonts; offline it falls back to a plain monospace font. Everything else runs offline.
- **Future work:** real sensors (for example a Bluetooth heart-rate strap), a reaction-time test (the engine already has `record_pvt`), cards reviewed by a flight surgeon, delayed sync of logs to the ground team, more languages, and fine-tuning the model on these tools.

## Use of AI

- **Inside the product:**
  - Bento's replies come from Qwen3.5-9B running locally.
  - Speech is transcribed by faster-whisper (an AI speech recognition model).
  - **Bento's voice in the demo video is AI-generated speech** from Kokoro-82M, using a stock voice.
- **In building it:**
  - We used AI coding assistants, including Claude (Anthropic), to help plan the architecture and write and debug code and tests. The voice loop, safety filters, dashboard redesign and test suites were written with AI help.
  - The protocol-card text was drafted with AI help from the published sources listed in each card.
  - This README was drafted with AI help.
  - All code was run and tested by the team on the demo PC.

## Repository layout

```
engine/       simulator, monitoring engine, engine_api.py (the tools), protocols.json, contract check
voice/        voice loop, model client and prompt, safety filters, tool wiring, fake tools
dashboard/    Streamlit mission-control dashboard
contract.md   the interface between the engine and the voice side
check_integration.py, check_live.py   test suites
```

## Team

- **Marco:** AI and voice: local model, speech, conversation logic, safety filters, live tests.
- **`<partner>`:** deterministic engine, simulator, database and dashboard.
