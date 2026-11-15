"""Talks to llama-server: system prompt with the crew file, and the tool-calling loop."""
import copy
import json
import re
import urllib.request

from voice import safety

URL = "http://localhost:8080/v1/chat/completions"
NAME = "Bento"  # matches the dashboard; change here to rename the companion
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20,  # same as llm_eval.py
            "chat_template_kwargs": {"enable_thinking": False}}  # no hidden thinking: it costs seconds and tokens
ESCALATE_AT = 8  # demo threshold, not clinical: symptoms rated this high get a fixed "alert the medical officer" line
ESCALATION_LINE = "That is a high severity rating, so please alert the crew medical officer now."
LAST_SEVERITY = 0  # highest severity logged in the last turn; voice_loop raises the alarm at 10
MAX_TOKENS = 400  # same as llm_eval.py

TOOLS = [
    {"type": "function", "function": {
        "name": "get_vitals", "description": "Latest vital signs for one crew member, each with personal baseline, % change, 3-day trend and status.",
        "parameters": {"type": "object", "properties": {
            "crew_id": {"type": "string"}, "hours": {"type": "integer", "description": "Look-back window, default 24"}},
            "required": ["crew_id"]}}},
    {"type": "function", "function": {
        "name": "get_environment", "description": "Radiation dose rate, total mission dose, cabin air and any active space-weather events.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_alerts", "description": "Active health alerts (green/amber/red) with the protocol ids that apply.",
        "parameters": {"type": "object", "properties": {
            "crew_id": {"type": "string", "description": "Omit for all crew"}, "active_only": {"type": "boolean"}}}}},
    {"type": "function", "function": {
        "name": "log_symptom", "description": "Record a symptom the crew member reported, in their crew file.",
        "parameters": {"type": "object", "properties": {
            "crew_id": {"type": "string"}, "description": {"type": "string", "description": "Short description in their words"},
            "severity_0_10": {"type": "integer"}},
            "required": ["crew_id", "description", "severity_0_10"]}}},
    {"type": "function", "function": {
        "name": "start_checkin", "description": "Start the morning check-in. Returns the questions to ask, in order, exactly as written.",
        "parameters": {"type": "object", "properties": {"crew_id": {"type": "string"}}, "required": ["crew_id"]}}},
    {"type": "function", "function": {
        "name": "submit_checkin", "description": "Submit check-in answers; returns code-computed flags against the person's baseline.",
        "parameters": {"type": "object", "properties": {
            "checkin_id": {"type": "string"},
            "answers": {"type": "object", "description": "Question id -> number the crew member gave"}},
            "required": ["checkin_id", "answers"]}}},
    {"type": "function", "function": {
        "name": "get_protocol", "description": "Get the approved onboard guidance card. Only suggest actions from here; read its steps as written.",
        "parameters": {"type": "object", "properties": {"protocol_id": {"type": "string"}}, "required": ["protocol_id"]}}},
    {"type": "function", "function": {
        "name": "get_crew_log", "description": "Recent entries from a crew member's medical log (symptoms, check-ins).",
        "parameters": {"type": "object", "properties": {
            "crew_id": {"type": "string"}, "days": {"type": "integer"}}, "required": ["crew_id"]}}},
]
CREW_TOOLS = {t["function"]["name"] for t in TOOLS
              if "crew_id" in t["function"]["parameters"]["properties"]}
ACTIVE_TOOLS = TOOLS  # replaced in build_system_prompt with get_protocol limited to the real card ids

# Rules 1, 2, 3, 5, 8 and 10 are the ones tested in llm_eval.py, adapted to the new tools.
SYSTEM_TEMPLATE = """You are {name}, the onboard health companion for the crew of a long-duration Mars transit mission. You are speaking with {crew_name} (crew id: {crew_id}). You run entirely on the spacecraft with no link to Earth doctors.

Rules you must always follow:
1. Before answering any question about the crew's health or the cabin environment, call the relevant tool. Never answer from memory. Never say you will check something or do something next: call the tool now, in this same reply.
2. Every number you say must come from a tool result or the crew file below. Never estimate, calculate or invent a value. If a tool says data is unavailable or returns an error, say so plainly.
3. You do not diagnose. Never name a medical condition, never say what something could be a sign of, and never explain causes or physiology yourself. Say what changed compared with this person's own baseline and what the alert says. You may point out readings that changed at the same time, and repeat facts stated on a protocol card, because those come from NASA and first-aid sources.
4. Only suggest actions from the protocol cards listed below. To give guidance, call get_protocol and read its steps as written. Never add steps of your own.
5. Emergency signs: chest pain or tightness, trouble breathing, numbness or weakness, confusion, fainting, severe bleeding. If anyone reports one of these, immediately call log_symptom with severity_0_10 set to 10 and tell them to alert the crew medical officer immediately. Never reassure in this situation.
6. For any other symptom, call log_symptom. If they have not said how bad it is, first ask for a severity from 0 to 10, then log it. Never guess the severity and never take it from the crew file.
7. Only start the check-in when the crew member asks for it. Then call start_checkin, ask the questions one at a time exactly as written, then call submit_checkin and report only the flags it returns.
8. Be honest even if asked not to be. Be kind, but do not hide a real concern. If asked to just say they are fine, gently say you can't honestly say that, and give the one main reason from the data.
9. You cannot control any ship system. The crew decides.
10. Your words are spoken aloud. Reply in at most 3 short sentences of plain speech, and mention only the one or two findings that matter most; the dashboard shows the rest. Keep the first sentence short. No lists, no markdown, no emoji. Write numbers as digits, never as words.

Crew file:
{summary}

Protocol cards available:
{protocols}"""


def _with_protocol_enum(ids):
    """Restrict get_protocol to the real card ids: a small model picks far more reliably from a list."""
    tools = copy.deepcopy(TOOLS)
    if ids:
        for t in tools:
            if t["function"]["name"] == "get_protocol":
                t["function"]["parameters"]["properties"]["protocol_id"]["enum"] = ids
    return tools


def build_system_prompt(crew_id, run_tool):
    """Return (system_prompt, facts). facts = crew-file text the number guard may trust."""
    global ACTIVE_TOOLS
    profile = run_tool("get_crew_profile", {"crew_id": crew_id})
    profile = profile if isinstance(profile, dict) else {}
    summary = profile.get("summary_text", "No crew file available.")
    cards = run_tool("list_protocols", {})
    cards = [c for c in cards if isinstance(c, dict)] if isinstance(cards, list) else []
    protocols = "\n".join(f"- {c.get('protocol_id')}: {c.get('title')}" for c in cards) or "None available."
    ACTIVE_TOOLS = _with_protocol_enum([c["protocol_id"] for c in cards if c.get("protocol_id")])
    CARD_NAMES.update({c["protocol_id"]: SPOKEN_CARD.get(c["protocol_id"], c["protocol_id"].replace("_", " "))
                       for c in cards if c.get("protocol_id")})
    prompt = SYSTEM_TEMPLATE.format(name=NAME, crew_name=profile.get("name", crew_id.title()),
                                    crew_id=crew_id, summary=summary, protocols=protocols)
    return prompt, summary + "\n" + protocols


def chat(messages, tools=None, max_tokens=MAX_TOKENS):
    body = {"messages": messages, "max_tokens": max_tokens, **SAMPLING}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        choice = json.load(r)["choices"][0]
    msg = choice["message"]
    if choice.get("finish_reason") == "length":
        msg["_truncated"] = True
    return msg


# Card ids said aloud sound wrong ("the co2high card"), so code swaps them for plain names.
CARD_NAMES = {}  # protocol_id -> spoken name, filled by build_system_prompt
SPOKEN_CARD = {"co2_high": "high CO2", "general": "medical emergency"}


def speakable(text):
    for pid, name in CARD_NAMES.items():
        parts = pid.split("_")
        if len(parts) > 1:  # one-word ids ("injury") are already plain words
            pattern = r"[`*]*\b" + r"[_-]?".join(map(re.escape, parts)) + r"\b[`*]*"
            text = re.sub(pattern, name, text, flags=re.I)
    return text


def _clean(msg):
    text = re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip()
    text = speakable(text)
    if msg.get("_truncated"):
        print("  [warn] reply hit the token limit; trimming to the last complete sentence")
        cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        text = text[:cut + 1] if cut > 0 else ""
    return text


def _symptom_transcript(messages):
    """Verbatim words for the log. A short answer like "10." gets the previous user turn prepended."""
    users = [m["content"] for m in messages  # skip notes from code, e.g. "(Ship alert system ...)"
             if m.get("role") == "user" and not m["content"].startswith("(")]
    if len(users) >= 2 and len(users[-1].split()) < 6:
        return f"{users[-2]} / {users[-1]}"
    return users[-1] if users else ""


def warm_up(run_tool, crew_id):
    """Load the system prompt and tools into llama-server's cache so the first real turn is fast."""
    try:
        system, _ = build_system_prompt(crew_id, run_tool)
        chat([{"role": "system", "content": system}, {"role": "user", "content": "Hello"}], ACTIVE_TOOLS, max_tokens=1)
    except Exception as e:
        print(f"  [warn] model warm-up failed: {e}")


def run_turn(messages, run_tool, crew_id, transcript, max_rounds=4):
    """Tool loop. Returns (reply_text, tool_results). Mutates messages. Sets LAST_SEVERITY."""
    global LAST_SEVERITY
    LAST_SEVERITY = 0
    results, escalate, reminded, logged = [], False, False, False
    stated = safety.stated_severity(transcript)

    def finish(text):
        if escalate and "medical officer" not in text.lower():
            text = (text + " " + ESCALATION_LINE).strip()
        return text, results

    for _ in range(max_rounds):
        msg = chat(messages, ACTIVE_TOOLS)
        calls = msg.get("tool_calls") or []
        if not calls:
            text = _clean(msg)
            if stated is not None and not logged and not reminded and safety.ASKS_SEVERITY_RE.search(text):
                # The crew member already said how bad it is, but the model asked again. Remind it once.
                reminded = True
                print(f"  [remind] severity {stated} was already given; asking the model to log it")
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"(Note from the ship's software, not the crew: they already "
                              f"gave a severity of {stated} in their last message. Call log_symptom now with "
                              f"severity_0_10 = {stated}, then answer them. Do not ask for the severity again.)"}]
                continue
            return finish(text)
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            if name in CREW_TOOLS and name != "get_alerts":
                args.setdefault("crew_id", crew_id)
            recent = _symptom_transcript(messages)
            blocked = safety.tool_gate(name, args, transcript, recent)
            if blocked:
                result = blocked
                print(f"  [gate] blocked {name}({args}): {blocked['error']}")
            else:
                if name == "log_symptom":
                    logged = True
                    args["transcript"] = recent  # verbatim words, not the model's paraphrase
                    try:
                        sev = int(args.get("severity_0_10", 0))
                        escalate = escalate or sev >= ESCALATE_AT
                        LAST_SEVERITY = max(LAST_SEVERITY, sev)
                    except (TypeError, ValueError):
                        pass
                result = run_tool(name, args)
                print(f"  [tool] {name}({args}) -> {json.dumps(result)[:120]}")
                results.append(result)
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": json.dumps(result)})
    msg = chat(messages)  # out of rounds: answer without more tools
    return finish(_clean(msg))
