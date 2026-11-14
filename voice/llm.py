"""Talks to llama-server: system prompt with the crew file, and the tool-calling loop."""
import json
import re
import urllib.request

URL = "http://localhost:8080/v1/chat/completions"
NAME = "Bento"  # working name; not HAL
# Match these to the settings llm_eval.py used, since that is what you tested.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}

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
        "name": "get_protocol", "description": "Fetch a protocol card. Read its steps as written; do not add steps.",
        "parameters": {"type": "object", "properties": {"protocol_id": {"type": "string"}}, "required": ["protocol_id"]}}},
    {"type": "function", "function": {
        "name": "get_crew_log", "description": "Recent entries from a crew member's medical log (symptoms, check-ins).",
        "parameters": {"type": "object", "properties": {
            "crew_id": {"type": "string"}, "days": {"type": "integer"}}, "required": ["crew_id"]}}},
]
CREW_TOOLS = {t["function"]["name"] for t in TOOLS
              if "crew_id" in t["function"]["parameters"]["properties"]}

SYSTEM_TEMPLATE = """You are {name}, the on-board health companion for a deep-space crew. You are speaking with {crew_name} (crew id: {crew_id}). Earth is minutes away by radio, so you help the crew understand their own health data and decide what to do.

Rules:
- You are speaking out loud. Reply in one to three short sentences of plain speech. No lists, no markdown, no emoji.
- Get every number from a tool or the crew file. Never estimate, calculate or remember a number. Always write numbers as digits, never as words.
- Always say why: name the reading and how it compares with this person's own baseline.
- Never diagnose and never name a medical condition. Describe what the data shows.
- Only suggest actions from the protocol cards listed below. To give guidance, call get_protocol and read its steps as written. Do not add steps of your own.
- When someone reports a symptom, call log_symptom. If they have not said how bad it is, first ask for a severity from 0 to 10.
- For the morning check-in, call start_checkin, ask the questions one at a time exactly as written, then call submit_checkin and report only the flags it returns.
- You cannot control any ship system. The crew decides.

Crew file:
{summary}

Protocol cards available:
{protocols}"""


def build_system_prompt(crew_id, run_tool):
    """Return (system_prompt, facts). facts = crew-file text the number guard may trust."""
    profile = run_tool("get_crew_profile", {"crew_id": crew_id})
    summary = profile.get("summary_text", "No crew file available.") if isinstance(profile, dict) else ""
    cards = run_tool("list_protocols", {})
    if isinstance(cards, list):
        protocols = "\n".join(f"- {c.get('protocol_id')}: {c.get('title')}" for c in cards)
    else:
        protocols = "None available."
    crew_name = profile.get("name", crew_id.title()) if isinstance(profile, dict) else crew_id.title()
    prompt = SYSTEM_TEMPLATE.format(name=NAME, crew_name=crew_name, crew_id=crew_id,
                                    summary=summary, protocols=protocols)
    return prompt, summary + "\n" + protocols


def chat(messages, tools=None, max_tokens=300):
    body = {"messages": messages, "max_tokens": max_tokens, **SAMPLING}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"]


def run_turn(messages, run_tool, crew_id, transcript, max_rounds=4):
    """Tool loop. Returns (reply_text, tool_results). Mutates messages."""
    results = []
    for _ in range(max_rounds):
        msg = chat(messages, TOOLS)
        calls = msg.get("tool_calls") or []
        if not calls:
            return re.sub(r"<think>.*?</think>", "", msg.get("content") or "", flags=re.S).strip(), results
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            if name in CREW_TOOLS and name != "get_alerts":
                args.setdefault("crew_id", crew_id)
            if name == "log_symptom":
                args["transcript"] = transcript  # code stores the verbatim words, not the model's paraphrase
            result = run_tool(name, args)
            print(f"  [tool] {name}({args}) -> {json.dumps(result)[:120]}")
            results.append(result)
            messages.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps(result)})
    msg = chat(messages)  # out of rounds: answer without more tools
    return (msg.get("content") or "").strip(), results
