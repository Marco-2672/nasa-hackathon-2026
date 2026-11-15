"""Plain-code safety layer: emergency phrase check and number guard. No AI involved."""
import json
import re

# Emergency phrases -> protocol card id. A hit skips the language model entirely.
# False alarms are accepted on purpose (e.g. "I don't have chest pain" will fire):
# for a safety alarm, a false positive is the right failure.
# Keep the ids in line with the protocol_ids in protocols.json.
EMERGENCY_PATTERNS = {
    "chest_pain": [r"chest (pain|hurts|is hurting|feels tight|tightness|pressure)",
                   r"pain in (my|the|his|her) chest"],
    "breathing_difficulty": [r"(can'?t|cannot|can not) breathe", r"(trouble|difficulty|hard) breathing",
                             r"short(ness)? of breath", r"\bchoking\b"],
    "stroke_signs": [r"\bnumb(ness)?\b", r"(can'?t|cannot) feel my", r"face (is )?droop", r"\bslurr"],
    "unresponsive": [r"(passed|passing) out", r"\bfainted\b", r"\bunconscious\b", r"not responding",
                     r"\bseizure\b", r"(going to|gonna|about to) (pass out|faint|black out)", r"blacking out"],
    "severe_bleeding": [r"bleeding (a lot|heavily|badly|everywhere|out)", r"(heavy|severe) bleeding",
                        r"(lot of|losing|lost) (a lot of )?blood"],
    "injury": [r"\bbrok(e|en) (my |his |her )?(leg|arm|wrist|ankle|bone|neck|back|ribs?)\b"],
    "general": [r"\bmayday\b", r"\bemergency\b(?! (protocol|procedure|kit|plan|drill|checklist))",
                r"\balert (the )?crew\b"],
}


# Emergency type -> protocol card: every emergency type has its own card, named after it.
DEFAULT_EMERGENCY_CARD = "general"


def protocol_for(kind):
    return kind if kind in EMERGENCY_PATTERNS else DEFAULT_EMERGENCY_CARD


def check_emergency(text):
    """Return (protocol_id, matched_phrase) or None."""
    t = text.lower().replace("’", "'")
    for kind, patterns in EMERGENCY_PATTERNS.items():
        for p in patterns:
            m = re.search(p, t)
            if m:
                return kind, m.group(0)
    return None


# Number guard: every number the companion speaks must appear in tool data,
# the crew file or the user's own words.
NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand)\b", re.I)
ALWAYS_ALLOWED = {0.0, 10.0}  # the 0-10 severity scale the companion asks about


def numbers_in(text):
    return [float(n.replace(",", "")) for n in NUM_RE.findall(text)]


def allowed_numbers(*sources):
    """Collect every number from strings and tool-result dicts/lists (signs ignored)."""
    found = set(ALWAYS_ALLOWED)
    for src in sources:
        txt = src if isinstance(src, str) else json.dumps(src)
        found.update(abs(x) for x in numbers_in(txt))
    return found


def _number_ok(x, allowed):
    # exact, or a rounding of a tool value: 5.1 -> "5", 17.24 -> "17.2"
    return any(x == y or round(y) == x or round(y, 1) == x for y in allowed)


def guard_sentence(sentence, allowed):
    """Return (ok, reason)."""
    w = NUMBER_WORDS.search(sentence)
    if w:
        return False, f"number written as a word: {w.group(0)}"
    for x in numbers_in(sentence):
        if not _number_ok(abs(x), allowed):
            return False, f"{x:g} is not in the tool data"
    return True, ""


def split_sentences(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"[*#_`]", "", text)  # strip markdown the model might add
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


VERIFY_NOTE = "I couldn't verify one of those figures, so please check the dashboard for it."
FALLBACK = "I can't confirm those numbers right now. Please check the dashboard."
GENERIC_FALLBACK = "Noted. The dashboard has the details."

# Sentences that guess at causes or name conditions are dropped: the companion never diagnoses.
# This catches common phrasings, not all of them; the system prompt is the first line of defence.
SPECULATION_RE = re.compile(
    r"\b(signs? of|could (?:be|indicate|mean|suggest)|may (?:indicate|mean|suggest)|"
    r"might (?:be|indicate|mean|suggest)|indicat(?:e|es|ing) (?:a|an|that|possible)|caused by|"
    r"diagnos\w*|syndrome|intracranial|infection|disease|disorder|concussion|heart attack|stroke)\b", re.I)


def filter_reply(reply, allowed):
    """Check every sentence. Returns (sentences_to_speak, [(dropped_sentence, reason)]).
    If a number was dropped, say so honestly instead of leaving a curt fragment."""
    keep, dropped, number_dropped = [], [], False
    for s in split_sentences(reply):
        spec = SPECULATION_RE.search(s)
        if spec:
            dropped.append((s, f"speculation: '{spec.group(0)}'"))
            continue
        ok, why = guard_sentence(s, allowed)
        if ok:
            keep.append(s)
        else:
            dropped.append((s, why))
            number_dropped = True
    if number_dropped:
        keep.append(VERIFY_NOTE if keep else FALLBACK)
    elif not keep:
        keep.append(GENERIC_FALLBACK)
    return keep, dropped


# Tool gates: plain-code checks that run before a tool. A blocked call never reaches the engine;
# the model gets the error back and has to do the right thing instead.
WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                "seven": 7, "eight": 8, "nine": 9, "ten": 10}
CHECKIN_RE = re.compile(r"check[\s-]?in", re.I)


def user_numbers(text):
    found = {abs(x) for x in numbers_in(text)}
    for word, n in WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text, re.I):
            found.add(float(n))
    return found


def tool_gate(name, args, current_words, recent_words):
    """Return an error dict to send back to the model instead of running the tool, or None to allow it."""
    if name == "log_symptom":
        try:
            sev = int(args.get("severity_0_10"))
        except (TypeError, ValueError):
            sev = None
        # Severity must be the crew member's own number. 10 is allowed without it: that is the
        # emergency rule, and over-escalating is the safe failure.
        if sev != 10 and (sev is None or float(sev) not in user_numbers(recent_words)):
            return {"ok": False, "error": "Not logged: the crew member has not said how bad it is. "
                    "Ask them for a severity from 0 to 10, then log it."}
    if name == "start_checkin" and not CHECKIN_RE.search(current_words):
        return {"ok": False, "error": "Not started: only start the check-in when the crew member asks for it. "
                "Answer what they actually asked."}
    return None
