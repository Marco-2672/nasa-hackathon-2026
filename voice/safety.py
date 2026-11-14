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
                     r"\bseizure\b"],
    "severe_bleeding": [r"bleeding (a lot|heavily|badly)", r"(heavy|severe) bleeding"],
    "general": [r"\bmayday\b", r"medical emergency"],
}


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
