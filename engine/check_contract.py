# compares engine_api output against tool_examples.json key for key
# runs on a temp copy of mission.db so mutating calls don't touch real data
# usage: python -m engine.check_contract

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from engine import engine_api as api
from engine.engine import derive_alerts

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "tool_examples.json"


def _compare(example, actual, path=""):
    # return list of differences, dict keys and value types must match
    diffs = []
    # allow int/float interchange (0 vs 0.0), everything else must be exact
    num = (int, float)
    same_kind = (
        type(example) is type(actual)
        or (isinstance(example, num) and isinstance(actual, num))
    )
    if not same_kind:
        return [f"{path}: type {type(actual).__name__} != {type(example).__name__}"]

    if isinstance(example, dict):
        only_exp = set(example) - set(actual)
        only_act = set(actual) - set(example)
        if only_exp:
            diffs.append(f"{path}: missing keys {sorted(only_exp)}")
        if only_act:
            diffs.append(f"{path}: extra keys {sorted(only_act)}")
        for k in example:
            if k in actual:
                diffs.extend(_compare(example[k], actual[k], f"{path}.{k}"))

    elif isinstance(example, list):
        # only check structure of the first element, length mismatch is a warning
        if example and actual:
            diffs.extend(_compare(example[0], actual[0], f"{path}[0]"))
        if len(actual) != len(example):
            print(f"warn {path}: list len {len(actual)} != {len(example)} (structure only)")

    return diffs


def main() -> int:
    examples = json.loads(EXAMPLES.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        test_db = Path(tmp) / "mission.db"
        shutil.copy(api.DEFAULT_DB, test_db)
        api.DEFAULT_DB = test_db  # type: ignore
        derive_alerts(test_db)

        checks = [
            ("get_vitals",       lambda: api.get_vitals("alex")),
            ("get_environment",  lambda: api.get_environment()),
            ("get_alerts",       lambda: api.get_alerts()),
            ("log_symptom",      lambda: api.log_symptom("alex", "test", 3, "test")),
            ("start_checkin",    lambda: api.start_checkin("alex")),
            ("get_protocol",     lambda: api.get_protocol("fatigue_recovery")),
            ("get_crew_log",     lambda: api.get_crew_log("alex")),
            ("get_crew_profile", lambda: api.get_crew_profile("alex")),
            ("raise_emergency",  lambda: api.raise_emergency("alex", "chest pain", "test")),
            ("list_protocols",   lambda: api.list_protocols()),
        ]

        failed = 0
        for name, call in checks:
            if name not in examples:
                print(f"skip {name}: not in tool_examples.json")
                continue
            try:
                actual = call()
            except Exception as e:
                print(f"FAIL {name}: exception: {e}")
                failed += 1
                continue
            if isinstance(actual, dict) and "error" in actual:
                print(f"FAIL {name}: returned error {actual['error']!r}")
                failed += 1
                continue
            diffs = _compare(examples[name], actual, name)
            if diffs:
                failed += 1
                print(f"FAIL {name}")
                for d in diffs:
                    print(f"  {d}")
            else:
                print(f"ok   {name}")

    print()
    print("all checks passed" if failed == 0 else f"{failed} failing")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())