"""One switch between fake tools and the real engine.

Fake (default):  python -m voice.voice_loop
Real engine:     ENGINE=real python -m voice.voice_loop
"""
import os

if os.environ.get("ENGINE") == "real":
    from engine import engine_api

    def run_tool(name, args):
        fn = getattr(engine_api, name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except Exception as e:  # the engine should not raise, but never crash the voice loop
            return {"error": f"{type(e).__name__}: {e}"}
else:
    from voice.fake_tools import run_tool  # noqa: F401
