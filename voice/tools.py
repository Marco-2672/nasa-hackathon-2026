"""One switch between fake tools and the real engine.

Fake (default):  python -m voice.voice_loop
Real engine:     ENGINE=real python -m voice.voice_loop
"""
import inspect
import os

if os.environ.get("ENGINE") == "real":
    from engine import engine_api

    def run_tool(name, args):
        fn = getattr(engine_api, name, None)
        if fn is None:
            return {"error": f"unknown tool {name}"}
        args = dict(args)
        if isinstance(args.get("crew_id"), str):
            args["crew_id"] = args["crew_id"].strip().lower()  # models write "Alex"; the engine uses "alex"
        # Drop arguments the engine function doesn't take (models add extras like crew_id on get_environment).
        params = inspect.signature(fn).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            args = {k: v for k, v in args.items() if k in params}
        # Models sometimes send numbers and booleans as text ("7", "true"): convert to what the engine expects.
        for k, v in list(args.items()):
            ann = str(params[k].annotation) if k in params else ""
            if isinstance(v, str) and ann.split("|")[0].strip() in ("int", "<class 'int'>") and v.strip().lstrip("-").isdigit():
                args[k] = int(v)
            elif isinstance(v, str) and "bool" in ann and v.strip().lower() in ("true", "false"):
                args[k] = v.strip().lower() == "true"
        try:
            return fn(**args)
        except Exception as e:  # the engine should not raise, but never crash the voice loop
            return {"error": f"{type(e).__name__}: {e}"}
else:
    from voice.fake_tools import run_tool  # noqa: F401
