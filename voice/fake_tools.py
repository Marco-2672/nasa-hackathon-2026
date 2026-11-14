import json, pathlib
EX = json.loads((pathlib.Path(__file__).parents[1] / "tool_examples.json").read_text())

def run_tool(name, args):
    return EX.get(name, {"error": f"unknown tool {name}"})
