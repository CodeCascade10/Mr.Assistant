import ast
import json
import operator
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests

from Utilities.utility import log


BASE_DIR = Path.cwd()
SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules"}


# ---------------------------------------------------------------------------
# TOOLS
# ---------------------------------------------------------------------------

def get_current_datetime():
    return datetime.now().strftime("%A, %d %B %Y, %I:%M:%S %p")


def calculator(expression: str):
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            left, right = evaluate(node.left), evaluate(node.right)
            # Guard against expressions like 9**9**9 that would hang the program
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("Exponent too large")
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](evaluate(node.operand))
        raise ValueError("Invalid mathematical expression")

    try:
        if len(expression) > 200:
            return "Calculation error: expression too long"
        return evaluate(ast.parse(expression.strip(), mode="eval").body)
    except Exception as e:
        return f"Calculation error: {e}"


def search_files(keyword: str):
    keyword = keyword.lower()
    results = []

    # os.walk lets us prune .venv/.git up front instead of walking into them
    for root, dirs, files in os.walk(BASE_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            if keyword in name.lower():
                results.append(str((Path(root) / name).relative_to(BASE_DIR)))
                if len(results) >= 30:
                    return results

    return results or "No matching files found."


def read_file(filename: str):
    try:
        path = (BASE_DIR / filename).resolve()

        if not path.is_relative_to(BASE_DIR.resolve()):
            return "Access denied: file is outside the project directory."
        if path.name.startswith(".env"):
            return "Access denied: this file contains secrets."
        if not path.is_file():
            return "File not found."
        if path.stat().st_size > 100_000:
            return "File is too large to read."

        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


def get_city_weather(city: str):
    try:
        resp = requests.get(f"https://wttr.in/{quote(city)}?format=3", timeout=10)
        if resp.status_code == 200:
            return resp.text.strip()
        return f"Error fetching weather for {city}: {resp.status_code}"
    except Exception as e:
        return f"Error fetching weather for {city}: {e}"


def get_user_city():
    for url in ("https://ipapi.co/json/", "https://ipinfo.io/json"):
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            d = r.json()
            if d.get("city"):
                return {
                    "city": d.get("city"),
                    "region": d.get("region"),
                    "country": d.get("country_name") or d.get("country"),
                }
        except Exception:
            continue
    return "Unable to detect location."


TOOL_FUNCTIONS = {
    "get_current_datetime": get_current_datetime,
    "calculator": calculator,
    "search_files": search_files,
    "read_file": read_file,
    "get_city_weather": get_city_weather,
    "get_user_city": get_user_city,
}


# ---------------------------------------------------------------------------
# TOOL DEFINITIONS (sent to the LLM)
# ---------------------------------------------------------------------------

def _tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


tools = [
    _tool("get_current_datetime", "Get the current date and time."),
    _tool(
        "calculator",
        "Calculate mathematical expressions.",
        {"expression": {"type": "string", "description": "Mathematical expression to calculate."}},
        ["expression"],
    ),
    _tool(
        "search_files",
        "Search for files inside the project directory.",
        {"keyword": {"type": "string", "description": "Filename keyword to search."}},
        ["keyword"],
    ),
    _tool(
        "read_file",
        "Read a text file inside the project directory.",
        {"filename": {"type": "string", "description": "Relative path of the file."}},
        ["filename"],
    ),
    _tool(
        "get_city_weather",
        "Get the current weather for a specified city.",
        {"city": {"type": "string", "description": "Name of the city to get weather for."}},
        ["city"],
    ),
    _tool("get_user_city", "Get the city of the user based on their IP address."),
]


# ---------------------------------------------------------------------------
# TOOL EXECUTION (everything ai_assistant.py needs)
# ---------------------------------------------------------------------------

def execute_tool(tool_name, arguments):
    func = TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return f"Tool not found: {tool_name}"
    if not isinstance(arguments, dict):
        return "Invalid tool arguments."
    try:
        return func(**arguments)
    except Exception as e:
        return f"Tool execution error: {e}"


def run_tool_call(tool_call):
    """Run one tool call from the LLM and return the result as a string."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = None

    result = execute_tool(name, args)
    if not isinstance(result, str):
        result = json.dumps(result, default=str)

    log(f"Tool called: {name} {args}")
    log(f"Tool result: {result[:200]}")
    return result



def add_tool_results(messages, msg, override_result=None):
    """Append tool requests and their execution results to conversation history."""

    messages.append({
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}"
                }
            }
            for tc in msg.tool_calls
        ]
    })

    for tc in msg.tool_calls:

        if override_result is not None:
            result = override_result
        else:
            result = run_tool_call(tc)

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result)
        })
