
import os
import ast
import operator
from datetime import datetime
from pathlib import Path
import requests



# Project directory
BASE_DIR = Path.cwd()


# TOOL 1: Get current date and time
def get_current_datetime():
    return datetime.now().strftime("%A, %d %B %Y, %I:%M:%S %p")


# TOOL 2: Calculator
def calculator(expression: str):
    allowed_operators = {
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

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](evaluate(node.operand))

        raise ValueError("Invalid mathematical expression")

    try:
        tree = ast.parse(expression, mode="eval")
        return evaluate(tree.body)
    except Exception as e:
        return f"Calculation error: {e}"


# TOOL 3: Search files
def search_files(keyword: str):
    results = []

    for path in BASE_DIR.rglob("*"):
        if not path.is_file():
            continue

        if any(part in {".venv", ".git", "__pycache__"} for part in path.parts):
            continue

        if keyword.lower() in path.name.lower():
            results.append(str(path.relative_to(BASE_DIR)))

    return results[:30]


# TOOL 4: Read a text file
def read_file(filename: str):
    try:
        path = (BASE_DIR / filename).resolve()

        if not path.is_relative_to(BASE_DIR.resolve()):
            return "Access denied: file is outside the project directory."

        if not path.is_file():
            return "File not found."

        if path.stat().st_size > 100_000:
            return "File is too large to read."

        return path.read_text(encoding="utf-8")

    except Exception as e:
        return f"Error reading file: {e}"

def get_city_weather(city: str):
    url= f"https://wttr.in/{city}?format=3"
    resp= requests.get(url)

    return resp.text if resp.status_code == 200 else f"Error fetching weather for {city}: {resp.status_code}"

import requests

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

# TOOL DEFINITIONS FOR OPENAI FUNCTION CALLING
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Calculate mathematical expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression to calculate."
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Filename keyword to search."
                    }
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file inside the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Relative path of the file."
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Get the current weather for a specified city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Name of the city to get weather for."
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_city",
            "description": "Get the city of the user based on their IP address.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }
]


# TOOL EXECUTOR
def execute_tool(tool_name, arguments):

    available_functions = {
        "get_current_datetime": get_current_datetime,
        "calculator": calculator,
        "search_files": search_files,
        "read_file": read_file,
        "get_city_weather": get_city_weather,
        "get_user_city": get_user_city
    }

    if tool_name not in available_functions:
        return "Tool not found."

    try:
        return available_functions[tool_name](**arguments)
    except Exception as e:
        return f"Tool execution error: {e}"

FUNCTION_NAME_TO_FUNCTION_MAP ={
    "get_current_datetime": get_current_datetime,
    "calculator": calculator,
    "search_files": search_files,
    "read_file": read_file,
    "get_city_weather": get_city_weather,
    "get_user_city": get_user_city
}