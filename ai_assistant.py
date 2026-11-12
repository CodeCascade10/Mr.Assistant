from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time

from tools import tools, execute_tool


# Load environment variables
load_dotenv()


# Request and time restrictions
MAX_REQUESTS = 10        # max user messages per session
MAX_RUNTIME = 300        # session length in seconds
MAX_TOOL_ROUNDS = 5      # max tool-call rounds per user message

request_count = 0
start_time = time.time()


# Initialize OpenAI client
client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    timeout=60.0,
    max_retries=1,
)
MODEL = os.getenv("MODEL")


# System message
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Always respond in less than 100 words. "
            "If the user asks about weather without naming a city, "
            "call get_user_city first, then get_city_weather. "
            "/no_think"
        ),
    }
]


def ask_llm(allow_tools=True):
    """Send the current conversation to the LLM and return its message."""
    kwargs = {"model": MODEL, "messages": messages}
    if allow_tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    else:
        kwargs["tool_choice"] = "none"
        kwargs["tools"] = tools  # some backends require tools when tool messages exist
    return client.chat.completions.create(**kwargs).choices[0].message


def run_tool_call(tool_call):
    """Execute a single tool call and return its result as a string."""
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        result = execute_tool(tool_call.function.name, args)
    except Exception as e:
        result = f"Tool execution error: {e}"

    print(f"  [tool] {tool_call.function.name} -> {str(result)[:80]}")
    return str(result)


print("AI Assistant Started!")


# Main conversation loop
while True:
    history_len = len(messages)  # used to roll back if a request fails

    try:
        if request_count >= MAX_REQUESTS:
            print("\nAssistant: I've reached the maximum request limit. Please restart me to continue.")
            break

        if time.time() - start_time >= MAX_RUNTIME:
            print("\nAssistant: My session has timed out. Please restart me to continue.")
            break

        user_input = input("\nuser: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("Assistant: Goodbye!")
            break

        request_count += 1
        history_len = len(messages)

        messages.append({"role": "user", "content": user_input})

        print("Sending request to LLM...")
        msg = ask_llm()

        # Tool-calling loop (capped so it can never spin forever)
        tool_rounds = 0
        while msg.tool_calls and tool_rounds < MAX_TOOL_ROUNDS:
            tool_rounds += 1
            messages.append(msg.model_dump(exclude_none=True))

            for tool_call in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": run_tool_call(tool_call),
                })

            print("Sending tool results to LLM...")
            msg = ask_llm()

        # If the cap was hit and the model still wants tools, force a text answer
        if msg.tool_calls:
            print("Tool limit reached, asking for a final answer...")
            messages.append(msg.model_dump(exclude_none=True))
            for tool_call in msg.tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": "Tool limit reached. Answer with what you already know.",
                })
            msg = ask_llm(allow_tools=False)

        reply = msg.content or ""
        messages.append({"role": "assistant", "content": reply})

        print("assistant:", reply)

    except KeyboardInterrupt:
        print("\nAssistant stopped by user.")
        break

    except Exception as e:
        # Roll back the half-finished turn so the history stays valid
        del messages[history_len:]
        print(f"\nError: {e}")
        print("Please try again.")