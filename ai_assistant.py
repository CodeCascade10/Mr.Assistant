
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from Utilities.utility import log
from tools import tools, add_tool_results


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

load_dotenv()

MAX_REQUESTS = 10
MAX_RUNTIME = 300
MAX_TOOL_ROUNDS = 5

MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")

if not all([MODEL, BASE_URL, API_KEY]):
    sys.exit("Missing BASE_URL, API_KEY or MODEL in your .env file.")

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=180.0,
    max_retries=0
)

log("AI Assistant initialized")
log(f"Model: {MODEL}")
log(f"Base URL: {BASE_URL}")


# --------------------------------------------------
# CONVERSATION HISTORY
# --------------------------------------------------

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Always respond in less than 100 words. "
            "Use available tools when necessary. "
            "If the user asks about weather without naming a city, "
            "first call get_user_city, then call get_city_weather "
            "using the detected city. "
            "If the user specifies a city, call get_city_weather directly. "
            "After receiving tool results, explain them clearly. "
            "/no_think"
        )
    }
]


# --------------------------------------------------
# LLM REQUEST
# --------------------------------------------------

def ask_llm(allow_tools=True):

    request_data = {
        "model": MODEL,
        "messages": messages,
    }

    if allow_tools:
        request_data["tools"] = tools
        request_data["tool_choice"] = "auto"

    log(f"Sending LLM request | Tools enabled: {allow_tools}")

    start_time = time.time()

    try:
        response = client.chat.completions.create(**request_data)

        elapsed = time.time() - start_time
        msg = response.choices[0].message

        log(f"LLM response received in {elapsed:.2f} seconds")
        log(f"Tool calls requested: {len(msg.tool_calls or [])}")
        log(f"Response content: {msg.content or ''}")

        return msg

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"LLM request failed after {elapsed:.2f} seconds")
        log(f"LLM error: {type(e).__name__}: {e}")
        raise


# --------------------------------------------------
# RESPONSE CLEANING
# --------------------------------------------------

def clean_response(content):

    if not content:
        return ""

    content = re.sub(
        r"<think>.*?</think>",
        "",
        content,
        flags=re.DOTALL
    )

    return content.strip()


# --------------------------------------------------
# CHAT FUNCTION
# --------------------------------------------------

def chat(user_input):

    messages.append({
        "role": "user",
        "content": user_input
    })

    log(f"User input: {user_input}")

    try:
        for round_number in range(MAX_TOOL_ROUNDS + 1):

            log(f"Starting conversation round {round_number + 1}")

            msg = ask_llm(allow_tools=True)

            if not msg.tool_calls:

                answer = clean_response(msg.content)

                messages.append({
                    "role": "assistant",
                    "content": answer
                })

                log(f"Final assistant response: {answer}")

                return answer

            log(f"LLM requested {len(msg.tool_calls)} tool call(s)")

            if round_number == MAX_TOOL_ROUNDS:

                log("Maximum tool rounds reached")

                add_tool_results(
                    messages,
                    msg,
                    override_result=(
                        "Tool execution limit reached. "
                        "Please answer using the available results."
                    )
                )

                final_msg = ask_llm(allow_tools=False)
                answer = clean_response(final_msg.content)

                messages.append({
                    "role": "assistant",
                    "content": answer
                })

                log(f"Final response after tool limit: {answer}")

                return answer

            # Execute tools and append their results
            for tool_call in msg.tool_calls:
                log(f"Tool requested: {tool_call.function.name}")
                log(f"Tool arguments: {tool_call.function.arguments}")

            add_tool_results(messages, msg)

            log("Tool execution cycle completed")

        return "Unable to complete the request."

    except Exception as e:

        log(f"Chat error: {type(e).__name__}: {e}")

        if messages and messages[-1]["role"] == "user":
            messages.pop()

        raise


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    request_count = 0
    start_time = time.time()

    log("Main program started")

    print("\nAI Assistant is ready!")
    print("Type 'exit' to quit.\n")

    while True:

        if request_count >= MAX_REQUESTS:
            log("Maximum request limit reached")
            print("\nMaximum request limit reached.")
            break

        if time.time() - start_time >= MAX_RUNTIME:
            log("Maximum runtime reached")
            print("\nMaximum runtime reached.")
            break

        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "bye"):
                log("User exited the assistant")
                print("Assistant: Goodbye!")
                break

            request_count += 1

            print("\nAssistant: ", end="", flush=True)

            request_start = time.time()

            answer = chat(user_input)

            print(answer)
            print()

            log(
                f"Request {request_count} completed in "
                f"{time.time() - request_start:.2f} seconds"
            )

        except KeyboardInterrupt:
            log("Keyboard interrupt received")
            print("\nAssistant: Goodbye!")
            break

        except Exception as e:
            log(f"Main error: {type(e).__name__}: {e}")
            print(f"\nERROR: {type(e).__name__}: {e}")
            print("Assistant: Sorry, I couldn't generate an answer.\n")


if __name__ == "__main__":
    main()
