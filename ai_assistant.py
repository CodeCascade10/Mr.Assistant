
import os
import re
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI

from Utilities.utility import log
from tools import tools, add_tool_results

from types import SimpleNamespace


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
    timeout=360.0,
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
            "You are a helpful general-purpose AI assistant. "
            "Answer general knowledge questions using your own knowledge. "
            "Use available tools whenever they are needed. "
            "Never claim that you cannot answer simply because a tool is unavailable. "
            "Always respond in less than 100 words. " 
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
        "stream": True
    }

    if allow_tools:
        request_data["tools"] = tools
        request_data["tool_choice"] = "auto"

    log(f"Sending streaming LLM request | Tools: {allow_tools}")

    start_time = time.time()

    content = ""
    tool_calls = {}
    tool_headers_printed = set()
    argument_headers_printed = set()

    try:
        stream = client.chat.completions.create(**request_data)

        for chunk in stream:

            delta = chunk.choices[0].delta

            # ----------------------------------
            # STREAM NORMAL RESPONSE
            # ----------------------------------

            if delta.content:
                content += delta.content
                print(delta.content, end="", flush=True)

            # ----------------------------------
            # STREAM TOOL CALLS
            # ----------------------------------

            if delta.tool_calls:

                for tool in delta.tool_calls:

                    index = tool.index

                    if index not in tool_calls:
                        tool_calls[index] = {
                            "id": "",
                            "name": "",
                            "arguments": ""
                        }

                    current = tool_calls[index]

                    if tool.id:
                        current["id"] = tool.id

                    if tool.function:

                        # Stream function name
                        if tool.function.name:

                            if index not in tool_headers_printed:
                                print(
                                    f"\n\n[Tool Call {index + 1}]",
                                    flush=True
                                )
                                print("Function: ", end="", flush=True)
                                tool_headers_printed.add(index)

                            current["name"] += tool.function.name

                            print(
                                tool.function.name,
                                end="",
                                flush=True
                            )

                        # Stream function arguments
                        if tool.function.arguments:

                            if index not in argument_headers_printed:
                                print("\nArguments: ", end="", flush=True)
                                argument_headers_printed.add(index)

                            current["arguments"] += tool.function.arguments

                            print(
                                tool.function.arguments,
                                end="",
                                flush=True
                            )

        formatted_calls = []

        for call in tool_calls.values():

            formatted_calls.append(
                SimpleNamespace(
                    id=call["id"],
                    function=SimpleNamespace(
                        name=call["name"],
                        arguments=call["arguments"]
                    )
                )
            )

        if formatted_calls:
            print("\n\n[Tool calls received completely]", flush=True)

        msg = SimpleNamespace(
            content=content,
            tool_calls=formatted_calls
        )

        log(
            f"Streaming completed in "
            f"{time.time() - start_time:.2f}s"
        )

        log(f"Tool calls received: {len(formatted_calls)}")

        return msg

    except Exception as e:

        log(
            f"Streaming request failed: "
            f"{type(e).__name__}: {e}"
        )

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
