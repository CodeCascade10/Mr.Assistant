
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from tools import tools, FUNCTION_NAME_TO_FUNCTION_MAP

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY"),
    timeout=180.0,
    max_retries=1
)

messages = []

messages.append({
    "role": "system",
    "content": "You are a helpful assistant."
})

messages.append({
    "role": "system",
    "content": "Always respond in less than 100 words."
})
messages.append({
    "role": "system",
    "content": "You are a helpful assistant. Always respond in less than 100 words. /no_think"
})


print("Sending request to LLM...")


while True:
    user_input = input("user: ")

    if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    print("Waiting for response...")
    response = client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)

        for tool_call in msg.tool_calls:
            function_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments or "{}")

            fn = FUNCTION_NAME_TO_FUNCTION_MAP.get(function_name)

            if fn:
                result = fn(**func_args)
            else:
                result = f"Function {function_name} not found."

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

        # Second call to API
        print("Waiting for response...")    
        response = client.chat.completions.create(
            model=os.getenv("MODEL"),
            messages=messages
        )

        msg = response.choices[0].message

    messages.append({
        "role": "assistant",
        "content": msg.content
    })

    print("LLM response received.")

    print("assistant:", msg.content)
