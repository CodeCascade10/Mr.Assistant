
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

messages =[]

while True:
    user_input=input("user:")
    if user_input.lower() in  ["exit", "quit","goodbye", "bye"]:
        break

    messages.append({"role": "user", "content":user_input})

    response =client.chat.completions.create(
        model=os.getenv("MODEL"),
        messages=messages
    )

    messages.append(
        {
            "role": "assistant",
            "content": response.choices[0].message.content
        }
    )
  
    print("Assistant:" , response.choices[0].message.content)
   


