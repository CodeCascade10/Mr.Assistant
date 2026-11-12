
import os
from dotenv import load_dotenv

load_dotenv()

START_SEQUENCE = "\033[90m"
END_SEQUENCE = "\033[0m"

def log(*text):
    if os.getenv("DEBUG", "False").lower() == "true":
        print(f"  {START_SEQUENCE}{' '.join(map(str, text))}{END_SEQUENCE}")
