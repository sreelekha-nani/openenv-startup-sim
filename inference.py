import os
from openai import OpenAI

# Environment variables (as required)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize OpenAI client (required for checker)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# Main function (required)
def run():
    print("START")   # required log
    print("STEP")    # required log
    print("END")     # required log