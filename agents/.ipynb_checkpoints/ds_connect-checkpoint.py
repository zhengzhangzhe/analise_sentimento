from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key=os.environ.get("api_key")

def ds_connect(key = api_key):
    client = OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
    )
    return client