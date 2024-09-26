import json
import os
import re
import time
import openai
from openai import OpenAI
import httpx

MODEL = "gpt-4o"

def run_llm(prompt):
    key = 'none'
    proxy = "http://127.0.0.1:7890"
    client = OpenAI(
        api_key=key,
        http_client=httpx.Client(
            proxies=proxy
        ),
    )

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)
    print("Start openai")
    while True:
        try:
            response = client.chat.completions.create(
                    model=MODEL,
                    messages = messages,
                    temperature=0,
                    max_tokens=1000,
                    frequency_penalty=0,
                    presence_penalty=0)
            result = response.choices[0].message.content
            break

        except openai.error.OpenAIError as e:
            print("openai error, retry: ",e)
            time.sleep(3)
    print("End openai")
    #print(result)
    return result