from .config import APIConfig
import requests
import base64
from openai import OpenAI
import os
client = OpenAI() if "OPENAI_API_KEY" in os.environ else None

def gpt_4o_azure(img_path, text):
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')

    headers = {
        "Content-Type": "application/json",
        "api-key": APIConfig.AZURE_OPENAI_API_KEY,
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text
                    },
                ]
            }
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 800
    }

    i = 0
    while i < 20:
        try:
            response = requests.post(APIConfig.AZURE_ENDPOINT, 
                                  headers=headers, 
                                  json=payload, 
                                  timeout=30)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"Failed to make the request. Error: {e}")
            i += 1
            print("Retrying...")
            continue
    
    data = response.json()
    print(data["usage"])
    return data['choices'][0]['message']['content']


def gpt(img_path, text):
    client = OpenAI()
    
    # Encode the image in base64
    encoded_image = base64.b64encode(open(img_path, 'rb').read()).decode('ascii')
    
    # Build the payload for the OpenAI API request
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, 
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                },
                {
                    "type": "text",
                    "text": text
                },
            ], 
        }
    ]

    # API request parameters
    max_retries = 20
    for i in range(max_retries):
        try:
            # Make the request to the OpenAI API
            response = client.chat.completions.create(
                model=APIConfig.OPENAI_MODEL,
                messages=messages,
                max_tokens=800
            )
            # Extract the response content
            data = response.choices[0].message.content
            print(response.usage)
            return data
        
        except Exception as e:
            print(f"Failed to make the request. Error: {e}")
            if i < max_retries - 1:
                print("Retrying...")
            else:
                print("Exceeded maximum retries. Exiting.")