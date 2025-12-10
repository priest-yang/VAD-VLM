from .config import APIConfig
from openai import OpenAI
import base64
import json

#  base 64 format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def qwen_api(image_path, text, model=None):
    if model is None:
        model = APIConfig.QWEN_DEFAULT_MODEL
        
    base64_image = encode_image(image_path)
    client = OpenAI(
        api_key=APIConfig.DASHSCOPE_API_KEY,
        base_url=APIConfig.DASHSCOPE_BASE_URL,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )
    result = completion.model_dump_json()
    result = json.loads(result)
    return result["choices"][0]["message"]["content"]
