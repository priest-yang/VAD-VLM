from .config import APIConfig
from zhipuai import ZhipuAI
import base64

def zhipuai_api(img_path, text):
    with open(img_path, 'rb') as img_file:
        img_base = base64.b64encode(img_file.read()).decode('utf-8')

    client = ZhipuAI(api_key=APIConfig.ZHIPUAI_API_KEY)
    response = client.chat.completions.create(
        model=APIConfig.ZHIPUAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_base
                        }
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    response = zhipuai_api("pic/0a04b286e2dd5602.jpg", "describe the scene")
    print(response)