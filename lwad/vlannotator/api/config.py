import os

class APIConfig:
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_ENDPOINT = "https://zhaohanggpt4v.openai.azure.com/openai/deployments/gpt4o/chat/completions?api-version=2024-02-15-preview"
    
    # ZhipuAI Configuration
    ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
    ZHIPUAI_MODEL = "glm-4v-plus"
    
    # Qwen Configuration
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_DEFAULT_MODEL = "qwen-vl-max-0809"

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = "gpt-4o"

def validate_api_keys():
    required_keys = [
        "AZURE_OPENAI_API_KEY",
        "ZHIPUAI_API_KEY",
        "DASHSCOPE_API_KEY"
    ]
    
    missing_keys = []
    for key in required_keys:
        if not getattr(APIConfig, key):
            missing_keys.append(key)
            
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")