import os
import sys
from typing import Optional

import requests

# Make the AgentFact project root importable so we can reuse image_to_code
# without duplicating the base64/resize logic.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent_utils import image_to_code  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.base import BaseVLM


class GPT4oMini(BaseVLM):
    name = "gpt-4o-mini"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.base_url = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        content = [{"type": "text", "text": prompt}]
        if image_path:
            code_image = image_to_code(image_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{code_image}"},
            })

        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 800,
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
