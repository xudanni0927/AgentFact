import os
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.base import BaseVLM


class GeminiFlash(BaseVLM):
    """Requires: pip install google-genai
    Requires env var GEMINI_API_KEY (get one at https://aistudio.google.com/apikey).
    """

    name = "gemini-3-flash"
    model_id = "gemini-3-flash-preview"

    def __init__(self):
        from google import genai  # deferred import: don't require the package
        # to be installed just to import this module.
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        from google.genai import types

        contents = []
        if image_path:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = "image/png" if ext == ".png" else "image/jpeg"
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        contents.append(prompt)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text
