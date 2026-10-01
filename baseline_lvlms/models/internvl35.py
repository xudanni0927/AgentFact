from typing import Optional

from models.base import BaseVLM


class InternVL35(BaseVLM):
    """Local HF-hosted backend for InternVL3.5 (using the -HF checkpoint,
    which works with standard transformers AutoModelForImageTextToText
    rather than the original repo's custom .chat() + manual image
    transforms).

    Requires transformers>=4.52.1 (>=4.55.0 if you switch model_id to the
    20B+ variant): pip install -U transformers
    Model card: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
    """

    name = "internvl3.5"
    model_id = "OpenGVLab/InternVL3_5-8B-HF"

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype="auto", device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        import torch
        from PIL import Image

        self._load()

        content = []
        if image_path:
            content.append({"type": "image", "image": Image.open(image_path).convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            # Explicit greedy decoding (matches the other models); this
            # checkpoint's generation_config.json has no do_sample/temperature
            # of its own, so this makes the deterministic behavior explicit
            # rather than relying on the transformers library default.
            output_ids = self._model.generate(**inputs, max_new_tokens=800, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]
