from typing import Optional

from models.base import BaseVLM


class Qwen3VL(BaseVLM):
    """Local HF-hosted backend for Qwen3-VL.

    Requires transformers built from source (or a release with Qwen3-VL
    support, e.g. >=4.57): pip install git+https://github.com/huggingface/transformers
    Model card: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
    """

    name = "qwen3-vl"
    model_id = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
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
            # do_sample=False overrides this checkpoint's bundled
            # generation_config.json (do_sample=True, temperature=0.7) to
            # get deterministic greedy decoding, matching the other models.
            output_ids = self._model.generate(**inputs, max_new_tokens=800, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]
