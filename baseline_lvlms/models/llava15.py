from typing import Optional

from models.base import BaseVLM


class Llava15(BaseVLM):
    """Local HF-hosted backend for LLaVA-1.5.

    NOTE: not part of this run — already has results from a prior script
    (see baseline_lvlms/llava_*_results.jsonl). Kept here for framework
    completeness / future re-runs.

    Model card: https://huggingface.co/llava-hf/llava-1.5-7b-hf
    """

    name = "llava-1.5"
    model_id = "llava-hf/llava-1.5-7b-hf"

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", low_cpu_mem_usage=True, device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        import torch
        from PIL import Image

        self._load()

        content = []
        if image_path:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        chat_prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [Image.open(image_path).convert("RGB")] if image_path else None
        inputs = self._processor(images=images, text=chat_prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=800, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)[0]
