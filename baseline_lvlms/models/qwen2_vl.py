from typing import Optional

from models.base import BaseVLM


class Qwen2VL(BaseVLM):
    """Local HF-hosted backend for Qwen2-VL.

    NOTE: not part of this run — already has results from a prior script
    (see baseline_lvlms/qwen_vl_*_results.jsonl). Kept here for framework
    completeness / future re-runs.

    Requires: pip install qwen-vl-utils
    Model card: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
    """

    name = "qwen2-vl"
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self):
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def generate(self, prompt: str, image_path: Optional[str] = None) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        self._load()

        content = []
        if image_path:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=800, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
