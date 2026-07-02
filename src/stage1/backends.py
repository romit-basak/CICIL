"""Vision-language backends behind one interface: ``caption(image, prompt)``.

Two interchangeable implementations of the generic Qwen2.5-VL baseline:
  * OllamaBackend  — local, quantized, fast on Apple Metal (for prelim numbers).
  * HFQwenBackend  — native HuggingFace weights (reproducible for the paper;
                     3B on local MPS, 7B on GCP).
"""

from __future__ import annotations

from pathlib import Path

from . import config


class OllamaBackend:
    """Serve Qwen2.5-VL locally via the ollama daemon.

    ``temperature=0`` with a fixed ``seed`` gives (near-)deterministic decoding for
    citable paper numbers; the default 0.2 matches the preliminary runs.
    """

    def __init__(self, model: str = config.OLLAMA_VLM,
                 temperature: float = 0.2, seed: int | None = None):
        import ollama

        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._client = ollama

    def caption(self, image_path: str | Path, prompt: str) -> str:
        # Normalize to RGB JPEG bytes: the dataset mixes jpg/png/webp and even
        # mislabels some files (e.g., a WebP with a .jpg extension), which the
        # ollama image loader rejects. PIL detects the true format regardless.
        from io import BytesIO

        from PIL import Image

        buf = BytesIO()
        Image.open(image_path).convert("RGB").save(buf, format="JPEG", quality=95)
        # num_ctx: ollama defaults to 4096, but a Qwen2.5-VL image plus the
        # cultural-VQA synthesis prompt (which folds in the per-question answers)
        # overflows it and silently yields empty output. 8192 clears the worst
        # observed prompt (~4460 tokens) with headroom.
        options = {"temperature": self.temperature, "num_ctx": 8192}
        if self.seed is not None:
            options["seed"] = self.seed
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt, "images": [buf.getvalue()]}],
            options=options,
        )
        return resp["message"]["content"].strip()


class HFQwenBackend:
    """Native Qwen2.5-VL via HuggingFace transformers (lazy-loaded)."""

    def __init__(self, model_id: str = config.HF_QWEN_ID):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = config.device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=config.dtype(self.device)
        ).to(self.device)
        self.model.eval()

    def caption(self, image_path: str | Path, prompt: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{Path(image_path).resolve()}"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def get_backend(name: str, model: str | None = None,
                temperature: float = 0.2, seed: int | None = None):
    if name == "ollama":
        return OllamaBackend(model or config.OLLAMA_VLM, temperature=temperature, seed=seed)
    if name == "hf":
        # HFQwenBackend already decodes greedily (do_sample=False), so it is
        # deterministic and ignores temperature/seed.
        return HFQwenBackend(model or config.HF_QWEN_ID)
    raise ValueError(f"Unknown backend: {name!r} (expected 'ollama' or 'hf')")
