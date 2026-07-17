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


class SmolVLMBackend:
    """SmolVLM via HuggingFace transformers, optionally with a LoRA adapter.

    This serves the fine-tuned Stage 1 model: it loads the base SmolVLM and, if an
    ``adapter`` directory is given, layers the trained LoRA weights on top via peft.
    With no adapter it's the off-the-shelf SmolVLM (the fine-tuning control).
    """

    def __init__(self, model_id: str = config.SMOLVLM_ID, adapter: str | Path | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = config.device()
        # do_image_splitting=False matches how the adapter was trained (and caps
        # image-token count / memory).
        self.processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=config.dtype(self.device)
        )
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.to(self.device)
        self.model.eval()

    def caption(self, image_path: str | Path, prompt: str) -> str:
        import torch
        from PIL import Image

        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[[Image.open(image_path).convert("RGB")]],
            return_tensors="pt",
        ).to(self.device)
        amp_dtype = config.dtype(self.device)
        use_autocast = self.device in ("cuda", "mps")
        with torch.no_grad(), torch.autocast(
            device_type=self.device, dtype=amp_dtype, enabled=use_autocast
        ):
            generated = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
        trimmed = generated[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True).strip()


def get_backend(name: str, model: str | None = None,
                temperature: float = 0.2, seed: int | None = None,
                adapter: str | None = None):
    if name == "ollama":
        return OllamaBackend(model or config.OLLAMA_VLM, temperature=temperature, seed=seed)
    if name == "hf":
        # HFQwenBackend already decodes greedily (do_sample=False), so it is
        # deterministic and ignores temperature/seed.
        return HFQwenBackend(model or config.HF_QWEN_ID)
    if name == "smolvlm":
        # Deterministic greedy decode; ignores temperature/seed. ``adapter`` points
        # at the trained LoRA dir (config.ADAPTER_DIR) for the fine-tuned arm.
        return SmolVLMBackend(model or config.SMOLVLM_ID, adapter=adapter)
    raise ValueError(f"Unknown backend: {name!r} (expected 'ollama', 'hf', or 'smolvlm')")
