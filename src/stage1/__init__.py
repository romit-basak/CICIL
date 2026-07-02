"""Stage 1 (Vision & VLM) for the CICIL pipeline.

Owns the cultural VQA prompting module, the generic Qwen2.5-VL baseline, the
SmolVLM LoRA fine-tuning scaffold, and generation of the Spanish intermediate
descriptions that feed Stage 2 retrieval + translation.
"""
