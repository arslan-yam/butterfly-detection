from __future__ import annotations

from .geometry import normalize_xyxy, parse_qwen


class PredictionError(ValueError):
    def __init__(self, message, raw_text):
        super().__init__(message)
        self.raw_text = raw_text


class HuggingFaceBackend:
    """Only image pixels and the fixed prompt reach the model."""

    def __init__(self, config, device, dtype, revision):
        import torch
        from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                                  Owlv2ForObjectDetection, Qwen3VLForConditionalGeneration)

        self.torch = torch
        self.config = config
        self.device = device
        self.dtype = getattr(torch, dtype)
        options = dict(revision=revision, trust_remote_code=False)
        self.processor = AutoProcessor.from_pretrained(config["model_id"], use_fast=False, **options)
        classes = dict(dino=AutoModelForZeroShotObjectDetection, owlv2=Owlv2ForObjectDetection,
                       qwen=Qwen3VLForConditionalGeneration)
        model_options = dict(options, dtype=self.dtype)
        if config["backend"] == "qwen":
            model_options["attn_implementation"] = "sdpa"
        if config.get("load_in_4bit", False):
            if device != "cuda":
                raise ValueError("4-bit backend requires CUDA")
            from transformers import BitsAndBytesConfig
            model_options.update(device_map={"": 0}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype))
        self.model = classes[config["backend"]].from_pretrained(config["model_id"], **model_options)
        if not config.get("load_in_4bit", False):
            self.model.to(device)
        self.model.eval()

    def synchronize(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def predict(self, image):
        torch, config = self.torch, self.config
        raw = None
        with torch.inference_mode():
            if config["backend"] == "qwen":
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image}, {"type": "text", "text": config["prompt"]}]}]
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt", min_pixels=config["min_pixels"], max_pixels=config["max_pixels"])
            else:
                text = config["prompt"] if config["backend"] == "dino" else [config["queries"]]
                inputs = self.processor(images=image, text=text, return_tensors="pt")
            inputs = inputs.to(self.device)
            # Cast floating image tensors only; input_ids must stay integral.
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    inputs[key] = value.to(self.dtype)
            if config["backend"] == "qwen":
                generated = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                                temperature=None, top_p=None, top_k=None,
                                                max_new_tokens=config["max_new_tokens"])
                new_ids = generated[:, inputs.input_ids.shape[1]:]
                raw = self.processor.batch_decode(new_ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)[0]
                # Even a valid JSON prefix is incomplete when the token budget is exhausted.
                eos = self.model.generation_config.eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                if new_ids.shape[1] >= config["max_new_tokens"] and int(new_ids[0, -1]) not in eos:
                    raise PredictionError("Qwen generation reached max_new_tokens", raw)
                try:
                    return parse_qwen(raw), raw
                except (ValueError, KeyError, TypeError) as exc:
                    raise PredictionError(f"Invalid Qwen JSON: {exc}", raw) from exc
            outputs = self.model(**inputs)
            if config["backend"] == "dino":
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=config["threshold"],
                    text_threshold=config["text_threshold"], target_sizes=[(image.height, image.width)])
            else:
                results = self.processor.post_process_grounded_object_detection(
                    outputs, threshold=config["threshold"], target_sizes=[(image.height, image.width)])
            result = results[0]
            boxes = [normalize_xyxy(box, image.width, image.height, score)
                     for box, score in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist())]
            return [box for box in boxes if box is not None], raw
