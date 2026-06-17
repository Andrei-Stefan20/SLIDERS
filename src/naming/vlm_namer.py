"""Feature naming using a multimodal VLM."""

import hashlib
import io
import json
from pathlib import Path

import torch
from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _crops_fingerprint(crops: list[Image.Image]) -> str:
    h = hashlib.sha1()
    for c in crops:
        buf = io.BytesIO()
        c.save(buf, format="PNG")
        h.update(buf.getvalue())
    return h.hexdigest()

CONTRASTIVE_PROMPT = (
    "You are analyzing visual features discovered by a neural network.\n\n"
    "I will show you two groups of image crops:\n"
    "- HIGH: crops where a specific feature is strongly present\n"
    "- LOW: crops where that same feature is absent\n\n"
    "Your task: identify the SINGLE visual property that distinguishes HIGH from LOW.\n\n"
    "Rules:\n"
    "- NAME: a concise label (2-5 words), lowercase, no punctuation\n"
    "- DESC: one sentence describing what the feature looks like visually and where it appears\n"
    "- NEVER use 'texture' or 'pattern' alone, be specific\n"
    "- If both groups look identical, output:\n"
    "  NAME: undifferentiated\n"
    "  DESC: No distinguishing visual property found.\n\n"
    "Output format (exactly two lines):\n"
    "NAME: <label>\n"
    "DESC: <one sentence>\n\n"
    "Examples:\n"
    "NAME: yellow halo spots\n"
    "DESC: Small circular lesions surrounded by a bright yellow chlorotic halo, scattered across the leaf surface.\n\n"
    "NAME: powdery white coating\n"
    "DESC: Dense white powdery residue covering the leaf surface, consistent with powdery mildew infection."
)

VERIFICATION_PROMPT = (
    "Look at the HIGH and LOW crops again. The proposed feature name is: '{name}'.\n\n"
    "Rate on a scale 1-5 how much more present '{name}' is in HIGH compared to LOW:\n"
    "  1 = not more present (name is wrong)\n"
    "  2 = barely more present\n"
    "  3 = somewhat more present\n"
    "  4 = clearly more present\n"
    "  5 = strongly more present (name is very accurate)\n\n"
    "Answer with ONLY the single digit (1-5)."
)

_FALLBACK = ("undifferentiated", "No distinguishing visual property found.")


class VLMFeatureNamer:
    """Names SAE features by showing image crops directly to a VLM.

    Uses Qwen3-VL-4B-Instruct by default. No text intermediate:
    the model sees the crops directly and returns a name + description.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-4B-Instruct",
        device: str | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.model_id = model
        self.processor = AutoProcessor.from_pretrained(model)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        dtype = torch.float16 if device != "cpu" else torch.float32

        logger.info(f"Loading {model} on {device} ({dtype})...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model, torch_dtype=dtype
        ).to(device).eval()

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, top_crops: list[Image.Image], bottom_crops: list[Image.Image]) -> Path | None:
        if not self.cache_dir:
            return None
        key = _crops_fingerprint(top_crops) + "_" + _crops_fingerprint(bottom_crops)
        return self.cache_dir / f"{self.model_id.replace('/', '_')}_{key}.json"

    def _build_messages(
        self,
        top_crops: list[Image.Image],
        bottom_crops: list[Image.Image],
        prompt: str,
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": "HIGH activation crops:"}]
        for crop in top_crops:
            content.append({"type": "image", "image": crop})
        content.append({"type": "text", "text": "LOW activation crops:"})
        for crop in bottom_crops:
            content.append({"type": "image", "image": crop})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _generate(self, messages: list[dict], max_new_tokens: int = 80) -> str:
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )

        generated = out_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=True,
        )[0].strip()

    @staticmethod
    def _parse(raw: str) -> tuple[str, str]:
        """Extract (name, description) from the two-line VLM output."""
        name = desc = ""
        for line in raw.splitlines():
            line = line.strip()
            if line.upper().startswith("NAME:"):
                name = line[5:].strip().rstrip(".").lower().strip("'\"")
            elif line.upper().startswith("DESC:"):
                desc = line[5:].strip()
        return name, desc

    def name_feature(
        self,
        top_crops: list[Image.Image],
        bottom_crops: list[Image.Image],
        verify: bool = True,
        avoid_names: list[str] | None = None,
    ) -> tuple[str, str]:
        """Return (name, description) for a feature identified contrastively.

        avoid_names: names already given to other features. When set, the VLM is asked for
        a distinct, more specific name (the cache is bypassed since the prompt differs)."""
        import re

        # avoid_names changes the prompt and is run-specific, so skip the crop-keyed cache.
        cache_path = None if avoid_names else self._cache_path(top_crops, bottom_crops)
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return cached["name"], cached["description"]

        prompt = CONTRASTIVE_PROMPT
        if avoid_names:
            prompt += (
                "\n\nThese names are ALREADY used by other features. Choose a DIFFERENT, "
                "more specific name that distinguishes this feature from them:\n- "
                + "\n- ".join(avoid_names)
            )
        messages = self._build_messages(top_crops, bottom_crops, prompt)
        raw = self._generate(messages, max_new_tokens=80)
        name, desc = self._parse(raw)

        if not name or name == "undifferentiated":
            if cache_path:
                cache_path.write_text(json.dumps({"name": _FALLBACK[0], "description": _FALLBACK[1]}))
            return _FALLBACK

        if verify:
            verify_msg = self._build_messages(
                top_crops, bottom_crops, VERIFICATION_PROMPT.format(name=name)
            )
            answer = self._generate(verify_msg, max_new_tokens=4).strip()
            match = re.search(r"[1-5]", answer)
            score = int(match.group()) if match else 0
            if score <= 2:
                logger.info(f"Verification score {score}/5 for '{name}', marked undifferentiated")
                return _FALLBACK
            if score == 3:
                name = f"{name} [weak]"

        if not desc:
            desc = f"Visual property: {name}."

        if cache_path:
            cache_path.write_text(json.dumps({"name": name, "description": desc}))
        return name, desc
