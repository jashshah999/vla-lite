"""Make OpenVLA-7B work on a 2026 transformers stack, and fit on a small GPU.

Drop-in replacement for the load snippet in OpenVLA's README. Usage:

    from openvla_compat import load_openvla

    model, processor = load_openvla(
        "openvla/openvla-7b-finetuned-libero-spatial", precision="nf4"
    )
    action = predict_action(model, processor, image, "pick up the red block",
                            unnorm_key="libero_spatial")

Why this file exists
--------------------
OpenVLA-7B pins torch 2.2 / transformers 4.40.1 / timm 0.9.10 (its own
pyproject.toml). On a current stack it fails in three ways, the last of which is
silent and therefore the dangerous one:

1. ``AttributeError: 'OpenVLAForActionPrediction' object has no attribute
   '_supports_sdpa'`` — the checkpoint declares ``_supports_sdpa`` as a property
   reading ``self.language_model``, but transformers now resolves the attention
   implementation inside ``PreTrainedModel.__init__``, before that submodule
   exists.

2. ``AttributeError: ... object has no attribute 'generate'`` — transformers
   >= 4.50 no longer has ``PreTrainedModel`` inherit ``GenerationMixin``, and the
   checkpoint never inherits it explicitly, so ``predict_action``'s internal
   ``self.generate(...)`` call disappears.

3. **Silent vision bypass.** ``prepare_inputs_for_generation`` slices the prompt
   to its last token whenever ``past_key_values is not None``. Modern
   transformers passes an *empty* ``DynamicCache`` on the first step instead of
   ``None``, so the full prompt is discarded, ``forward`` takes its
   ``input_ids.shape[1] == 1`` cached-generation branch, and ``pixel_values`` is
   never consumed. The model then emits one constant token for every input.
   Measured on openvla-7b-finetuned-libero-spatial before the fix: token 31872
   repeated 7x for real LIBERO frames, a pure black image, a pure white image and
   random noise — all producing the identical "action"
   ``[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]``.
   It looks like a working policy and is completely blind.

Issues #286/#287 (4-bit ``.to()`` failure) and #311 ("can't run on rigs with
<= 11 GB cards") are also addressed here: a bitsandbytes model is placed by
accelerate via ``device_map`` and must never have ``.to()`` called on it.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

__all__ = ["load_openvla", "predict_action", "modernize_openvla_class", "vision_dtype"]

_CODE_FALLBACK_REPO = "openvla/openvla-7b"


def _cache_length(past_key_values) -> int:
    """Length of whatever cache object transformers handed us (0 if empty)."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length() or 0
    try:
        return past_key_values[0][0].shape[2]
    except Exception:
        return 0


def _prepare_inputs_for_generation(
    self,
    input_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    pixel_values=None,
    attention_mask=None,
    **kwargs,
):
    """Cache-object-aware replacement for the checkpoint's version.

    Key difference: an *empty* cache is treated as no cache, and is forwarded as
    None so that forward()'s multimodal branch (which asserts
    ``past_key_values is None``) is reachable on the first step.
    """
    cache_len = _cache_length(past_key_values)

    if cache_len > 0:
        model_inputs = {"input_ids": input_ids[:, -1:], "past_key_values": past_key_values}
    elif inputs_embeds is not None:
        model_inputs = {"input_embeds": inputs_embeds, "past_key_values": None}
    else:
        model_inputs = {"input_ids": input_ids, "past_key_values": None}

    model_inputs.update(
        {
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "use_cache": kwargs.get("use_cache"),
        }
    )
    return model_inputs


def modernize_openvla_class(model_id: str):
    """Fetch the checkpoint's model class and repair the three incompatibilities."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.generation.utils import GenerationMixin

    target = "modeling_prismatic.OpenVLAForActionPrediction"
    try:
        cls = get_class_from_dynamic_module(target, model_id)
    except OSError:
        # finetuned checkpoints ship weights only and point at the base repo
        cls = get_class_from_dynamic_module(target, _CODE_FALLBACK_REPO)

    applied = []

    # (1) property -> plain attribute; the Llama backbone does support SDPA
    for klass in cls.__mro__:
        if isinstance(klass.__dict__.get("_supports_sdpa"), property):
            setattr(klass, "_supports_sdpa", True)
            applied.append(f"{klass.__name__}._supports_sdpa -> True")

    # (2) GenerationMixin re-mixed after cls so checkpoint overrides still win
    bases = (cls,) if issubclass(cls, GenerationMixin) else (cls, GenerationMixin)
    if len(bases) == 2:
        applied.append("re-mixed GenerationMixin")

    # (3) the silent one
    cls = type(
        f"{cls.__name__}Modernized",
        bases,
        {"prepare_inputs_for_generation": _prepare_inputs_for_generation},
    )
    applied.append("empty-cache-aware prepare_inputs_for_generation")

    return cls, applied


def _quant_config(precision: str):
    if precision not in {"int8", "nf4", "fp4"}:
        return None
    from transformers import BitsAndBytesConfig

    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4" if precision == "nf4" else "fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_openvla(
    model_id: str = "openvla/openvla-7b",
    precision: str = "nf4",
    device: str = "cuda",
    verbose: bool = True,
):
    """Load OpenVLA at the requested precision without tripping over the above.

    precision: "bf16" | "fp32" | "int8" | "nf4" | "fp4"
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model_cls, applied = modernize_openvla_class(model_id)
    if verbose:
        for line in applied:
            print(f"[openvla-compat] {line}")

    kwargs: dict = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    quant = _quant_config(precision)
    if quant is not None:
        kwargs["quantization_config"] = quant
        # never .to() a bitsandbytes model -- this is issues #286/#287
        kwargs["device_map"] = {"": 0} if device.startswith("cuda") else "cpu"
    else:
        kwargs["dtype"] = torch.bfloat16 if precision == "bf16" else torch.float32

    model = model_cls.from_pretrained(model_id, **kwargs)
    if quant is None:
        # config carries torch_dtype=bfloat16 for the nested language model, and
        # the dtype= kwarg does not always reach it, which surfaces later as
        # "expected m1 and m2 to have the same dtype". Force it explicitly.
        model = model.float() if precision == "fp32" else model.to(torch.bfloat16)
        model = model.to(device)
    model.eval()
    return model, processor


def vision_dtype(model) -> torch.dtype:
    """dtype of the vision tower.

    Under bitsandbytes the LLM is quantized but the vision backbone keeps the
    checkpoint's native dtype (fp16 for these releases). Feeding bf16 pixels
    raises "Input type (BFloat16) and bias type (Half) should be the same".
    """
    try:
        return next(model.vision_backbone.parameters()).dtype
    except Exception:
        return torch.float32


def predict_action(
    model,
    processor,
    image: Image.Image,
    instruction: str,
    unnorm_key: str | None = None,
    device: str | None = None,
) -> np.ndarray:
    """One 7-DoF action for an image + natural-language instruction."""
    if device is None:
        device = next(model.parameters()).device.type
    if unnorm_key is None:
        stats = getattr(model, "norm_stats", None) or {}
        unnorm_key = sorted(stats)[0] if stats else "bridge_orig"

    prompt = f"In: What action should the robot take to {instruction.lower().rstrip('.')}?\nOut:"
    inputs = processor(prompt, image)
    vdt = vision_dtype(model)
    moved = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, dtype=vdt) if v.is_floating_point() else v.to(device)
        else:
            moved[k] = v

    with torch.no_grad():
        action = model.predict_action(**moved, unnorm_key=unnorm_key, do_sample=False)
    return np.asarray(action, dtype=np.float64)


def self_check(model_id: str = "openvla/openvla-7b-finetuned-libero-spatial", precision: str = "nf4"):
    """Prove the model is actually looking at the image.

    A black frame and a white frame must not produce the same action. If they
    do, generation is bypassing vision (bug 3) and any benchmark built on top of
    it is meaningless.
    """
    model, processor = load_openvla(model_id, precision=precision)
    black = Image.fromarray(np.zeros((224, 224, 3), np.uint8))
    white = Image.fromarray(np.full((224, 224, 3), 255, np.uint8))
    a = predict_action(model, processor, black, "pick up the red block")
    b = predict_action(model, processor, white, "pick up the red block")
    identical = bool(np.allclose(a, b))
    print(f"black action: {np.round(a, 4)}")
    print(f"white action: {np.round(b, 4)}")
    print("VISION BYPASSED (bad)" if identical else "vision active (good)")
    return not identical


if __name__ == "__main__":
    raise SystemExit(0 if self_check() else 1)
