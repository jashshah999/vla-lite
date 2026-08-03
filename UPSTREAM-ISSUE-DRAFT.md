Draft for https://github.com/openvla/openvla/issues/new — review and file when ready.
(Also worth posting as a HF discussion on openvla/openvla-7b, since the fix lives in the checkpoint's remote code, not this repo.)

---

**Title:** OpenVLA-7B silently ignores the image on transformers ≥4.50 — constant action for every input

**Body:**

On a current stack, `predict_action` returns a *plausible but input-independent* action: the vision tower is never consumed. There is no error message, so this is easy to miss — the model appears to work.

### Reproduction

`openvla/openvla-7b-finetuned-libero-spatial`, greedy decoding, torch 2.10.0+cu128 / transformers 4.56.2 / timm 0.9.16 / bitsandbytes 0.50.0, RTX 5000 Ada 16 GB.

Every one of these inputs produced the identical output — generated token `31872` seven times, unnormalizing to `[0.096, 0.1071, -0.0027, -0.0016, -0.015, -0.0193, 0.0]`:

- a real LIBERO frame (episode 0, frame 0)
- a different real LIBERO frame (frame 9, visibly different scene)
- a pure black 224×224 image
- a pure white 224×224 image
- uniform random noise

### Root cause

`modeling_prismatic.py::prepare_inputs_for_generation` contains:

```python
if past_key_values is not None:
    input_ids = input_ids[:, -1:]
```

Under transformers 4.40 the first `generate()` step passed `past_key_values=None`. Modern transformers passes an **empty `DynamicCache`** instead, which is not `None`. So on step 0 the full 35-token prompt is sliced to its last token. `forward` then routes on `input_ids.shape[1] == 1` into the cached-generation branch, which never reads `pixel_values`. The model decodes from a degenerate state and emits the same token every step.

Confirmed by hooking `forward` during generation — step 0 arrives with `input_ids=(1, 1)` instead of `(1, 35)`, with `pixel_values` present but unused by that branch.

Note that simply restoring the full prompt is **not sufficient**: the multimodal branch also asserts `past_key_values is None`, so an empty cache has to be passed through as `None`.

### Fix

Make the slice conditional on actual cache *length*, and normalize an empty cache to `None`:

```python
def _cache_length(pkv):
    if pkv is None:
        return 0
    if hasattr(pkv, "get_seq_length"):
        return pkv.get_seq_length() or 0
    try:
        return pkv[0][0].shape[2]
    except Exception:
        return 0

def prepare_inputs_for_generation(self, input_ids=None, past_key_values=None,
                                  inputs_embeds=None, pixel_values=None,
                                  attention_mask=None, **kwargs):
    cache_len = _cache_length(past_key_values)
    if cache_len > 0:
        model_inputs = {"input_ids": input_ids[:, -1:], "past_key_values": past_key_values}
    elif inputs_embeds is not None:
        model_inputs = {"input_embeds": inputs_embeds, "past_key_values": None}
    else:
        model_inputs = {"input_ids": input_ids, "past_key_values": None}
    model_inputs.update({"attention_mask": attention_mask,
                         "pixel_values": pixel_values,
                         "use_cache": kwargs.get("use_cache")})
    return model_inputs
```

After this change the same frames produce distinct, image-dependent actions, including the gripper channel switching between 0.0 and 0.996:

| input | action |
|---|---|
| real frame 0 | `[0.3336, 0.4689, -0.252, 0.0, -0.021, -0.0387, 0.9961]` |
| real frame 5 | `[-0.003, -0.0014, 0.0926, 0.0, 0.0, -0.0206, 0.0]` |
| real frame 9 | `[-0.2076, -0.1401, -0.9338, 0.0149, -0.1459, 0.0001, 0.9961]` |

### Two other modern-stack blockers hit on the way

1. `AttributeError: 'OpenVLAForActionPrediction' object has no attribute '_supports_sdpa'` — `_supports_sdpa` is declared as a property reading `self.language_model`, but transformers now resolves the attention implementation inside `PreTrainedModel.__init__`, before that submodule is assigned. A plain class attribute (`_supports_sdpa = True`) is behaviour-preserving since the Llama backbone supports SDPA.
2. `AttributeError: ... has no attribute 'generate'` — transformers ≥4.50 no longer has `PreTrainedModel` inherit `GenerationMixin`; the class needs to inherit it explicitly (the warning transformers prints about this is easy to scroll past).

### Suggested regression test

Because the failure is silent, an assertion that an action was returned is not enough. A black frame and a white frame must not produce the same action:

```python
a = predict_action(model, processor, black_image, instruction)
b = predict_action(model, processor, white_image, instruction)
assert not np.allclose(a, b), "vision is being bypassed"
```

### Related open issues this also resolves

- #286 / #287 — 4-bit loading crashing on `.to()`. A bitsandbytes model is placed by accelerate via `device_map`; `.to()` must not be called afterwards. The README snippet does `.from_pretrained(...).to("cuda:0")`, which is why everyone following it hits this.
- #311 — "can't run on rigs with ≤11 GB cards". With the above, nf4 peaks at **4.60 GB** (vs 15.29 GB bf16) on a LIBERO-finetuned checkpoint, so an 8 GB card is enough. Measured latency is essentially unchanged (245 ms vs 260 ms p50); the win is memory, not speed.

Happy to open a PR against the HF checkpoint's remote code if that's the preferred route.
