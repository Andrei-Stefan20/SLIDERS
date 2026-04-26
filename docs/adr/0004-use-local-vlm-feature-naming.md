# Use a local VLM for feature naming

## Context

SAE features are identified by integer ids (0–8191). For sliders to be usable, each id needs a short human-readable label. The label should describe what images with high activation share visually, specifically the property that distinguishes them from images with low activation on the same feature.

Naming runs once offline after SAE training, not at query time.

## Decision

Use `Qwen/Qwen3-VL-4B-Instruct` (or any `AutoModelForVision2Seq` model) to name features contrastively. For each feature, rank corpus images by activation, take the top-N and bottom-N, localize a 96px crop per image around the highest-activation patch, then show those crops to the VLM in a single prompt with HIGH crops first and LOW crops second. Parse a `NAME` and `DESC` from the output. Run a second verification pass where the model scores (1–5) how well the proposed name discriminates HIGH from LOW; reject names scoring ≤2, mark scores of 3 as `[weak]`.

Results land in `models/feature_names.json`.

Crop localization. The 16×16 patch grid from DINOv2 (ViT-L/14, 14px patches) gives 256 patches per image. Running the SAE encoder on each patch token gives a (256, 8192) activation matrix; argmax on `feature_id` column identifies which patch fires most. The crop is a 96px window centered on that patch in 224×224 image coordinates.

Inference uses `do_sample=False` (greedy decoding) for deterministic output.

## Why this approach

Using an API model (GPT-4V, Claude) would require uploading images and managing credentials. Local inference keeps the naming pipeline self-contained and works on sensitive or proprietary datasets.

Showing crops rather than full images matters because a full 224×224 image contains many visual properties irrelevant to the feature. The crop isolates the region the feature actually responds to, giving the VLM a tighter signal. The crop coordinates come from the same SAE activations used for retrieval, so the model is naming exactly what the feature detects.

The contrastive structure (HIGH vs LOW) is the key prompt design. Showing only high-activation images would produce names for generic properties common to that dataset (green, round, textured). The LOW crops act as a negative condition, forcing the model to identify what's present in HIGH that's absent in LOW. That delta is the specific visual property encoded by that feature.

Qwen3-VL-4B fits on a single consumer GPU and handles multiple images in a single context window natively via `qwen_vl_utils.process_vision_info`. Smaller models struggle with the structured two-line output format; larger models don't improve name quality enough to justify the memory cost.

The two-pass verification catches hallucinated names. The generation pass can produce confident-sounding labels for features where HIGH and LOW crops look identical. The verification pass re-presents the same crops and asks the model to score its own name; this self-consistency check has higher recall for bad names than a single-pass threshold.

Verification scoring

| Score | Action |
| --- | --- |
| 1–2 | return `"undifferentiated"` |
| 3 | accept with `"[weak]"` suffix |
| 4–5 | accept |

`"undifferentiated"` names in the output indicate either a dead/polysemantic SAE feature or a crop localization failure (the patch grid placed the crop on a background region). Both are diagnostic. If a large fraction of named features come back undifferentiated, the SAE likely needs retraining with different λ or the dataset is too visually narrow.

## Consequences

Naming runs offline. Changing the VLM only requires passing a different `model=` argument to `VLMFeatureNamer`; the prompt, parsing, and verification logic are model-agnostic.

## Owner files

| File | Role |
| --- | --- |
| `src/naming/vlm_namer.py` | VLM loader, contrastive prompt, generation, verification, parsing |
| `src/naming/spatial_localization.py` | patch-token crop creation |
| `src/naming/feature_ranking.py` | ranks corpus images by SAE activation |
| `scripts/name_features.py` | orchestrates the full naming pipeline |
