# v9 reproducibility facts on the server <GPU_SERVER> (v9j) — 22/11/2026

> English translation of the original (Vietnamese) document; content, numbers and codes are unchanged.

Read-only (`<GPU_SERVER>` (remote shell), BatchMode; nothing written, nothing installed, no GPU run). Project env: `<PYTHON_ENV>/bin/python` (Python 3.10; transformers 4.57.2; faiss 1.15.1) — exactly as stated on line 5 of `prereg_v9_capture.py`.
v9 pipeline: `prereg_v9_capture.py` -> `tierA_smoke.load_meta_index` reads `data/vectorstore/{visual_siglip.index,visual.index}`; SigLIP2 queries/gallery go through `tools/siglip_engine.load_siglip()`, CLIP through the `clip` package (`clip.load("ViT-B/32")`, per the docstring of `tierA_smoke.reembed`).

## 1. Checkpoint revision

### SigLIP2 — CONFIDENCE: HIGH (revision + sha256 read directly)
- HF repo: `google/siglip2-so400m-patch14-384` (the `REPO = ...` line in `siglip_engine.py`), loaded with `SiglipModel.from_pretrained(REPO, dtype=fp16)` + `SiglipProcessor`.
- Cache: `~/.cache/huggingface/hub/models--google--siglip2-so400m-patch14-384/`
  - `refs/main` = `e8e487298228002f3d8a82e0cd5c8ea9c567f57f` (also the name of the `snapshots/` directory, exactly 1 snapshot)
  - `model.safetensors` -> blob `9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2`; recomputing `sha256sum` gives exactly this value.
- Honesty note: the cache was downloaded on 27/10/2026 (dated 23-30/10), the snapshot has not changed since; there is no log proving that this very snapshot was loaded during the 21/11 capture run, but it is the only snapshot in the cache and there is no other HF_HOME (`HF_HOME` is empty) -> a strong inference, not direct evidence from the time of the run.

### CLIP ViT-B/32 — CONFIDENCE: HIGH for the weights file; MEDIUM for "this exact file was used"
- NOT an HF checkpoint: uses the OpenAI `clip` package (`import clip; clip.load("ViT-B/32")`), `visual_manifest.json` records `model_id: openai/clip-vit-b32, model_name: ViT-B/32, clip_version 1.0`.
- Weights: `~/.cache/clip/ViT-B-32.pt`, 353 976 522 bytes, `sha256 = 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
- Original URL in the package (`clip.clip._MODELS["ViT-B/32"]`): `http<REDACTED_PATH>` — the hash in the URL matches the recomputed sha256.
- Limitation: `visual_manifest.json` is stale (n_vectors 532, whereas the index is currently 1667 and was rebuilt in fp32 on 18/11, D8), so it is used only to confirm the model name.

## 2. Resample / preprocessing filter

### SigLIP2 — CONFIDENCE: HIGH
`snapshots/e8e4.../preprocessor_config.json`: `image_processor_type SiglipImageProcessor`, `do_resize true`, `size {height 384, width 384}`, **`resample: 2`** (PIL: 2 = BILINEAR; 0 NEAREST, 1 LANCZOS, 3 BICUBIC), `do_rescale true` (1/255), `do_normalize true`, `image_mean = image_std = [0.5,0.5,0.5]`, no crop (direct resize to 384x384, aspect ratio not preserved). Wrapper: `preprocess(img) = processor(images=img)["pixel_values"][0].float()`; encode in fp16, L2-normalize in fp32.
(The value `2` is a numeric code in the config file; the mapping 2 = BILINEAR follows the standard `PIL.Image.Resampling`/`PILImageResampling`, and no code was run to print the enum — high confidence but it is standard knowledge, not output from a run.)

### CLIP ViT-B/32 — CONFIDENCE: HIGH (printed from `inspect.getsource(clip.clip._transform)`, package at `<PYTHON_ENV>/lib/python3.10/site-packages/clip/`)
`Resize(224, interpolation=BICUBIC)` (short side -> 224, aspect ratio preserved) -> `CenterCrop(224)` -> convert RGB -> `ToTensor` -> `Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))`.

## 3. FAISS index class — CONFIDENCE: HIGH (read with `faiss.read_index`, faiss 1.15.1)
```
visual_siglip.index  IndexFlatIP  ntotal=1667  d=1152  metric_type=0 (METRIC_INNER_PRODUCT)  is_trained=True
visual.index         IndexFlatIP  ntotal=1667  d=512   metric_type=0 (METRIC_INNER_PRODUCT)  is_trained=True
```
- `visual_siglip.index` (SigLIP2-I) sha256 `6377a3769d7ae6d6023b637f576053c1adad13c4d333b4e907f7247f8f09afef`; `visual.index` (CLIP-B/32-I, fp32 version rebuilt on 18/11) sha256 `31b8f9d21462cbb52864e0924466e8bc41585bd1c5a67350be5fae3198ff8dc0`.
- Search is an inner product over L2-normalized vectors (flat, exact, no approximation).
- In v9, the gallery vectors are taken from `T.stored_vectors(index)` and then `renorm`-ed; identity scores are computed in numpy from that matrix (`index.search` is not used), so the index layer is only a vector store.

## Could not be determined
- The commit/revision of the `clip` package (only `clip_version 1.0` in the old manifest; the package is installed in the env, and no git commit could be read).
- Direct evidence (a log) that exactly the SigLIP2 snapshot e8e4... was loaded during the 21/11 capture (see section 1).
