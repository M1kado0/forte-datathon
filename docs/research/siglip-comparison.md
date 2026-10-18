# SigLIP embedding replacement (TZ-5.2, TZ-F.8; #27, #43)

The replacement is `google/siglip-base-patch16-224`, already listed in `docs/MODELS.md`
and downloaded by `make weights`. Its [model card](https://huggingface.co/google/siglip-base-patch16-224)
declares Apache-2.0 and specifies 224×224 RGB inputs with mean/std 0.5. The
[Transformers vision-model documentation](https://huggingface.co/docs/transformers/model_doc/siglip#transformers.SiglipVisionModel)
exposes pooled image features through `pooler_output`. Only the local vision encoder is loaded.
No text generation, image upload, or remote inference is involved.

SigLIP is the default frame encoder. `FORTE_EMBEDDER=dinov2-small` selects the previous
native-resolution CLS-plus-mean-patch implementation for comparison. SigLIP resizes
the entire RGB frame with bicubic interpolation to its pretrained 224×224 resolution;
it does not crop. This changes spatial preprocessing as well as the encoder. D-FINE,
OWLv2, rules, sampling rate, window labels and logistic-regression settings stay fixed.
The two feature spaces have separate caches and model fingerprints. An old DINOv2
classifier is rejected by SigLIP inference until retrained.

## Reproduce

Place organizer videos in `data/raw/` (or the established `data/raw/videos/`), with
`data/labels.csv`. Put self recordings in `data/self_videos/`, with
`data/labels_self.csv`. `FORTE_DATA_ROOT` and `FORTE_WEIGHTS_DIR` may point a worktree
to shared local data and weights. They are filesystem paths, never remote endpoints.

```
uv run python -m scripts.weights
uv run python -m src.models.compare
```

The comparison runs only SigLIP through the full `src.models.train` procedure: grouped
LOVO, leave-one-arrangement-out, nested threshold check, and training-only scaling.
The existing `reports/metrics.json` is archived as the historical DINOv2 baseline
before training. DINOv2 is not rerun, and there is no DINOv2 self-recorded baseline.
The earlier baseline used a different GPU, so its timings cannot establish a speedup.
Thresholds and heuristic/type settings are fitted using organizer videos only.
Self recordings are a separate held-out evaluation. The reported LOVO recall uses
a threshold selected on those same OOF scores and is optimistic; the nested and
self-recorded checks are essential context. Do not retune on the self recordings.

Outputs: `reports/embedding_comparison/comparison.{json,md}`, historical DINOv2 organizer
metrics, fresh SigLIP organizer and self metrics, `models/window_clf_siglip-base.pkl`,
and append-only experiment rows. The run leaves the SigLIP artifact in `models/window_clf.pkl` and its organizer
metrics in `reports/metrics.json`. Runtime includes local decoding; self-video timing
may reuse detector tracks and excludes loading and JPEG evidence, so it is not API latency.

The self dataset has few negatives: show counts alongside rates. A single missed
violation matters more than a lower false-positive rate under the repo's recall-first
criterion. Do not infer an overall improvement from LOVO recall alone.
