# SigLIP versus DINOv2

SigLIP run only; DINOv2 organizer metrics are the existing repository baseline, not a new run. Same 4 fps, windows, grouped LOVO, train-only scalers, logistic C, type/heuristic selection and nested threshold procedure. No DINOv2 self baseline. Self videos held out from all fitting. LOVO recall is optimistic because its threshold is selected on those OOF scores. Historical timings were measured on a different machine and are not a speed comparison.

| Split | Encoder | Recall | False-positive rate | Median time error (s) | Type accuracy |
|---|---|---:|---:|---:|---:|
| organizer | dinov2-small | 1.0000 | 0.9474 | 5.0000 | 0.0952 |
| organizer | siglip-base | 1.0000 | 1.0000 | 3.5000 | 0.1429 |
| self | siglip-base | 1.0000 | 0.2500 | 5.0000 | 0.3333 |

Full per-video results, nested checks, layout checks and timings: `comparison.json`. Runtime excludes model loading. Self-video timings may reuse detector tracks and are not end-to-end API latency.

Hardware: NVIDIA GeForce RTX 4070 Laptop GPU; torch 2.14.1+cu132; transformers 5.20.1.
