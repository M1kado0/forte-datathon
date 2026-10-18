# Action-first experiment: TZ-5.2, TZ-4.4, TZ-F.8 (#27, #28, #43)

The user requested activity detection before violation classification, then a type
and timecode. This experiment compares two frozen temporal encoders, keeping both
supervised decision stages identical. No cloud inference, audio or face recognition.

## Model sources

- [Microsoft X-CLIP base/patch32](https://huggingface.co/microsoft/xclip-base-patch32)
  is MIT-licensed, trained on Kinetics-400 with eight frames at 224×224. Its
  [documented interface](https://huggingface.co/docs/transformers/model_doc/xclip)
  returns video representations and video/text scores. We use a fixed activity
  vocabulary (counting, handing money, opening doors, leaving a desk, etc.), containing
  no violation verdicts or legal claims. Text is encoded locally, not generated.
- [Torchvision R3D-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html)
  is a Kinetics-400 video classifier distributed in the
  [BSD-3-Clause Torchvision package](https://github.com/pytorch/vision/blob/main/LICENSE).
  Its 512-dimensional pooled features and 400 action probabilities feed the two heads.
- Excluded: [VideoMAE K400](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)
  and [TimeSformer K400](https://huggingface.co/facebook/timesformer-base-finetuned-k400):
  their model cards declare CC-BY-NC-4.0, which does not meet this repo's commercial-use rule.

## Exact pipeline

1. Decode locally at 8 fps; retain the whole scene using square letterboxing. Sample
   two-second windows every second, including the final fractional tail. R3D receives
   16 frames at 112×112; X-CLIP receives eight at 224×224. These temporal/spatial
   transforms differ from the checkpoints' original evaluation recipes.
2. The frozen model supplies an activity label, relative confidence and video features
   for every window. This is dense clip-level activity recognition, not a pretrained
   bank-specific temporal action detector. No confidence/motion gate drops quiet scenes.
3. A separate binary logistic head classifies the activity windows. The scaler is
   fitted only on training videos. Labels use the existing weak time neighborhoods:
   positive overlap around violation points, negatives safely before the event,
   unannotated trailing windows excluded from fitting.
4. Adjacent accepted windows with the same activity form events. Only those events
   invoke the type classifier, trained on positive windows from training videos.
5. The event's strongest window supplies its type and centre time. At most two
   events at least five seconds apart are returned. We do not claim learned action
   onsets: the provided labels have point times, not action start/end annotations.
   Types whose description needs an observed object fall back to `other`; neither
   action model supplies an object detection that can fill that template honestly.

## Evaluation and serving

```
uv run python -m scripts.action_weights
uv run python -m src.models.action_train
```

Both models run grouped LOVO and strict nested LOVO. Each outer held-out video is
excluded from both heads, all scaling, and the inner-LOVO threshold selection.
The ordinary LOVO report uses a global OOF threshold and retains the earlier
baseline's optimistic recall convention; the nested report is the decision basis.
Model selection prioritizes nested recall, then false positives, then time error,
and is saved before any self-recorded evaluation. The self set was used in the prior
SigLIP experiment; it is a repeated benchmark, not a newly untouched test set. We
do not tune any new hyperparameter or threshold on it.

All numbers and per-video outputs go to `reports/action_comparison/` and append-only
`experiments.csv`. Saved heads: `models/action_<name>.pkl`. The previous SigLIP
artifact is not overwritten. Set `FORTE_ACTION_MODEL=r3d-18` or
`FORTE_ACTION_MODEL=xclip-base-patch32` with `FORTE_ANALYZER=real` to enable the new
warm serving adapter. Its outputs use the existing API contract and legal templates;
uploaded videos are analyzed directly and never feature-cached.

No ground truth exists for ordinary actions, so action-recognition precision/recall
cannot be measured here. Violation metrics measure the end-to-end sequence. Type
classes represented by just one training video are unseen when that video is held
out; a supervised type head cannot learn them in that fold. Any improvement relative
to SigLIP includes the changed decision pipeline, not just a different backbone.
