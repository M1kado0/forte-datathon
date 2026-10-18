# Action-first comparison

Sparse violation point labels do not measure generic action-recognition accuracy. Dense proposals preserve static scenes. Type classes absent from a training fold cannot be learned. Historical baselines differ in both features and decision pipeline.

| Model | Evaluation | Recall | FP rate | Time error (s) | Type accuracy |
|---|---|---:|---:|---:|---:|
| r3d-18 | lovo | 1.0000 | 0.9474 | 3.4333 | 0.0000 |
| r3d-18 | nested | 1.0000 | 0.9474 | 3.4333 | 0.0000 |
| r3d-18 | self | 1.0000 | 1.0000 | 8.0000 | 0.1667 |
| xclip-base-patch32 | lovo | 1.0000 | 0.9474 | 3.0000 | 0.0000 |
| xclip-base-patch32 | nested | 0.9048 | 0.9474 | 3.0000 | 0.0000 |
| xclip-base-patch32 | self | 1.0000 | 1.0000 | 3.5000 | 0.0000 |

Selected on organizer nested CV before self evaluation: `r3d-18`.

Only supervised violation/type heads are fitted; frozen action models receive no violation prompts. Times are score-peak centres, not claimed action onset annotations.
