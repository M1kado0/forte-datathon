# Data card (P3, 11:00–11:30)

| Item | Value |
|---|---|
| Files | {{name · rows · cols · size}} |
| Grain (one row = ?) | |
| Period covered | |
| Target & base rate | |
| ID columns / repeat clients? | |
| Date columns → split scheme | time-based / grouped / random (why) |
| Missingness (top columns) | |
| Suspicious / leaky columns | {{anything computed after the outcome}} |
| Join keys between files | |
| External data allowed? | {{answer from QUESTIONS.md}} |

Quick report: `uv run python -c "from skrub import TableReport; import pandas as pd; TableReport(pd.read_csv('data/raw/X.csv')).open()"`
