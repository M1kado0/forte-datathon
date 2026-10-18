# Порог выбираем по recall 1.0, потом минимум ложных срабатываний

Банк считает пропуск нарушения недопустимым, а ложные срабатывания только нежелательными (FAQ 16). Поэтому порог вердикта выбираем на leave-one-video-out CV так: сначала все 21 видео с нарушениями (22 метки после склейки дублей) должны получить `violation`, потом из таких порогов берём тот, при котором меньше всего видео без нарушений получают `violation`. Метрику "accuracy" и F1 для выбора порога не используем. Leave-one-arrangement-out считаем рядом, но порог по нему не выбираем.

## Consequences

- На видео без нарушений будут ложные срабатывания. Это осознанная цена за recall, её показываем в `reports/metrics.json` и в README (TZ-8.6).
- Дубли Х-4/Х-47 и Х-5/Х-48 склеиваются до CV, иначе один и тот же ролик попадёт в train и в val.

## Threshold of the model fit on all videos (#63)

`models/window_clf.pkl` holds a model fit on all 40 videos, while the threshold comes from out-of-fold scores of 39-video models. We keep that threshold and add no calibration (no `CalibratedClassifierCV`, no ensemble of fold models). `uv run python -m src.models.train --nested` repeats the whole procedure inside each held-out group (threshold from LOVO on the other videos, a model fit on all of them, verdict on the held-out video) and logs one `experiments.csv` row:

- The two score scales sit together. Against the average of the fold models, the all-videos model has a median shift of +0.16 logit (mean absolute 0.58 per video), which is noise around zero, not a bias. Rank correlation between the two is 0.98.
- Nested video recall is 0.952, FP rate 0.947. The one miss is X-11, the weakest violation video. It sets the threshold, so the rule "lowest violation score" cannot catch it when it is held out. That is a property of the rule on 21 violation videos, not of the model fit.

Consequence: the 1.0 recall in `reports/metrics.json` is by construction. Expect a hidden violation weaker than X-11 to be missed.
