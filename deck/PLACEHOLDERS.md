# Deck template — 6 slides + appendix (P4)

Base: `forte-presentation.pptx` from forte.kz/docs (or the Claude Slides artifact with Forte tokens).
16:9 · white slides, magenta accents · ONE Dark Magenta → Coral hero slide (slide 1) · Open Sans.
Every chart title is a takeaway. Charts = grey + one magenta accent.

**Banned slides:** problem statement · tech stack · progress · team bios · "future of AI".

---

## 1 · Title (hero gradient) — 5 s
- Forte logo (official, not recoloured)
- **{{PRODUCT_NAME}}**
- {{who}} получает {{what}} за {{how fast}}
- small: команда {{TEAM_NAME}} · DATATHON '26
- (only if organizers require a problem line) Задача: {{one line}}

## 2 · Решение вживую — 25 s
- Full-width screenshot `reports/screens/desktop-3-result.png` (or 10–15 s silent looping clip)
- Callout ①: «Вводим {{X}}»  · Callout ②: «Видим {{result}} и почему» · Callout ③: «{{action}}»

## 3 · Сигнал — 20 s
- Title = takeaway: {{"Просрочка 60+ дней — риск в 2,8× выше"}}
- One chart (from `/api/insights` hero card / `reports/figures/`)
- **{{BIG NUMBER}}**
- Почему это работает: {{one plain sentence}}

## 4 · ТЗ: всё сделано ✅ — 15 s
| ✅ | Требование ТЗ | Где посмотреть |
|---|---|---|
| ✅ | {{TZ-1.1}} | {{экран / файл}} |
| ✅ | {{TZ-1.2}} | {{…}} |
(copy from `docs/acceptance.md`; if > 7 rows, group them)

## 5 · Эффект и доверие — 15 s
- **{{₸ / время сэкономлено}}** — small print: допущения {{…}}
- Проверка: «держится на последнем периоде / holdout, не утечка» {{metric before → after}}
- (optional) Нишевый метод: {{name}} ({{citation}}) — {{before → after}}

## 6 · Финал — 10 s
- {{memorable line}}
- QR → {{live link}}  ·  «Попробуйте сами»
- {{TEAM_NAME}} · {{names}}

---

## Appendix (Q&A only, not presented)
- A1 Данные: data card (rows, period, target rate, cleaning)
- A2 Валидация и метрики: split scheme, metrics table, calibration plot
- A3 Модель: features, SHAP global, why this model
- A4 Ограничения и следующие шаги

## Dummy content for tonight's 90-s rehearsal
1. «Мы — команда Pickaxe. Forte Риск-скоринг: кредитный специалист получает решение по клиенту за 5 секунд.»
2. «Сотрудник вводит ID клиента → видит вероятность дефолта 47% и пять причин простыми словами → рекомендация: запросить залог.»
3. «Мы нашли: просрочка больше 60 дней повышает риск в 2,8 раза. Это держится на отложенной выборке.»
4. «Всё из ТЗ сделано: модель, объяснения, интерфейс, README, демо по ссылке.»
5. «Эффект — 230 тысяч тенге предотвращённых потерь на одного клиента высокого риска.»
6. «Двигаем один ползунок — клиент из „отказать" становится „одобрить". Ссылка в QR — попробуйте сами.»
