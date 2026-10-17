# Brief checklist (source of truth). Built from `docs/Кейс_ForteBank.docx`, 17 Oct 12:45

Status: 🟥 not started · 🟨 in progress · 🟩 done and visible · ⬜ explained / out of scope (with reason)
IDs follow the brief's sections (TZ-4.x = section 4, TZ-F.x = FAQ). Quote the brief; never invent.

## How we are scored (section 9, 100 pts). Jury runs test videos through our API, scoring is automatic
| # | Criterion | Pts | What moves it |
|---|---|---|---|
| 1 | Detection: violation yes/no per whole video | 25 | window classifier + threshold tuned for recall (FAQ 16) |
| 2 | Localization: timecode + relevant evidence frame | 15 | onset of the above-threshold run for lasting states, peak for short actions; chosen by median \|Δt\| on LOVO |
| 3 | Classification: event type | 15 | our own type taxonomy from the 40 videos |
| 4 | Rule grounding: exact source, no invented links | 15 | verified НБРК/adilet table, quotes checked by a human |
| 5 | Robustness: unseen videos, other camera angle | 10 | angle-free features, augmentation, own recordings |
| 6 | Product: UI, reliable upload/processing, clear result | 10 | one screen, status, states |
| 7 | Tech: architecture, practicality, locality | 10 | README architecture, offline weights |

## Requirements
| ID | Requirement (brief) | Type | Owner | Ticket | Where the jury sees it | Status |
|---|---|---|---|---|---|---|
| TZ-4.1 | Пользователь открывает веб-интерфейс и загружает видеофайл | UI | P1 | T01, T07 | main screen | 🟥 |
| TZ-4.2 | Сервис запускает анализ и отображает статус обработки | UI | P1 | T01, T07 | progress state | 🟥 |
| TZ-4.3 | Результат: «Нарушение не обнаружено» / «Обнаружены признаки возможных нарушений» | UI+API | P1 | T01 | result header, `verdict` | 🟥 |
| TZ-4.4 | Для каждого нарушения (их может быть несколько): тип, таймкод, описание | model | P2, P3 | T03, T06, T15 | result list, `violations[]` | 🟥 |
| TZ-4.5 | Один или несколько кадров-доказательств из фрагмента | model | P3 | T08 | frame gallery, `evidence_frames` | 🟥 |
| TZ-4.6 | Кратко: что видно на видео и почему это может указывать на нарушение | model | P4 | T09 | `description` | 🟥 |
| TZ-4.7 | Правило и нормативный источник: название документа, пункт/раздел | rules | P4 | T05, T09 | result card + `docs/RULES.md` | 🟥 |
| TZ-4.8 | Сервис доступен через API (раздел 10) | API | P1 | T01 | `/api/v1/*` | 🟥 |
| TZ-5.1 | Фронтенд + сервер; результат без ручного запуска кода | app | P1 | T01, T10 | public URL | 🟥 |
| TZ-5.2 | Обнаружение визуально наблюдаемых действий | model | P2 | T03 | `violations[]` | 🟥 |
| TZ-5.6 | Обоснование без неподтверждённых предположений | rules | P4 | T09 | template descriptions only | 🟥 |
| TZ-6.1 | Найти и изучить акты РК (НБРК, adilet.zan.kz); указать использованные источники | rules | P4 | T05 | `docs/RULES.md` | 🟥 |
| TZ-6.2 | Показать связь: источник → предполагаемое нарушение → фрагмент видео | rules | P4 | T05, T09 | result card | 🟥 |
| TZ-7.1 | Не утверждать нарушение, если нужны невидимые данные (проводка, сверка) | rules | P4 | T05 | types list excludes them | 🟥 |
| TZ-7.2 | Результат = подсказка для человека, не заключение | UI | P1 | T07 | one line under verdict | 🟥 |
| TZ-7.3 | Все модели внутри нашего сервиса; облачные AI API запрещены (видео, кадры, описания) | tech | P1 | all | README, no outbound calls | 🟥 |
| TZ-7.4 | После загрузки весов интернет не нужен | tech | P3 | T02 | `make weights`, offline test | 🟥 |
| TZ-8.1 | Адрес UI, адрес API, API-ключ для жюри | deliverable | P3 | T10 | submission form | 🟥 |
| TZ-8.2 | Инструкция по запуску и использованию | deliverable | P4 | T11 | README | 🟥 |
| TZ-8.3 | Архитектура, подход, модели | deliverable | P4 | T11 | README | 🟥 |
| TZ-8.4 | Нормативные источники и принцип сопоставления правил с видео | deliverable | P4 | T05 | `docs/RULES.md` | 🟥 |
| TZ-8.5 | Перечень распознаваемых типов нарушений с описанием | deliverable | P4 | T04 | `docs/TYPES.md` | 🟥 |
| TZ-8.6 | Ограничения и известные случаи ошибок | deliverable | P2 | T12 | README | 🟥 |
| TZ-10.1 | Ключ в заголовке `X-API-Key`; без верного ключа запрос отклоняется | API | P1 | T01 | 401 | 🟥 |
| TZ-10.2 | `POST /api/v1/analyze`, .mov в поле `file` → `job_id` | API | P1 | T01 | curl | 🟥 |
| TZ-10.3 | `GET /api/v1/jobs/{job_id}` → `processing/done/failed` + результат по схеме | API | P1 | T01 | curl | 🟥 |
| TZ-10.4 | `evidence_frames`: 1–3 JPEG в base64, `timestamp_sec` | API | P3 | T08 | JSON | 🟥 |
| TZ-10.5 | Результат за ≤ 5 минут, иначе видео не засчитано | API | P3 | T10 | latency log, 5 parallel jobs | 🟥 |
| TZ-F.1 | Лица не распознавать и не идентифицировать (FAQ 29) | rule | all | all | person boxes only | 🟥 |
| TZ-F.2 | Звук не использовать (FAQ 28) | rule | all | all | — | 🟥 |
| TZ-F.3 | Только open source с лицензией для коммерческого использования (FAQ 24, 35) | rule | P3 | T02 | `docs/MODELS.md` license column | 🟥 |
| TZ-F.4 | Публичный URL обязателен (FAQ 23) | rule | P3 | T10 | tunnel URL | 🟥 |
| TZ-F.5 | Тестовые видео не для обучения, не публиковать, не хранить (раздел 10, FAQ 6) | rule | P1 | T01 | uploads deleted after job | 🟥 |
| TZ-F.6 | Ложных срабатываний минимум, пропуск недопустим (FAQ 16) | model | P2 | T06 | threshold note in `reports/metrics.json` | 🟥 |
| TZ-F.7 | Конкретный кадр и таймкод (FAQ 17) | model | P2 | T06 | `timecode_sec` = onset (states) or peak (actions), first evidence frame at the same second | 🟥 |
| TZ-F.8 | Финальное видео снято под другим углом (FAQ 4) | model | P2 | T03 | leave-one-angle-out metrics next to LOVO | 🟥 |
| TZ-F.9 | В тесте возможны типы, которых нет в обучении (FAQ 9) | model | P2, P3 | T06, T15 | fallback type | 🟥 |
| TZ-F.10 | Дедлайн 12:00 (FAQ 33) | rule | P4 | — | submit by 11:00 | 🟥 |
| TZ-P.1 | Питч только для шортлиста: 5 мин + 2 мин вопросы, презентация по желанию (FAQ 19) | pitch | P4 | T14 | — | 🟥 |

## Checkpoints
- 17:00: every row has an owner and a ticket; public URL answers `/api/v1/analyze`
- 00:00: no 🟥 rows
- 06:00: all 🟩 or ⬜ with a reason
