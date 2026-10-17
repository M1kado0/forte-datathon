# Forte Datathon '26: проверка кассовой дисциплины по видео

Локальный сервис ForteBank-кейса. Вы загружаете видео кассы (.mov), сервис отвечает: есть ли признаки возможных нарушений, в какую секунду смотреть, какой кадр это показывает и какой пункт норматива НБРК подходит.
Это подсказка проверяющему, а не юридический вывод. Решение принимает человек.

Все модели работают на нашей машине. Облачные AI-сервисы не используются ни для видео, ни для кадров, ни для описаний (TZ-7.3).
Загруженное видео удаляется после задачи и не идёт в обучение.

- Требования брифа и где их смотреть: [`docs/acceptance.md`](docs/acceptance.md)
- Термины (вердикт, нарушение, окно, расстановка, эвристика): [`GLOSSARY.md`](GLOSSARY.md)
- Публичный адрес UI и API и API-ключ жюри получает в форме подачи (TZ-8.1). Ключ в репозиторий не кладём.

## Запуск

Нужны [uv](https://docs.astral.sh/uv/), Node.js 20+ с npm и git. Python ставить не нужно: uv сам скачает версию из `.python-version` (сейчас 3.12, проект требует 3.11+). `make` нужен только для коротких команд: на Windows без WSL используйте строки `uv run` и `npm` рядом с ними.

| Система | uv | Node.js 20+ | make |
|---|---|---|---|
| Ubuntu | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | NodeSource или nvm (версия из `apt` часто ниже 20), проверка `node --version` | `sudo apt install make` |
| Mac | `brew install uv` или тот же скрипт | `brew install node` | уже есть |
| Windows | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `winget install OpenJS.NodeJS.LTS` | не нужен, берите строки без `make`; или WSL |

Остальные инструменты команды (gh, ngrok, драйвер NVIDIA): [`docs/SETUP.md`](docs/SETUP.md).

`uv sync` на Linux и Windows ставит сборку torch под CUDA 13.2 (`cu132`), это несколько гигабайт, даже на ноутбуке без GPU. На Mac ставится обычная сборка.

```bash
git clone https://github.com/M1kado0/forte-datathon.git   # или git@github.com:M1kado0/forte-datathon.git
cd forte-datathon
cp .env.example .env     # затем впишите длинный случайный API_KEY в .env
make setup               # зависимости Python и web/
make test                # быстрые детерминированные тесты
make app                 # собирает web и запускает сервис на http://localhost:8000
```

Без `make` (PowerShell, Git Bash, Mac, Ubuntu одинаково) берите строки из столбца «Без `make`» таблицы ниже, в том же порядке: зависимости, тесты, сборка web, запуск. Команды в таблице без `&&` и `cd`, поэтому работают и в Windows PowerShell 5.1.

В PowerShell вместо `cp` используйте `Copy-Item .env.example .env`.

`make app` не требует ни данных, ни весов: сервис стартует и отвечает по контракту. Веса и данные нужны для анализа реальных видео и для `make all`. Пока воркер на заглушке, сервис отдаёт один запасной тип (см. «Что уже подключено к API»).

| Что | `make` | Без `make` |
|---|---|---|
| Зависимости | `make setup` | `uv sync`, затем `npm ci --prefix web` (`make setup` ещё ставит `nbstripout`: `uv run nbstripout --install`) |
| Тесты | `make test` | `uv run pytest` |
| Только сборка web | `make web` | `npm run build --prefix web` |
| Сервис и UI на одном порту | `make app` | `npm run build --prefix web`, затем `uv run uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| Разработка с перезагрузкой | `make dev` | в одном окне `uv run uvicorn api.main:app --reload --port 8000`, в другом `npm run dev --prefix web` (UI на :5173, `/api` проксируется на :8000) |
| Форматирование и линтер | `make lint` | `uv run ruff check . --fix`, затем `uv run ruff format .` |
| Заглушка пайплайна (старый учебный стенд) | `make demo` | `uv run python -m src.pipeline --stub`, затем запуск сервиса |
| Веса моделей, один раз | `make weights` | `uv run python -m scripts.weights` |
| Проверка весов без интернета | `make weights-check` | `HF_HUB_OFFLINE=1 uv run python -m scripts.weights --check` (PowerShell: сначала `$env:HF_HUB_OFFLINE=1`) |
| Пересобрать метрики, типы и модель окон (нужны видео в `data/raw/videos/`) | `make all` | `uv run python -m src.rules.types_doc`, затем `uv run python -m src.pipeline` |
| Публичный адрес через ngrok | `make tunnel` | `ngrok http 8000` |
| Запасной туннель | `make tunnel-cloudflared` | `cloudflared tunnel --no-autoupdate --url http://localhost:8000` |
| Скриншоты UI | `make screenshots` | `npm run screenshots --prefix web` |
| Удалить сборку и метрики | `make clean` | удалить `web/dist`, `models/*.pkl`, `reports/metrics.json` |
| Симуляция жюри | `make jury-sim PUBLIC_URL=https://...` | `uv run python -m scripts.jury_sim --url https://...` |

Порт меняется так: `make app PORT=9000` или `--port 9000`. Для `make dev` порт 8000 фиксирован: прокси Vite смотрит на него (`web/vite.config.ts`).

### Веса и данные

Модели скачиваются один раз (`make weights`) в `models/weights`, после этого сервис работает с `HF_HUB_OFFLINE=1`, без сети.
Датасет организаторов в git не лежит: положите `open_labels.xlsx` в `data/raw/` и видео `.mov` в `data/raw/videos/`. У части файлов в имени кириллическая «Х» (`Х-10.mov`), поэтому имена лучше подставлять автодополнением.
Разметка `data/labels.csv` лежит в git. `make all` берёт её и видео: пересобирает справочник типов, метрики и модель окон. Разметку из xlsx пересобирает отдельная команда `uv run python -m src.data.labels` (её нет в `make all`), проверку разметки делает `uv run python -m src.data.check_labels`.

### Постоянный адрес для жюри

Машина с Ubuntu и RTX 5080 отдаёт сервис через Tailscale Funnel под systemd: [`docs/SERVING.md`](docs/SERVING.md), решение в [`docs/adr/0001-serving-on-ubuntu-rtx-5080.md`](docs/adr/0001-serving-on-ubuntu-rtx-5080.md).

## Использование

### UI

Откройте `http://localhost:8000`, выберите файл .mov и нажмите «Проверить видео». Страница показывает ход проверки, затем вердикт и список нарушений: тип, таймкод, описание, пункт норматива и кадры-доказательства.
UI ходит в `/api/ui/*`: те же обработчики, что у `/api/v1/*`, но без ключа.

### API

Все ручки `/api/v1/*` требуют заголовок `X-API-Key` со значением `API_KEY` из `.env`. Без ключа и с неверным ключом сервис отвечает 401.

```bash
export API_KEY=...   # значение из .env

# Проверка, что сервис жив (ключ не нужен)
curl localhost:8000/api/health

# Отправить видео: сразу вернётся job_id
curl -X POST localhost:8000/api/v1/analyze \
  -H "X-API-Key: $API_KEY" \
  -F "file=@data/raw/videos/X-1.mov"
# {"job_id":"3f2a..."}

# Спросить результат (повторять, пока status = processing)
curl localhost:8000/api/v1/jobs/3f2a... -H "X-API-Key: $API_KEY"
```

Ответ `jobs/{job_id}` строго по разделу 10 брифа, своих полей нет (пример сокращён: `…` вместо текста и кадра; с заглушкой тип всегда «Иное подозрительное действие»):

```json
{
  "job_id": "3f2a...",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Наличные на столе при обслуживании следующего клиента",
      "timecode_sec": 41.5,
      "description": "…",
      "evidence_frames": [{"timestamp_sec": 41.5, "image_base64": "…JPEG в base64…"}]
    }
  ]
}
```

- `status`: `processing`, `done` или `failed`. Пока задача не готова, `verdict` пуст, а `violations` пуст.
- `verdict`: `violation` тогда и только тогда, когда список `violations` не пуст; иначе `no_violation`.
- У каждого нарушения от одного до трёх кадров JPEG; `timecode_sec` совпадает с первым кадром с точностью до секунды.
- Загружать можно только `.mov`, иначе 422.
- Очередь живёт в памяти одного процесса: запускайте один процесс uvicorn, без `--workers`. После перезапуска сервиса `job_id` старых задач недействителен.

Чтобы не печатать кадры в терминал, уберите их через `jq` (его ставят отдельно):

```bash
curl -s localhost:8000/api/v1/jobs/3f2a... -H "X-API-Key: $API_KEY" \
  | jq 'del(.violations[].evidence_frames[].image_base64)'
```

В PowerShell те же запросы одной строкой (`curl.exe`, а не псевдоним `curl`):

```powershell
$env:API_KEY = "..."   # значение из .env
curl.exe -X POST localhost:8000/api/v1/analyze -H "X-API-Key: $env:API_KEY" -F "file=@data/raw/videos/X-1.mov"
curl.exe localhost:8000/api/v1/jobs/3f2a... -H "X-API-Key: $env:API_KEY"
```

Схемы ответов лежат в [`contracts/schemas.py`](contracts/schemas.py), тест контракта в [`tests/test_api.py`](tests/test_api.py).

## Как это устроено

```mermaid
flowchart LR
    U["UI или жюри"] -->|POST /api/v1/analyze| Q["Очередь в памяти и один воркер"]
    Q --> V["Декодирование видео<br/>src/data/video.py"]
    V --> A["Ветка A: что-то не так, в секунду X<br/>эмбеддинги DINOv2, классификатор окон"]
    V --> B["Ветка B: что именно произошло<br/>детекции и треки"]
    B --> H["Эвристики<br/>src/heuristics/"]
    A -->|скор окон, порог| M["Вердикт, таймкод"]
    H -->|тип, голос за вердикт| M
    M --> R["Справочник типов и норм<br/>src/rules/rules.yaml"]
    M --> E["Кадры-доказательства<br/>src/data/evidence.py"]
    R --> O["JSON по контракту: вердикт, нарушения"]
    E --> O
    O -->|GET /api/v1/jobs/id| U
```

| Часть | Что делает | Где |
|---|---|---|
| Очередь | Одна очередь в памяти, один фоновый воркер, анализатор подставляется. Исключение даёт `failed`, загруженный файл удаляется при любом исходе | [`api/jobs.py`](api/jobs.py), [`api/main.py`](api/main.py) |
| Ветка A | Эмбеддинги кадров DINOv2 → короткие окна внахлёст → логистическая регрессия. Говорит, что в окне что-то не так, и даёт таймкод. Порог выбран по recall 1.0, потом минимум ложных срабатываний ([ADR 0002](docs/adr/0002-threshold-recall-first.md)) | [`src/features/embeddings.py`](src/features/embeddings.py), [`src/models/windows.py`](src/models/windows.py), [`src/models/train.py`](src/models/train.py), [`src/models/analyze.py`](src/models/analyze.py) |
| Ветка B | Детекции людей и предметов (D-FINE-S), треки, деньги (OWLv2 по тексту) → признаки по зонам. В работе: пока в коде только проверка денег на кадрах ([`src/features/cash_check.py`](src/features/cash_check.py)), см. статус ниже. Без позы: готовые веса позы обучены на некоммерческих данных | [ADR 0003](docs/adr/0003-violation-taxonomy.md) (набор типов заменён [ADR 0005](docs/adr/0005-scenario-taxonomy.md)), источник про позу: [`docs/research/models.md`](docs/research/models.md), раздел 8 |
| Эвристики | Проверяют наблюдаемые факты (кто у кассы, где деньги, какие предметы, есть ли клиент), выбирают тип нарушения и могут поднять вердикт. Это наши проверки, к нормативу они отношения не имеют | [`src/heuristics/`](src/heuristics/) |
| Справочник | Тип нарушения → документ, пункт, цитата, ссылка на adilet и шаблон описания. Описание собирается из шаблона и наблюдённых фактов, юридические выводы не генерируются | [`src/rules/rules.yaml`](src/rules/rules.yaml), [`docs/RULES.md`](docs/RULES.md), [`docs/TYPES.md`](docs/TYPES.md) |
| Кадры-доказательства | От одного до трёх JPEG вокруг таймкода, в base64 | [`src/data/evidence.py`](src/data/evidence.py) |
| API и UI | FastAPI отдаёт `/api/v1`, `/api/ui` и собранный `web/dist` на одном порту; UI на React, Tailwind и токенах Forte | [`api/`](api/), [`web/`](web/) |

Если ни один тип не подходит, нарушение получает тип «Иное подозрительное действие» без точного пункта ([ADR 0005](docs/adr/0005-scenario-taxonomy.md)).
Классификатор типов не обучаем: на тип приходится слишком мало примеров, поэтому тип выбирают эвристики. По плану ([`docs/PLAN.md`](docs/PLAN.md)) запасной вариант, когда эвристики не уверены, это kNN по размеченным окнам; в коде его пока нет.

### Что уже подключено к API

Сборка идёт этапами, поэтому статус ниже важнее схемы: он показывает, что ответит сервис прямо сейчас.

| Часть | Статус | Тикет |
|---|---|---|
| Контракт, ключ, очередь, UI | работает; воркер пока вызывает заглушку и отдаёт один запасной тип | [#29](https://github.com/M1kado0/forte-datathon/issues/29) подключает `analyze(path)` к воркеру |
| Ветка A: `analyze(path)` | готова как функция и CLI (`uv run python -m src.models.analyze <видео>`), в воркер ещё не подключена | [#29](https://github.com/M1kado0/forte-datathon/issues/29) |
| Ветка B: детекции и треки | в работе | [#32](https://github.com/M1kado0/forte-datathon/issues/32) |
| Эвристики и их голос за вердикт | первые эвристики готовы ([#24](https://github.com/M1kado0/forte-datathon/issues/24)), остальные в работе | [#33](https://github.com/M1kado0/forte-datathon/issues/33), [#34](https://github.com/M1kado0/forte-datathon/issues/34) |

План: после слияния этих тикетов `/api/health` начнёт отдавать `model_loaded: true`, а типы нарушений будут приходить от эвристик. Сейчас `model_loaded` равен `false`.

## Числа

В README нет чисел, набранных руками. Все результаты (recall по видео, доля ложных срабатываний, медианная ошибка таймкода, порог, число размеченных видео) считает `make all`:

- [`reports/metrics.json`](reports/metrics.json) — последняя метрика модели окон;
- [`experiments.csv`](experiments.csv) — по строке на каждый запуск, только дописывается;
- [`docs/TYPES.md`](docs/TYPES.md) — типы нарушений и число меток, собирается из `data/labels.csv`;
- [`docs/DATA_CARD.md`](docs/DATA_CARD.md) — описание данных и расстановок.

Проверка сейчас идёт по схеме «одно видео за раз» (leave-one-video-out), порог выбирается по ней ([ADR 0002](docs/adr/0002-threshold-recall-first.md)). Соседние видео набора похожи друг на друга (одна и та же обстановка), поэтому такая оценка ненадёжна. Проверка на невиданной расстановке и по группам похожих видео — [#38](https://github.com/M1kado0/forte-datathon/issues/38), её результаты появятся в `reports/metrics.json`.

## Модели и лицензии

Берём только открытые модели с лицензией Apache-2.0, MIT или BSD. Ultralytics YOLO (AGPL-3.0) не используем. Полная таблица моделей, библиотек и принятых исключений: [`docs/MODELS.md`](docs/MODELS.md).

| Модель | Роль | Лицензия |
|---|---|---|
| DINOv2-small (`facebook/dinov2-small`) | эмбеддинги кадров, ветка A | Apache-2.0 |
| D-FINE-S (`ustc-community/dfine-small-coco`) | детекция людей и предметов, ветка B | Apache-2.0 |
| OWLv2 (`google/owlv2-base-patch16-ensemble`) | поиск денег по тексту, ветка B | Apache-2.0 |
| SigLIP-base (`google/siglip-base-patch16-224`) | запасные эмбеддинги, поиск по тексту; в сервисе пока не используется | Apache-2.0 |

Исключения из правила лицензий (CUDA-колёса, FFmpeg в opencv и другие), версии библиотек и лицензии туннелей: [`docs/MODELS.md`](docs/MODELS.md). Туннели в сервис не входят.

## Ограничения

Полный разбор ошибок с названиями видео будет в [#38](https://github.com/M1kado0/forte-datathon/issues/38) и [#42](https://github.com/M1kado0/forte-datathon/issues/42). Сейчас известно главное: порог выбран по recall 1.0, поэтому ложные срабатывания на видео без нарушений есть ([ADR 0002](docs/adr/0002-threshold-recall-first.md)), а модель окон их различает слабо. Долю ложных срабатываний смотрите в `reports/metrics.json` (`fp_rate`).
