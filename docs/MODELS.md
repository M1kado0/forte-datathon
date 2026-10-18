# Модели и библиотеки: лицензии

Правило: только Apache-2.0, MIT, BSD (коммерческое использование). Ultralytics YOLO (AGPL-3.0) не используем.
Веса лежат в `models/weights/<имя>` (`make weights`, один раз), дальше всё грузится с `HF_HUB_OFFLINE=1`.
Источник лицензии весов: поле `license` в карточке HF; лицензии библиотек: метаданные установленных пакетов (`importlib.metadata`).

## Модели

| Имя | HF-репозиторий | Роль | Лицензия | Размер весов |
|---|---|---|---|---|
| dinov2-small | `facebook/dinov2-small` | эмбеддинги кадров | Apache-2.0 | 88 MB |
| siglip-base | `google/siglip-base-patch16-224` | эмбеддинги, zero-shot по тексту | Apache-2.0 | 813 MB |
| dfine-small | `ustc-community/dfine-small-coco` | детекция людей и объектов (D-FINE-S) | Apache-2.0 | 41 MB |
| owlv2-base | `google/owlv2-base-patch16-ensemble` | open-vocabulary детекция (OWLv2) | Apache-2.0 | 620 MB |

## Библиотеки

| Пакет | Версия | Лицензия | Примечание |
|---|---|---|---|
| torch | 2.14.1 (cu132 на Linux/Windows, PyPI на macOS) | BSD-3-Clause и др. (в метаданных: Apache-2.0, BSD-2/3, BSL-1.0, MIT) | |
| torchvision | 0.29.1 | BSD | |
| transformers | 5.20.1 | Apache-2.0 | |
| huggingface-hub | 2.2.0 | Apache-2.0 | |
| safetensors | 0.8.0 | Apache-2.0 | |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | колёса содержат FFmpeg под LGPLv2.1 (динамическая линковка, коммерческое использование допускается) |
| sentencepiece | 0.2.2 | Apache-2.0 | токенайзер SigLIP |
| protobuf | 7.36.2 | BSD-3-Clause | нужен токенайзеру SigLIP |
| numpy | 2.5.4 | BSD-3-Clause | |
| scipy | 1.18.1 | BSD-3-Clause | `linear_sum_assignment` в ByteTrack (`src/features/bytetrack.py`) |
| joblib | 1.6.0 | BSD-3-Clause | параллельный nested-LOVO в `src/models/train.py`, транзитивно через scikit-learn |
| threadpoolctl | 3.7.0 | BSD-3-Clause | один поток BLAS на процесс в nested-LOVO, транзитивно через scikit-learn |
| pillow | 12.3.0 | MIT-CMU (HPND, разрешающая, близка к MIT) | транзитивно через transformers, напрямую не объявлена |
| nvidia-* (15 колёс CUDA: cublas, cudnn, nccl и др.) | 13.2.x | NVIDIA Software License / CUDA EULA (проприетарная, redistribution разрешён вместе с приложением) | транзитивно через torch cu132; на macOS не ставятся |

## Исключения из правила лицензий (приняты)

Эти пакеты не входят в список Apache-2.0 / MIT / BSD, но без них нет ни CUDA, ни декодирования:

1. nvidia-* (CUDA-колёса torch): проприетарная лицензия NVIDIA, обязательны для GPU.
2. opencv-python-headless: сам код Apache-2.0, но колёса содержат FFmpeg под LGPLv2.1.
3. pillow: HPND (MIT-CMU), по сути разрешающая.
4. torch: в метаданных перечислена и BSL-1.0 (Boost) для части кода.

Приняты как исключения, запись есть в AGENTS.md (раздел Code rules, Licenses).

## Системные инструменты (вне Python-зависимостей, в сервис не входят)

Ставятся на машину сервинга системным пакетным менеджером, в `src/` и `api/` их нет; правило лицензий относится к моделям и пакетам `uv`.

| Инструмент | Роль | Лицензия |
|---|---|---|
| Tailscale (клиент, Funnel) | постоянный публичный адрес для жюри | BSD-3-Clause (клиент); координационный сервер Tailscale проприетарный |
| ngrok (агент) | быстрый запасной туннель `make tunnel`, не для итогового адреса | проприетарный, закрытый код; вспомогательный, жюри не видит |
| cloudflared | запасной туннель `make tunnel-cloudflared` | Apache-2.0 |
| ffmpeg | `make jury-sim`: сборка длинных клипов для стресса | LGPL-2.1+ (GPL в зависимости от сборки); отдельный бинарник, не линкуется |
