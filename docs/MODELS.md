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
| numpy | 2.5.4 | BSD-3-Clause | |
| pillow | 12.3.0 | MIT-CMU (HPND, разрешающая, близка к MIT) | приходит транзитивно через transformers |
