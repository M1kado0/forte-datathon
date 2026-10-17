# Модели и железо для видео-пайплайна (research B)

2026-10-17, 14:16 · Статус: готово (см. список НЕ ПРОВЕРЕНО в конце раздела 7) · Все страницы открыты через playwright-cli 17.10.2026.

**TL;DR**
1. Эмбеддинги: DINOv2-small/base (с registers и без) и SigLIP / SigLIP 2 base имеют Apache-2.0 и на код, и на веса.
2. Детекция: RT-DETR(v2) и D-FINE есть в transformers, веса на HF под Apache-2.0; D-FINE-S 3.49 мс, RT-DETR-R18 217 FPS (T4, TensorRT FP16).
3. Не брать: VideoMAE (веса CC-BY-NC-4.0), V-JEPA 1 (код CC-BY-NC-4.0), Qwen2.5-VL-3B (только некоммерческое), Ultralytics (AGPL-3.0).
4. RTX 5070: PyTorch ≥ 2.7 с CUDA 12.8; сейчас стабильный 2.14.1 на Windows идёт только с cu126 и cu132, для Blackwell нужен индекс cu128 (torch ≤ 2.11) или cu132.
5. Видео: opencv-python декодирует из колеса (FFmpeg LGPLv2.1 внутри всех колёс); PyAV тоже со встроенным FFmpeg, требует Python ≥ 3.12.

## 1. Модели-кандидаты

Размер и лицензия весов взяты из HF API `https://huggingface.co/api/models/<id>?blobs=true` (поля `cardData.license`, `safetensors.total`, размер файлов). Офлайн: для моделей с HF работает переменная `HF_HUB_OFFLINE=1`: «If set, no HTTP calls will be made to the Hugging Face Hub. If you try to download files, only the cached files will be accessed» ([документация huggingface_hub](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)). Для весов вне HF (download.openmmlab.com, storage.googleapis.com, GitHub releases, Google Drive) офлайн значит: заранее положить файл в `models/weights` и передавать локальный путь.

| Модель | Роль | Лицензия кода (источник) | Лицензия весов (источник) | Размер | Формат, офлайн | Py 3.12 | Годится? |
|---|---|---|---|---|---|---|---|
| DINOv2-small `facebook/dinov2-small` | эмбеддинги | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/facebookresearch/dinov2/HEAD/LICENSE)); README: «DINOv2 code and model weights are released under the Apache License 2.0» ([README](https://raw.githubusercontent.com/facebookresearch/dinov2/HEAD/README.md)) | apache-2.0 ([HF](https://huggingface.co/facebook/dinov2-small)) | 22.06 M, safetensors 88.2 MB | transformers; HF кэш, офлайн да | transformers: requires_python ≥3.10, классификатор 3.12 ([PyPI](https://pypi.org/pypi/transformers/json)) | да |
| DINOv2-base `facebook/dinov2-base` | эмбеддинги | как выше | apache-2.0 ([HF](https://huggingface.co/facebook/dinov2-base)) | 86.58 M, 346.3 MB | transformers | да | да |
| DINOv2-small с registers `facebook/dinov2-with-registers-small` | эмбеддинги | как выше; статья [arXiv 2309.16588](https://arxiv.org/abs/2309.16588) | apache-2.0 ([HF](https://huggingface.co/facebook/dinov2-with-registers-small)) | 22.06 M, 88.3 MB | transformers | да | да |
| DINOv2-base с registers `facebook/dinov2-with-registers-base` | эмбеддинги | как выше | apache-2.0 ([HF](https://huggingface.co/facebook/dinov2-with-registers-base)) | 86.58 M, 346.4 MB | transformers | да | да |
| (внимание) XRay-DINO, Cell-DINO в том же репо | не наша роль | Cell-DINO: «CC BY NC» | «FAIR Noncommercial Research License» ([README](https://raw.githubusercontent.com/facebookresearch/dinov2/HEAD/README.md)) | | | | нет |
| SigLIP base `google/siglip-base-patch16-224` | эмбеддинги, zero-shot по тексту | Apache-2.0 ([big_vision LICENSE](https://raw.githubusercontent.com/google-research/big_vision/HEAD/LICENSE)) | apache-2.0 ([HF](https://huggingface.co/google/siglip-base-patch16-224)) | 203.16 M (картинка + текст), 812.7 MB | transformers | да | да |
| SigLIP 2 base `google/siglip2-base-patch16-224` | эмбеддинги, zero-shot | Apache-2.0 (big_vision, как выше) | apache-2.0 ([HF](https://huggingface.co/google/siglip2-base-patch16-224)) | 375.19 M, 1500.8 MB | transformers | да | да |
| OpenCLIP ViT-B/32 `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | эмбеддинги, zero-shot | MIT ([open_clip LICENSE](https://raw.githubusercontent.com/mlfoundations/open_clip/HEAD/LICENSE), текст MIT «Permission is hereby granted…») | mit ([HF](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K)); датасет весов LAION-2B (в имени модели `laion2B`). В карточке: «Any deployed use case of the model - whether commercial or not - is currently out of scope» | 151.28 M, 605.2 MB | open_clip (`open_clip_model.safetensors`); open-clip-torch 3.3.0 ([PyPI](https://pypi.org/pypi/open-clip-torch/json)) | классификатор 3.12 есть | осторожно (карточка против deployed use) |
| OpenCLIP ViT-B/16 `laion/CLIP-ViT-B-16-laion2B-s34B-b88K` | то же | MIT | mit ([HF](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K)), LAION-2B | 598.5 MB (число параметров в API нет) | open_clip | да | осторожно |
| OpenAI CLIP `openai/clip-vit-base-patch32` | эмбеддинги | MIT ([openai/CLIP LICENSE](https://raw.githubusercontent.com/openai/CLIP/HEAD/LICENSE)) | поле license в карточке пустое ([HF API](https://huggingface.co/api/models/openai/clip-vit-base-patch32)) | 605.2 MB (pytorch_model.bin) | transformers | да | нет (лицензия весов не указана) |
| RT-DETR R18 `PekingU/rtdetr_r18vd` | детекция людей и COCO-объектов | Apache-2.0 ([RT-DETR LICENSE](https://raw.githubusercontent.com/lyuwenyu/RT-DETR/HEAD/LICENSE)) | apache-2.0 ([HF](https://huggingface.co/PekingU/rtdetr_r18vd)) | 20.21 M, 80.9 MB | transformers | да | да |
| RT-DETR R50 `PekingU/rtdetr_r50vd` | детекция | как выше | apache-2.0 ([HF](https://huggingface.co/PekingU/rtdetr_r50vd)) | 43.02 M, 172.2 MB | transformers | да | да |
| RT-DETRv2 R18 `PekingU/rtdetr_v2_r18vd` | детекция | как выше | apache-2.0 ([HF](https://huggingface.co/PekingU/rtdetr_v2_r18vd)) | 20.21 M, 80.9 MB | transformers | да | да |
| D-FINE-S `ustc-community/dfine-small-coco` | детекция | Apache-2.0 ([D-FINE LICENSE](https://raw.githubusercontent.com/Peterande/D-FINE/HEAD/LICENSE)) | apache-2.0 ([HF](https://huggingface.co/ustc-community/dfine-small-coco)) | 10.35 M, 41.5 MB | transformers | да | да |
| D-FINE-N `ustc-community/dfine-nano-coco` | детекция | как выше | apache-2.0 ([HF](https://huggingface.co/ustc-community/dfine-nano-coco)) | 3.80 M, 15.3 MB | transformers | да | да |
| DEIM (D-FINE-S/N) | детекция | Apache ([DEIM LICENSE](https://raw.githubusercontent.com/ShihuaHuang95/DEIM/HEAD/LICENSE): «DEIM is licensed under the Apache License») | отдельной лицензии весов нет; чекпойнты на Google Drive ([README](https://raw.githubusercontent.com/ShihuaHuang95/DEIM/HEAD/README.md)). Репо на HF `Intellindust/DEIM_D-FINE_S_COCO` не открылось: НЕ ПРОВЕРЕНО | S: 10M, N: 4M (README) | свой код репо, .pth | НЕ ПРОВЕРЕНО | запасной |
| RTMDet (tiny/s) | детекция | Apache-2.0 ([mmdetection LICENSE](https://raw.githubusercontent.com/open-mmlab/mmdetection/HEAD/LICENSE)) | отдельной лицензии весов на странице не видел; веса на download.openmmlab.com ([README](https://raw.githubusercontent.com/open-mmlab/mmdetection/HEAD/configs/rtmdet/README.md)): НЕ ПРОВЕРЕНО | tiny 4.8 M, s 8.89 M | mmdet (тяжёлые зависимости mmcv) | НЕ ПРОВЕРЕНО | нет (зависимости) |
| YOLOX-s | детекция | Apache-2.0 ([YOLOX LICENSE](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/HEAD/LICENSE)) | отдельной лицензии весов не видел; веса в GitHub releases, есть `yolox_s.onnx` ([rtmlib README](https://raw.githubusercontent.com/Tau-J/rtmlib/HEAD/README.md)): НЕ ПРОВЕРЕНО | 9.0 M ([README](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/HEAD/README.md)) | ONNX через rtmlib | rtmlib да | запасной |
| RTMPose-t/s/m (через rtmlib) | поза тела | rtmlib Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/Tau-J/rtmlib/HEAD/LICENSE)); mmpose Apache-2.0: «This project is released under the Apache 2.0 license» ([README](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/README.md)) | зеркало весов `Tau-J/RTMPose` на HF: apache-2.0 ([HF API](https://huggingface.co/api/models/Tau-J/RTMPose)). Обучены на 7 датасетах (Body7: AI Challenger, COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18) ([RTMPose README](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/projects/rtmpose/README.md)). Лицензии этих датасетов: НЕ ПРОВЕРЕНО | t 3.34 M, s 5.47 M, m 13.59 M | ONNX (zip), onnxruntime; детектор по умолчанию YOLOX (HumanArt) | rtmlib 0.0.16: requires_python ≥3.10, wheel py3-none-any ([PyPI](https://pypi.org/pypi/rtmlib/json)); onnxruntime 1.31.0 cp312 ([PyPI](https://pypi.org/pypi/onnxruntime/json)) | да |
| MediaPipe Pose / Hand Landmarker | поза, руки | Apache-2.0 ([LICENSE](https://raw.githubusercontent.com/google-ai-edge/mediapipe/HEAD/LICENSE)) | НЕ ПРОВЕРЕНО: лицензия в PDF model card ([BlazePose](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf), [Hand](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf)), PDF не открылся в playwright-cli. Страница задачи: контент CC-BY-4.0, код Apache-2.0 ([pose_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)) | .task, float16; вход 224 и 256 px | файлы .task с storage.googleapis.com, локально | mediapipe 1.1.1, классификатор 3.12, колёса win_amd64, macOS arm64, manylinux ([PyPI](https://pypi.org/pypi/mediapipe/json)) | да, если подтвердить лицензию весов |
| ViTPose-base `usyd-community/vitpose-base-simple` | поза | Apache-2.0 ([ViTPose LICENSE](https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/HEAD/LICENSE)) | apache-2.0 ([HF](https://huggingface.co/usyd-community/vitpose-base-simple)) | 85.91 M, 343.7 MB | transformers | да | запасной |
| ViTPose ONNX `JunkyByte/easy_ViTPose` | поза (rtmlib) | | поле license пустое ([HF API](https://huggingface.co/api/models/JunkyByte/easy_ViTPose)) | vitpose-b 360.1 MB | ONNX | | нет |
| VideoMAE-base `MCG-NJU/videomae-base` | видео | CC-BY-NC-4.0 ([LICENSE](https://raw.githubusercontent.com/MCG-NJU/VideoMAE/HEAD/LICENSE)) | cc-by-nc-4.0 ([HF](https://huggingface.co/MCG-NJU/videomae-base)); kinetics-версия тоже cc-by-nc-4.0 ([HF](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)) | 94.21 M, 376.9 MB | transformers | да | **нет** |
| X-CLIP base/32 `microsoft/xclip-base-patch32` | видео, zero-shot | MIT ([VideoX LICENSE](https://raw.githubusercontent.com/microsoft/VideoX/HEAD/LICENSE)) | mit ([HF](https://huggingface.co/microsoft/xclip-base-patch32)); карточка: «trained fully-supervised on Kinetics-400» | 196.59 M, 786.4 MB | transformers | да | нет (веса на Kinetics, правило задачи) |
| V-JEPA 1 | видео | CC-BY-NC-4.0 ([jepa LICENSE](https://raw.githubusercontent.com/facebookresearch/jepa/HEAD/LICENSE)) | НЕ ПРОВЕРЕНО отдельно | | | | **нет** |
| V-JEPA 2 ViT-L `facebook/vjepa2-vitl-fpc64-256` | видео | MIT ([LICENSE](https://raw.githubusercontent.com/facebookresearch/vjepa2/HEAD/LICENSE)); README: «The majority of V-JEPA 2 is licensed under MIT», три файла аугментаций под Apache-2.0 ([README](https://raw.githubusercontent.com/facebookresearch/vjepa2/HEAD/README.md)) | mit ([HF](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)). Данные: «over 1 million hours of internet video» ([arXiv 2506.09985](https://arxiv.org/abs/2506.09985)); состав датасета (есть ли Kinetics): НЕ ПРОВЕРЕНО | 325.97 M, 1303.9 MB | transformers | да | осторожно, тяжёлая |
| Qwen2.5-VL-7B-Instruct | stretch VLM | Apache-2.0 ([Qwen2.5-VL LICENSE](https://raw.githubusercontent.com/QwenLM/Qwen2.5-VL/HEAD/LICENSE)) | apache-2.0 ([HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)) | 8.29 B, 5 файлов ≈ 16.6 GB | transformers | да | stretch |
| Qwen2.5-VL-7B-Instruct-AWQ | stretch VLM | как выше | apache-2.0 ([HF](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-AWQ)) | 2 файла: 3982.2 + 2941.8 MB ≈ 6.9 GB | transformers + AWQ | да | stretch, на 8 GB VRAM впритык (оценка) |
| Qwen2.5-VL-3B-Instruct | VLM | | license_name `qwen-research`; LICENSE: «FOR NON-COMMERCIAL PURPOSES ONLY» ([LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE)) | 3.75 B | | | **нет** |
| Qwen3-VL-2B-Instruct | лёгкая VLM | НЕ ПРОВЕРЕНО (LICENSE репо не открывал) | apache-2.0 ([HF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)) | 2.13 B, 4255.1 MB | transformers | да | лёгкий stretch |
| Qwen3-VL-4B-Instruct | VLM | НЕ ПРОВЕРЕНО | apache-2.0 ([HF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)) | 4.44 B, ≈ 8.9 GB | transformers | да | stretch |
| SmolVLM2-2.2B-Instruct | лёгкая VLM | НЕ ПРОВЕРЕНО | apache-2.0 ([HF](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct)) | 2.25 B, ≈ 9.0 GB | transformers | да | лёгкий stretch |
| SmolVLM2-500M-Video-Instruct | лёгкая VLM | НЕ ПРОВЕРЕНО | apache-2.0 ([HF](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)) | 0.51 B, 2030 MB; есть ONNX (q4 ≈ 229 MB) | transformers или ONNX | да | лёгкий stretch |
| Ultralytics YOLO | детекция | AGPL-3.0 ([LICENSE](https://raw.githubusercontent.com/ultralytics/ultralytics/HEAD/LICENSE)) | | | | | **исключён** |

Версии пакетов на PyPI 17.10.2026: torch 2.14.1, torchvision 0.29.1, transformers 5.20.1, timm 1.0.30 (Apache-2.0, классификатор 3.12), open-clip-torch 3.3.0, onnxruntime / onnxruntime-gpu 1.31.0 (MIT, requires_python ≥3.11, колёса cp312 для Windows), rtmlib 0.0.16, mediapipe 1.1.1, opencv-python 5.0.0.93. Источник: `https://pypi.org/pypi/<pkg>/json`.

## 2. Скорость

Официальных цифр для DINOv2, SigLIP, OpenCLIP, V-JEPA 2 и Qwen на CPU, M1 или RTX 40/50 не нашёл. Цифры ниже сняты на другом железе (T4, V100, GTX 1660 Ti, RTX 3090, i7-11700, Pixel 6), на RTX 5070 и M1 они не переносятся напрямую. Оценка на клип: 145 с × 4 fps = 580 кадров, батч 1, без декодирования и пред/постобработки. Это **оценка**, не замер.

| Модель | CPU/M1 | GPU | Источник | Оценка на клип 145 с @4 fps |
|---|---|---|---|---|
| DINOv2-small/base | нет данных | нет данных | | нет данных, замерить в T02 |
| SigLIP / SigLIP 2 base | нет данных | нет данных | | нет данных |
| OpenCLIP ViT-B | нет данных | нет данных | | нет данных |
| RT-DETR-R18 | нет данных | 217 FPS, T4, bs=1 (таблица FPS_bs=1) | [HF card](https://huggingface.co/PekingU/rtdetr_r18vd), [arXiv 2304.08069](https://arxiv.org/abs/2304.08069) | ≈ 2.7 с GPU (оценка) |
| D-FINE-S | нет данных | 3.49 мс, T4, bs=1, fp16, TensorRT 10.4.0 | [D-FINE README](https://raw.githubusercontent.com/Peterande/D-FINE/HEAD/README.md) | ≈ 2.0 с GPU (оценка) |
| D-FINE-N | нет данных | 2.12 мс, те же условия; «472 FPS on T4» | там же | ≈ 1.2 с GPU (оценка) |
| DEIM-S / DEIM-N | нет данных | 3.49 / 2.12 мс (условия замера в README не сверял: НЕ ПРОВЕРЕНО) | [DEIM README](https://raw.githubusercontent.com/ShihuaHuang95/DEIM/HEAD/README.md) | ≈ 2.0 / 1.2 с (оценка) |
| RTMDet-tiny / s | нет данных | TRT-FP16: tiny 0.98 мс RTX3090, 2.34 мс T4; s 1.22 / 2.96 мс | [RTMDet README](https://raw.githubusercontent.com/open-mmlab/mmdetection/HEAD/configs/rtmdet/README.md) | ≈ 1.4–1.7 с на T4 (оценка) |
| YOLOX-s | нет данных | 9.8 мс, V100, 640 px | [YOLOX README](https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/HEAD/README.md) | ≈ 5.7 с (оценка) |
| RTMPose-t / s / m (Body7, 256×192) | ORT, i7-11700: 3.20 / 4.48 / 11.06 мс | TRT-FP16, GTX 1660 Ti: 1.06 / 1.39 / 2.29 мс | [RTMPose README](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/projects/rtmpose/README.md); также «RTMPose-m … 90+ FPS on an Intel i7-11700 CPU and 430+ FPS on an NVIDIA GTX 1660 Ti GPU» | на одного человека: CPU t 1.9 с, m 6.4 с; GPU m 1.3 с. При 3 людях в кадре ×3 (оценка) |
| MediaPipe HandLandmarker (full) | Pixel 6 CPU 17.12 мс | Pixel 6 GPU 12.27 мс | [hand_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) | ≈ 9.9 с CPU (оценка, телефонный CPU) |
| MediaPipe Pose Landmarker | нет данных (таблицы бенчмарка на странице не нашёл) | нет данных | [pose_landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) | нет данных |
| ViTPose-base | нет данных | нет данных | | нет данных |
| Qwen2.5-VL-7B / Qwen3-VL / SmolVLM2 | нет данных | нет данных | | нет данных |

## 3. PyTorch на Blackwell + uv

Факты:
- PyTorch 2.7 «introduces support for NVIDIA's new Blackwell GPU architecture and ships pre-built wheels for CUDA 12.8» ([блог PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/)). Значит минимум torch 2.7.0 + cu128.
- Сейчас стабильная версия 2.14.1. Селектор на [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) для Windows + Pip даёт варианты CUDA 12.6, CUDA 13.2, Default. Команды:
  `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132` и `.../whl/cu126`; Default: `pip3 install torch torchvision`. Python рекомендован 3.10–3.14.
- Индекс cu128 ([download.pytorch.org/whl/cu128/torch/](https://download.pytorch.org/whl/cu128/torch/)) содержит колёса `cp312-win_amd64` для torch 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1, 2.10.0, 2.11.0. Новее 2.11.0 под cu128 нет.
- Индекс cu132 ([download.pytorch.org/whl/cu132/torch/](https://download.pytorch.org/whl/cu132/torch/)) содержит `cp312-win_amd64` для 2.12.0, 2.12.1, 2.13.0, 2.14.0, 2.14.1.
- Драйвер. CUDA 12.8 GA: Linux `>=570.26`, Windows `>=570.65` ([CUDA 12.8 release notes, таблица Toolkit Driver Version](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)). CUDA 13.2 GA: Linux `>=595.45.04`, Windows в таблице N/A ([CUDA 13.2 release notes](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-toolkit-release-notes/index.html)). Текущие release notes: «Existing CUDA 13.x applications run on drivers >=580 under CUDA minor version compatibility», ветка драйвера для CUDA 13.2 = R595 ([release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
- uv: «PyTorch would be installed from PyPI, which hosts CPU-only wheels for Windows and macOS, and GPU-accelerated wheels on Linux (targeting CUDA 13.0, as of PyTorch 2.11.0)». То есть простой `uv add torch` на Windows даёт CPU-сборку ([uv guide](https://docs.astral.sh/uv/guides/integration/pytorch/)).
- Пример из гайда uv с маркерами платформы (дословно, раздел про разные индексы для разных платформ; в гайде CUDA только для Linux):

```toml
[project]
name = "project"
version = "0.1.0"
requires-python = ">=3.14.0"
dependencies = [
  "torch>=2.11.0",
  "torchvision>=0.26.0",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu", marker = "sys_platform != 'linux'" },
  { index = "pytorch-cu130", marker = "sys_platform == 'linux'" },
]
torchvision = [
  { index = "pytorch-cpu", marker = "sys_platform != 'linux'" },
  { index = "pytorch-cu130", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

Адаптация под нас (рекомендация, не из гайда; маркеры той же формы, что и в разделе XPU гайда `sys_platform == 'win32' or sys_platform == 'linux'`): CUDA для win32 и linux, macOS из PyPI по умолчанию.

```toml
[tool.uv.sources]
torch = [
  { index = "pytorch-cu132", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
]
torchvision = [
  { index = "pytorch-cu132", marker = "sys_platform == 'win32' or sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cu132"
url = "https://download.pytorch.org/whl/cu132"
explicit = true
```

Вариант cu128 тот же, но с `url = "https://download.pytorch.org/whl/cu128"` и `torch>=2.7,<2.12`. На macOS без source-записи пакет берётся с PyPI (обычная сборка с MPS).

## 4. Декодирование видео

| Библиотека | Лицензия | H.264 .mov из коробки | Выборка каждого N-го кадра |
|---|---|---|---|
| opencv-python 5.0.0.93 (и -headless) | Apache-2.0 ([PyPI](https://pypi.org/pypi/opencv-python/json)). README: «All wheels ship with FFmpeg licensed under the LGPLv2.1» ([README](https://raw.githubusercontent.com/opencv/opencv-python/HEAD/README.md)); LGPL допускает коммерческое использование при динамической линковке (вывод, не цитата). Колёса abi3: macOS arm64, manylinux x86_64, Windows | да, FFmpeg внутри колеса (README) | `VideoCapture` + `grab()`/`CAP_PROP_POS_FRAMES`; страница docs.opencv.org закрыта Cloudflare-проверкой: НЕ ПРОВЕРЕНО |
| PyAV (`av` 19.0.1) | BSD-3-Clause ([PyPI](https://pypi.org/pypi/av/json)). Лицензия встроенного FFmpeg в README не указана: НЕ ПРОВЕРЕНО | да: «Binary wheels are provided on PyPI for Linux, macOS, and Windows with FFmpeg bundled»; «PyAV requires Python 3.12 or later» ([README](https://raw.githubusercontent.com/PyAV-Org/PyAV/HEAD/README.md)) | API для пропуска кадров в этой проверке не открывал: НЕ ПРОВЕРЕНО |
| decord 0.6.0 | Apache ([PyPI](https://pypi.org/pypi/decord/json)) | на PyPI колёса только `manylinux2010_x86_64` и `win_amd64` (py3-none), последняя загрузка 2021-06-14; macOS только сборка из исходников ([README](https://raw.githubusercontent.com/dmlc/decord/HEAD/README.md)) | да: `vr.get_batch([1, 3, 5, 7, 9])` «the efficient way to obtain a long list of frames» (README) |
| torchvision.io (0.29.1) | BSD ([PyPI](https://pypi.org/pypi/torchvision/json)) | в доках 0.21: «the torchvision video decoders will eventually be deprecated» в пользу TorchCodec ([io 0.21](https://docs.pytorch.org/vision/0.21/io.html)); в stable 0.29 раздел io посвящён изображениям и помечен deprecated ([io stable](https://docs.pytorch.org/vision/stable/io.html)) | не рекомендуется |
| torchcodec 0.17.0 | BSD 3-Clause ([LICENSE](https://raw.githubusercontent.com/pytorch/torchcodec/HEAD/LICENSE)) | нет: «TorchCodec uses the version of FFmpeg you already have installed»; на Windows нужны «shared» сборки FFmpeg ([README](https://raw.githubusercontent.com/pytorch/torchcodec/HEAD/README.md)) | да: `decoder[0 : -1 : 20]` и `decoder.get_frames_at(indices=[2, 100])` (README) |

## 5. Подходы к локализации

| Подход | Источник | Вход | Как получаем таймкод | Перенос на другой угол | Риски |
|---|---|---|---|---|---|
| A. Линейная проба / kNN на замороженных эмбеддингах кадров | DINOv2: «all-purpose visual features, i.e., features that work across image distributions and tasks without finetuning» ([arXiv 2304.07193](https://arxiv.org/abs/2304.07193)) | эмбеддинги кадров 4 fps, окна 2 с | пик оценки окна | слабо: эмбеддинг всего кадра зависит от ракурса | 22 положительных точки; классификатор выучит фон и людей, а не действие |
| B. Слабый MIL / ранжирование аномалий по видео-меткам | Sultani et al.: «the training labels (anomalous or normal) are at video-level instead of clip-level … normal and anomalous videos as bags and video segments as instances» ([arXiv 1801.04264](https://arxiv.org/abs/1801.04264)) | эмбеддинги сегментов, метка клипа | сегмент с максимальным score | как у A, зависит от признаков | 42 мешка мало; без таймкода пик может уехать на соседнее событие |
| C. Одна точка на действие (single-frame supervision) | SF-Net: «annotators are asked to identify only a single frame within the temporal window of an action»; расширение размеченного кадра на соседние и майнинг фона ([arXiv 2003.06845](https://arxiv.org/abs/2003.06845)) | покадровые признаки + один таймкод | покадровая оценка действия | как у A | ровно наш тип разметки, но метод учит сеть; на 42 клипах только упрощённая версия (окно ± k с как позитив) |
| D. Zero-shot по тексту (SigLIP) | SigLIP: zero-shot классификация картинка-текст ([arXiv 2303.15343](https://arxiv.org/abs/2303.15343)); SigLIP 2 улучшает «Localization, and Dense Features» ([arXiv 2502.14786](https://arxiv.org/abs/2502.14786)) | кадры + текстовые запросы | кадр с максимальным сходством с запросом | лучше A (нет обучения на нашем ракурсе) | типы неизвестны (FAQ 8–9); мелкие действия рук сверху плохо описываются текстом |
| E. Признаки «рука–объект–зона» из позы и детекции | RTMPose ([arXiv 2303.07399](https://arxiv.org/abs/2303.07399)), RT-DETR ([arXiv 2304.08069](https://arxiv.org/abs/2304.08069)), D-FINE ([arXiv 2410.13842](https://arxiv.org/abs/2410.13842)) | ключевые точки, боксы людей и предметов | первый кадр, где правило сработало | лучше: признаки в координатах тела и стола, а не пикселей | позы сверху хуже (детекторы учены на фронтальных фото); зоны надо размечать заново под новый угол |

## 6. Устойчивость к углу

Факты из источников:
- Перспективная аугментация: `torchvision.transforms.v2.RandomPerspective`: «Perform a random perspective transformation of the input with a given probability» ([docs](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.v2.RandomPerspective.html)).
- Перевод скелета в «виртуальную точку обзора»: View Adaptive networks: «a novel view adaptation module learns and determines the most suitable observation viewpoints, and transforms the skeletons to those viewpoints» ([arXiv 1804.07453](https://arxiv.org/abs/1804.07453)). Основа для признаков в координатах относительно тела кассира, а не пикселей.
- Самообучаемые признаки, устойчивые к смене распределения: DINOv2 заявляет признаки, которые «work across image distributions and tasks without finetuning» ([arXiv 2304.07193](https://arxiv.org/abs/2304.07193)). Про смену ракурса камеры статья прямо не говорит.
- Гомография плоскости стола (cv2.findHomography / warpPerspective): документация docs.opencv.org не открылась (Cloudflare): НЕ ПРОВЕРЕНО.

## 7. Рекомендации (не факты)

Shortlist для T02/T03:
- **Эмбеддинги кадров:** DINOv2-small with registers (22 M, 88 MB, Apache-2.0 код и веса). Запасной: SigLIP 2 base, он же даёт zero-shot по тексту для подсказки типа. OpenCLIP LAION не брать: карточка прямо выводит deployed use за рамки.
- **Детекция людей и предметов:** D-FINE-S через transformers (10 M, 41.5 MB). Запасной: RT-DETRv2-R18. Оба Apache-2.0 и в transformers, ставить mmdet не нужно.
- **Поза:** RTMPose-s или -m через rtmlib + onnxruntime (CPU работает и на M1). Запасной: ViTPose-base в transformers. MediaPipe брать только после того, как человек откроет PDF model card и подтвердит лицензию весов.
- **Видео-модели:** не брать в базу. VideoMAE и V-JEPA 1 некоммерческие, X-CLIP учён на Kinetics. V-JEPA 2 по лицензии MIT, но 326 M и неясный состав данных.
- **VLM (stretch):** Qwen2.5-VL-7B-AWQ (≈ 6.9 GB весов) на 8 GB VRAM рядом с детектором почти наверняка не влезет. Если нужна VLM, лучше Qwen3-VL-2B (Apache-2.0, 4.3 GB) или SmolVLM2-500M (Apache-2.0, есть ONNX). Qwen2.5-VL-3B не брать (non-commercial).
- **Декодирование:** opencv-python(-headless), читать все кадры и брать каждый 7-й (≈29/4) или `grab()` без `retrieve()` для пропусков. Запасной: PyAV. decord не брать: нет колёс под macOS. torchcodec не брать: нужен системный FFmpeg на Windows.
- **PyTorch:** на RTX 5070 ставить индекс cu132 с torch 2.14.x и драйвер ветки R595+, или cu128 с torch ≤ 2.11 и драйвер ≥ 570.65 (Windows). Перед T02 проверить `torch.cuda.get_device_capability()` и `torch.cuda.is_available()` на ноутбуке. Скорость эмбеддингов неизвестна, её мерить первым делом в T02.
- **Локализация:** начать с A (окна + логрегрессия, позитив = таймкод ± 1.5 с, это упрощённый SF-Net), добавить E для типа и устойчивости к углу. B и D как эксперимент при свободном времени.
- **Угол:** аугментация RandomPerspective при обучении головы на эмбеддингах; признаки ветки B нормировать на размер и центр тела кассира.

Не проверено (НЕ ПРОВЕРЕНО):
1. Лицензия весов MediaPipe Pose/Hand (model card в PDF не открылся).
2. HF-репо DEIM и условия замера latency в README DEIM.
3. Отдельная лицензия весов RTMDet и YOLOX; лицензии датасетов Body7 для весов RTMPose. Разобрано в разделе 8: веса Body7 и детектор YOLOX-HumanArt не годятся; открытыми остались sub-JHMDB, PoseTrack18, HICO-DET.
4. Состав датасета V-JEPA 2 (есть ли Kinetics).
5. Лицензия кода репозиториев Qwen3-VL и SmolVLM2 (веса проверены: apache-2.0).
6. Лицензия FFmpeg внутри колёс PyAV; API пропуска кадров в PyAV.
7. Документация OpenCV (VideoCapture.grab, findHomography): docs.opencv.org за Cloudflare.
8. Скорость DINOv2, SigLIP, ViTPose, VLM на любом железе: официальных цифр нет.

## 8. Дополнение 17.10: лицензия весов RTMPose

Все страницы открыты через playwright-cli 17.10.2026. Цитаты скопированы со страниц.

| Артефакт | Лицензия / условия | Источник |
|---|---|---|
| rtmlib (код) | Apache-2.0 | [LICENSE](https://raw.githubusercontent.com/Tau-J/rtmlib/HEAD/LICENSE) |
| rtmlib `Body`, режим по умолчанию `mode='balanced'` | детектор `yolox_m_8xb8-300e_humanart-c2c7a14a.zip`, поза `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip`; режим `lightweight`: `yolox_tiny_8xb8-300e_humanart-6f3252f9.zip` и `rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip`. Всё с download.openmmlab.com | [body.py](https://raw.githubusercontent.com/Tau-J/rtmlib/HEAD/rtmlib/tools/solution/body.py) |
| Веса RTMPose на download.openmmlab.com | отдельной лицензии весов нет. README mmpose: «This project is released under the Apache 2.0 license». В docs model zoo и FAQ слов «licen», «commercial» нет | [mmpose README](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/README.md), [RTMPose README](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/projects/rtmpose/README.md), [model zoo](https://mmpose.readthedocs.io/en/latest/model_zoo.html), [FAQ](https://mmpose.readthedocs.io/en/latest/faq.html) |
| Зеркало `Tau-J/RTMPose` на HF | тег apache-2.0, карточка пустая: «README.md exists but content is empty». Автор Tau-J (автор rtmlib), не OpenMMLab | [HF](https://huggingface.co/Tau-J/RTMPose), [HF API](https://huggingface.co/api/models/Tau-J/RTMPose) |
| Состав Body7 | «`*` denotes model trained on 7 public datasets»: AI Challenger, MS COCO, CrowdPose, MPII, sub-JHMDB, Halpe, PoseTrack18 | [RTMPose README](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/projects/rtmpose/README.md) |
| Чекпойнты RTMPose «COCO» | имена `rtmpose-*_simcc-coco_pt-aic-coco_420e`: дообучены на COCO после предобучения на AIC+COCO. Конфиг инициализирует backbone из `cspnext-m_udp-aic-coco_210e`. Статья: «combining COCO and AI Challenger datasets together for pre-training and fine-tuning». Чекпойнтов без AI Challenger в model zoo нет | [rtmpose_coco.md](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/configs/body_2d_keypoint/rtmpose/coco/rtmpose_coco.md), [конфиг](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-m_8xb256-420e_coco-256x192.py), [arXiv 2303.07399](https://arxiv.org/html/2303.07399) |
| Детектор YOLOX-HumanArt | обучен на `HumanArt/annotations/training_humanart_coco.json` | [конфиг yolox_s humanart](https://raw.githubusercontent.com/open-mmlab/mmpose/HEAD/projects/rtmpose/yolox/humanart/yolox_s_8xb8-300e_humanart.py) |
| COCO | аннотации: «licensed under a Creative Commons Attribution 4.0 License». Картинки: «The COCO Consortium does not own the copyright of the images. Use of the images must abide by the Flickr Terms of Use» | [cocodataset.org/#termsofuse](https://cocodataset.org/#termsofuse) |
| AI Challenger | **некоммерческое**. Сайт challenger.ai не открылся; архивная копия условий, статья 3 п.3: «选手应保证其仅在科学研究或课堂教学等非商业性目的范围内使用基础数据» (участник использует базовые данные только в некоммерческих целях: наука, обучение). README на GitHub лицензии не содержит | [архив challenger.ai/terms](https://web.archive.org/web/20180811180623/https://challenger.ai/terms), [GitHub](https://raw.githubusercontent.com/AIChallenger/AI_Challenger_2017/HEAD/README.md) |
| CrowdPose | лицензия не указана: в README её нет, файла LICENSE нет (404) | [README](https://raw.githubusercontent.com/Jeff-sjtu/CrowdPose/HEAD/README.md) |
| MPII | **некоммерческое**. «Licensed under the Simplified BSD License», но: «freely available for research purposes. Commercial use is not allowed due to the fact that the authors do not have the copyright for the images themselves» | [MPII download](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download) |
| sub-JHMDB | НЕ ПРОВЕРЕНО: jhmdb.is.tue.mpg.de не открылся (timeout); в архивной копии страницы /dataset текста лицензии нет | [архив](https://web.archive.org/web/20201204142009/http://jhmdb.is.tue.mpg.de/dataset) |
| Halpe | лицензия не указана: в README её нет, LICENSE 404. Train-картинки взяты из HICO-DET (лицензия HICO-DET: НЕ ПРОВЕРЕНО) | [README](https://raw.githubusercontent.com/Fang-Haoshu/Halpe-FullBody/HEAD/README.md) |
| PoseTrack18 | НЕ ПРОВЕРЕНО: posetrack.net не резолвится (DNS); в архивных копиях главной и download.php текста лицензии нет, скачивание за логином | [архив](https://web.archive.org/web/20210121121235/https://posetrack.net/) |
| Human-Art (данные детектора по умолчанию) | **некоммерческое**: «request authorization to use Human-Art for non-commercial purposes». Код репо Apache-2.0 | [README](https://raw.githubusercontent.com/IDEA-Research/HumanArt/HEAD/README.md), [LICENSE](https://raw.githubusercontent.com/IDEA-Research/HumanArt/HEAD/LICENSE) |
| ViTPose-base `usyd-community/vitpose-base-simple` | веса apache-2.0 (`cardData.license`), карточка: «License: Apache-2.0». Код ViTPose Apache-2.0 | [HF API](https://huggingface.co/api/models/usyd-community/vitpose-base-simple), [карточка](https://huggingface.co/usyd-community/vitpose-base-simple), [LICENSE](https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/HEAD/LICENSE) |
| Данные ViTPose-B simple | скрипт конвертации берёт `vitpose-b-simple.pth`; в README ViTPose-B simple decoder стоит в таблице «MS COCO val set (single-task training)», Pretrain = MAE. Раздел Training Data в карточке HF общий для всех вариантов: COCO, AI Challenger, MPII, CrowdPose | [convert_vitpose_to_hf.py](https://raw.githubusercontent.com/huggingface/transformers/HEAD/src/transformers/models/vitpose/convert_vitpose_to_hf.py), [ViTPose README](https://raw.githubusercontent.com/ViTAE-Transformer/ViTPose/HEAD/README.md) |
| MAE (инициализация ViTPose-B) | **некоммерческое**: «This project is under the CC-BY-NC 4.0 license». README ViTPose: «The base, large, and huge pre-trained models using MAE can be found in the MAE official repo» | [MAE LICENSE](https://raw.githubusercontent.com/facebookresearch/mae/HEAD/LICENSE), [MAE README](https://raw.githubusercontent.com/facebookresearch/mae/HEAD/README.md) |

**Вывод (рекомендация).** Веса RTMPose Body7 из rtmlib не брать. OpenMMLab не даёт отдельной лицензии на веса, а в данных обучения есть MPII («Commercial use is not allowed») и AI Challenger (только некоммерческие цели). Детектор по умолчанию YOLOX-HumanArt тоже не брать: Human-Art выдаётся только для non-commercial. Замена на «COCO»-чекпойнты RTMPose не спасает: они предобучены на AIC+COCO. ViTPose-base-simple чище: дообучен только на COCO, веса помечены apache-2.0. Но он инициализирован из MAE под CC-BY-NC 4.0; распространяется ли NC на дообученные веса, юридический вопрос, и это риск. Рекомендация: в базовом пайплайне позу не использовать, брать эмбеддинги DINOv2 и боксы людей от D-FINE-S или RT-DETRv2 (оба Apache-2.0, COCO). Если поза всё же нужна, кандидат ViTPose-base-simple через transformers, с записью риска MAE в `docs/MODELS.md` и решением P1. Второй кандидат MediaPipe Pose, если человек подтвердит лицензию весов в PDF model card (раздел 7, п.1). Предобучение на ImageNet есть почти у всех моделей shortlist; условия ImageNet здесь не проверял: НЕ ПРОВЕРЕНО.
