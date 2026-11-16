# butterfly-detection

Сравнение Grounding DINO tiny, OWLv2 base и Qwen3-VL-2B-Instruct для поиска бабочек на фотографиях.

## Запуск

Требуются Linux amd64, NVIDIA GPU, Docker и NVIDIA Container Toolkit.

```bash
git clone https://github.com/arslan-yam/butterfly-detection.git
cd butterfly-detection
docker build --platform linux/amd64 -t butterfly-models:1.0 .
bash models/docker.sh bash models/docker/compare.sh raw
```

Используются 400 записей из `bbox/main/bbox_raw.jsonl` и `bbox/pilot/bbox_raw.jsonl`.
Фотографии и веса скачиваются автоматически. Все три модели ищут взрослых дневных и ночных бабочек.
Эксперимент с уверенностью: DINO — пороги 0,25/0,35, OWLv2 — 0,10/0,20;
запросы, веса и NMS остаются теми же. Qwen возвращает координаты без confidence рамок.
По умолчанию используется GPU 0; другую карту можно выбрать через `GPU_DEVICE`.

Для сравнения с собственной проверенной разметкой сохраните 400 записей в `bbox/final.jsonl` и выполните:

```bash
bash models/docker.sh bash models/docker/compare.sh golden /inputs/final.jsonl
```

## Вход

JSONL, одна строка на фотографию:

```json
{"photo_id": 123, "image": "/inputs/images/123.jpg", "status": "butterflies", "boxes": [{"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}]}
```

`photo_id` — уникальный числовой ID; `image` — URL или путь внутри контейнера.
Папка `bbox/` подключается как `/inputs`, поэтому пример соответствует файлу `bbox/images/123.jpg`.
При отсутствии бабочек: `status: "no_butterflies"`, `boxes: []`.
Координаты нормализованы от 0 до 1: `left`, `top` — левый верхний угол, `width`, `height` — размеры рамки.

## Выход

Сохранённый прогон: [предварительный лидерборд](models/results/raw/leaderboard.csv) и предсказания в `models/results/raw/`.

Результаты находятся в `models/artifacts/runs/raw/` (для проверенной разметки — `golden/`):

- `<config>/predictions.jsonl` — рамки, время обработки и ошибки по каждому фото;
- `<config>/metrics.json` — precision, recall, F1@IoU 0.5/0.75, TP/FP/FN и время;
- `leaderboard.csv` — сравнение конфигураций; `selection.json` — выбор модели по проверенной разметке.

Формат предсказаний:

```json
{"photo_id": 123, "boxes": [{"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4}], "error": null}
{"photo_id": 124, "boxes": [], "error": null}
```

`boxes: []` означает отсутствие объектов. При технической ошибке — `boxes: null` и причина в `error`.
Рабочий файл также содержит время, scores детекторов и исходный ответ Qwen.
`nms_iou: 0.5` удаляет дубли: при IoU двух рамок выше 0,5 остаётся рамка с большим score
(у Qwen — первая в ответе). Это отдельный параметр от уверенности и IoU в метрике F1.
Для продолжения прерванного прогона повторите команду запуска.

## Пересчёт по golden set

После review метрики сохранённых предсказаний можно обновить без GPU (нужен Python 3):

```bash
python3 -m models.rescore --sources bbox/final.jsonl \
  --previous models/results/raw --output models/results/golden
```

ID и источники фотографий должны совпадать с исходным набором; порядок строк может отличаться.
