# Ю-ри: Пайплайн нарезки законов и API-сервис юриста

chunking/ — нарезка .rtf из base_files на атомы.
service/ — FastAPI-сервис «Ю-ри» (main, indexer).

## 1. Подготовка
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

## 2. Нарезка и векторизация
python -m chunking.pipeline \
  --input-dir ./base_files \
  --output ./data/atoms.jsonl \
  --api-key sk-... \
  [--migrate-legacy ./data/laws.jsonl] \
  [--concurrency 5]
--api-key – апи ключ Deepseek, нужен для аугметации поисковой базы.

## 3. Индексация
python service/indexer.py --atoms-jsonl ./data/atoms.jsonl

## 4. Запуск сервиса
python -m uvicorn service.main:app --host 0.0.0.0 --port 8000
Swagger: http://localhost:8000/docs.
