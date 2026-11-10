# ЛР4 MLOps — вариант 2: сервис фильтрации SMS-спама

Герасимов С. П., АВТ-313. Отчёт: `ЛР4_Герасимов_АВТ-313.docx` (тот же отчёт в виде ноутбука: `ЛР4_Герасимов_АВТ-313.ipynb`).

Цикл: данные → эксперименты в MLflow → Model Registry (alias `champion`) → экспорт → FastAPI в Docker → обновление и откат → Docker Compose с healthcheck.

## Данные
UCI SMS Spam Collection (CC BY 4.0), https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Файл `data/raw/SMSSpamCollection`: TSV без заголовка, UTF-8, `label\ttext`.
SHA-256: `7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`.
Предобработка в `src/data.py`: удаление дубликатов текста (5574 → 5171), разбиение 60/20/20 по классу, `random_state=42`.

## Воспроизведение (macOS/Linux, Python 3.12, Docker)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p mlflow-data/artifacts
mlflow server --backend-store-uri sqlite:///mlflow-data/mlflow.db \
  --artifacts-destination ./mlflow-data/artifacts --serve-artifacts --host 127.0.0.1 --port 5000
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python src/train.py --algorithm nb --alpha 1.0 --ngram-max 1
python src/train.py --algorithm nb --alpha 0.1 --ngram-max 1
python src/train.py --algorithm nb --alpha 0.1 --ngram-max 2
python src/train.py --algorithm lr --c 1   --ngram-max 1
python src/train.py --algorithm lr --c 10  --ngram-max 1
python src/train.py --algorithm lr --c 100 --ngram-max 2
python src/train.py --algorithm lr --c 10  --ngram-max 2 --class-weight balanced
python src/train.py --algorithm nb --alpha 0.1 --ngram-max 1 --register
python src/train.py --algorithm lr --c 10 --ngram-max 2 --class-weight balanced --register
python src/promote.py --version 1
python src/export_model.py && python src/evaluate.py
docker build -t sms-spam-api:v1 .
docker run -d --name sms-spam-api -p 127.0.0.1:8000:8000 sms-spam-api:v1
python src/check_api.py
python src/check_errors.py
```
Обновление и откат: `promote.py --version 2` → `export_model.py` → `docker build -t sms-spam-api:v2 .` → перезапуск контейнера на v2;
откат: перезапуск на `sms-spam-api:v1`, `promote.py --version 1`, `export_model.py`.
Compose: `IMAGE_TAG=compose-demo docker compose up -d --build`, `docker compose ps` (статус healthy), `docker compose down`.

## API
| Метод | Путь | Описание |
|---|---|---|
| GET | `/health` | `{"status":"ok","model_version":"1"}` |
| GET | `/metadata` | версия, run_id, параметры, классы, контракт входа |
| POST | `/predict` | `{"texts": ["..."]}` от 1 до 100 непустых строк длиной не более 1000 символов |

Ответ `/predict`: `predictions` (0/1), `labels` (ham/spam), `spam_probability`, `model_version`, `latency_ms`.
Лишние поля, нестроковые значения, пустой список и пустые строки дают 422. Пример запроса лежит в `sample_request.json`.
