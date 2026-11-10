import json
import logging
import time
from pathlib import Path
from typing import Annotated

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MODEL_DIR = Path('/app/exported_model')
app = FastAPI(title='SMS spam classifier', version='1.0')
model = mlflow.sklearn.load_model(str(MODEL_DIR))
meta = json.loads((MODEL_DIR / 'metadata.json').read_text())
log = logging.getLogger('uvicorn.error')

SmsText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    texts: list[SmsText] = Field(min_length=1, max_length=100)


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': meta['version']}


@app.get('/metadata')
def metadata():
    return meta


@app.post('/predict')
def predict(body: PredictRequest):
    start = time.perf_counter()
    X = pd.DataFrame({'text': body.texts})
    result = model.predict(X).astype(int).tolist()
    proba = model.predict_proba(X)[:, 1].round(4).tolist()
    elapsed = (time.perf_counter() - start) * 1000
    log.info('prediction model_version=%s rows=%s latency_ms=%.3f', meta['version'], len(X), elapsed)
    return {'predictions': result, 'labels': [meta['classes'][v] for v in result],
            'spam_probability': proba, 'model_version': meta['version'], 'latency_ms': round(elapsed, 3)}
