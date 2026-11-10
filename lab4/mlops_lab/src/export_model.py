import json
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from data import CLASSES, TEXT_COL, file_sha256, split

MODEL_NAME = 'sms-spam-classifier'

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
c = MlflowClient()
v = c.get_model_version_by_alias(MODEL_NAME, 'champion')
uri = f'models:/{MODEL_NAME}/{v.version}'
model = mlflow.sklearn.load_model(uri)
signature = mlflow.models.get_model_info(uri).signature
run = c.get_run(v.run_id)
Xtrain, Xval, Xtest, ytrain, yval, ytest = split()
out = Path('exported_model')
if out.exists():
    shutil.rmtree(out)
mlflow.sklearn.save_model(model, str(out), signature=signature, input_example=Xtrain.head(2),
                          pip_requirements=Path('requirements.txt').read_text().splitlines())

metadata = {'name': MODEL_NAME, 'version': v.version, 'run_id': v.run_id,
            'algorithm': run.data.params.get('algorithm'),
            'params': {k: run.data.params[k] for k in ('alpha', 'C', 'ngram_range', 'class_weight')
                       if k in run.data.params},
            'input': {'field': 'texts', 'type': 'list[str]', 'min_items': 1, 'max_items': 100,
                      'max_chars': 1000},
            'classes': CLASSES, 'positive_class': 'spam', 'dataset_sha256': file_sha256()}
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

sample = Xtest[ytest == 0].head(2)[TEXT_COL].tolist() + Xtest[ytest == 1].head(2)[TEXT_COL].tolist()
Path('sample_request.json').write_text(json.dumps({'texts': sample}, indent=2, ensure_ascii=False))
Path('expected_predictions.json').write_text(json.dumps(model.predict(pd.DataFrame({TEXT_COL: sample})).tolist()))
print(json.dumps(metadata, indent=2, ensure_ascii=False))
