import argparse
import os

import mlflow
from mlflow import MlflowClient

MODEL_NAME = 'sms-spam-classifier'

p = argparse.ArgumentParser()
p.add_argument('--version', required=True)
p.add_argument('--alias', default='champion')
a = p.parse_args()
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
c = MlflowClient()
v = c.get_model_version(MODEL_NAME, a.version)
c.set_registered_model_alias(MODEL_NAME, a.alias, a.version)
print(f'{a.alias} -> version {v.version}, run {v.run_id}')
