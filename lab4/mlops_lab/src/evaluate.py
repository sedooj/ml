import json
from pathlib import Path

import mlflow.sklearn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score)

from data import CLASSES, split

m = mlflow.sklearn.load_model('exported_model')
meta = json.loads(Path('exported_model/metadata.json').read_text())
_, _, Xtest, _, _, ytest = split()
pred = m.predict(Xtest)
r = {'model_version': meta['version'], 'n_test': len(ytest),
     'test_f1_spam': f1_score(ytest, pred, pos_label=1),
     'test_precision_spam': precision_score(ytest, pred, pos_label=1),
     'test_recall_spam': recall_score(ytest, pred, pos_label=1),
     'test_accuracy': accuracy_score(ytest, pred),
     'test_f1_macro': f1_score(ytest, pred, average='macro'),
     'confusion_matrix': confusion_matrix(ytest, pred).tolist()}
Path('test_metrics.json').write_text(json.dumps(r, indent=2))
print(json.dumps(r, indent=2))
print(classification_report(ytest, pred, target_names=CLASSES, digits=4))
