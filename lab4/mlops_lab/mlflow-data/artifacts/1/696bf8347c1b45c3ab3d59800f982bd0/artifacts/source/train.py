import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from data import CLASSES, RAW_PATH, SEED, SOURCE_URL, TEXT_COL, file_sha256, load_clean, split

MODEL_NAME = 'sms-spam-classifier'

p = argparse.ArgumentParser()
p.add_argument('--algorithm', choices=['nb', 'lr'], default='nb')
p.add_argument('--alpha', type=float, default=1.0, help='MultinomialNB smoothing')
p.add_argument('--c', type=float, default=1.0, help='LogisticRegression inverse regularization')
p.add_argument('--ngram-max', type=int, default=1, help='TF-IDF ngram_range=(1, ngram_max)')
p.add_argument('--class-weight', choices=['none', 'balanced'], default='none')
p.add_argument('--register', action='store_true')
a = p.parse_args()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000'))
mlflow.set_experiment('lab4-sms-spam')

df = load_clean()
Xtrain, Xval, Xtest, ytrain, yval, ytest = split(df)  # test в этом скрипте НЕ используется

clf = (MultinomialNB(alpha=a.alpha) if a.algorithm == 'nb' else
       LogisticRegression(C=a.c, max_iter=2000, random_state=SEED,
                          class_weight=None if a.class_weight == 'none' else 'balanced'))
# TF-IDF обучается только на train внутри Pipeline и сохраняется вместе с классификатором.
model = Pipeline([
    ('features', ColumnTransformer([('tfidf', TfidfVectorizer(lowercase=True, ngram_range=(1, a.ngram_max),
                                                              sublinear_tf=True), TEXT_COL)])),
    ('clf', clf),
])

with mlflow.start_run(run_name=f'{a.algorithm}-experiment') as run:
    mlflow.log_params({'algorithm': a.algorithm, 'seed': SEED, 'dataset': 'UCI SMS Spam Collection',
                       'ngram_range': f'(1, {a.ngram_max})', 'n_train': len(Xtrain),
                       'n_validation': len(Xval), 'n_test': len(Xtest)})
    if a.algorithm == 'nb':
        mlflow.log_params({'alpha': a.alpha})
    else:
        mlflow.log_params({'C': a.c, 'class_weight': a.class_weight})
    mlflow.set_tags({'dataset_sha256': file_sha256(), 'purpose': 'lab4', 'variant': '2',
                     'selection_metric': 'val_f1_spam'})
    model.fit(Xtrain, ytrain)
    pred = model.predict(Xval)
    metrics = {'val_f1_spam': f1_score(yval, pred, pos_label=1),
               'val_precision_spam': precision_score(yval, pred, pos_label=1),
               'val_recall_spam': recall_score(yval, pred, pos_label=1),
               'val_accuracy': accuracy_score(yval, pred),
               'val_f1_macro': f1_score(yval, pred, average='macro')}
    mlflow.log_metrics(metrics)
    mlflow.log_input(mlflow.data.from_pandas(df[[TEXT_COL, 'target']], source=SOURCE_URL, targets='target',
                                             name='sms_spam_dedup'), context='full_dataset_before_split')
    mlflow.log_dict({'train_indices': Xtrain.index.tolist(), 'validation_indices': Xval.index.tolist(),
                     'test_indices': Xtest.index.tolist(), 'columns': [TEXT_COL],
                     'target_names': CLASSES, 'raw_file': RAW_PATH.name,
                     'preprocessing': 'drop_duplicates(text, keep=first); spam=1, ham=0'}, 'split.json')
    mlflow.log_artifact(__file__, artifact_path='source')
    mlflow.log_artifact(str(Path(__file__).with_name('data.py')), artifact_path='source')
    mlflow.log_artifact('requirements.txt', artifact_path='source')
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(yval, pred, display_labels=CLASSES, ax=ax)
    ax.set_title('Validation confusion matrix')
    mlflow.log_figure(fig, 'validation_confusion_matrix.png')
    plt.close(fig)
    info = mlflow.sklearn.log_model(model, name='model', input_example=Xtrain.head(2),
                                    signature=infer_signature(Xtrain, model.predict(Xtrain)),
                                    pip_requirements=Path('requirements.txt').read_text().splitlines())
    print(json.dumps({'run_id': run.info.run_id, 'model_uri': info.model_uri,
                      **{k: round(v, 4) for k, v in metrics.items()}}, indent=2))
    if a.register:
        mv = mlflow.register_model(info.model_uri, MODEL_NAME)
        print('REGISTERED_VERSION=', mv.version)
