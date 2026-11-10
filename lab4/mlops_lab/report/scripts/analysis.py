import json, sys, re, collections
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, mlflow.sklearn
from pathlib import Path
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from data import load_raw, load_clean, split
OUT = Path('report/analysis'); res = {}
raw = load_raw(); clean = load_clean()
res['raw_counts'] = raw['label'].value_counts().to_dict()
dup = raw[raw.duplicated('text', keep='first')]
res['dup_counts'] = dup['label'].value_counts().to_dict()
res['clean_counts'] = clean['label'].value_counts().to_dict()
clean['len'] = clean['text'].str.len(); clean['words'] = clean['text'].str.split().str.len()
clean['digits'] = clean['text'].str.count(r'\d'); clean['upper_share'] = clean['text'].apply(lambda s: sum(c.isupper() for c in s)/max(1,sum(c.isalpha() for c in s)))
clean['has_url'] = clean['text'].str.contains(r'http|www\.', case=False)
clean['has_phone'] = clean['text'].str.contains(r'\d{5,}')
clean['has_currency'] = clean['text'].str.contains('£|\\$|€')
st = clean.groupby('label').agg(len_mean=('len','mean'), len_median=('len','median'), len_max=('len','max'), len_min=('len','min'),
    words_mean=('words','mean'), digits_mean=('digits','mean'), upper_share=('upper_share','mean'), url=('has_url','mean'),
    phone=('has_phone','mean'), currency=('has_currency','mean'))
res['stats'] = st.round(4).to_dict(orient='index')
res['len_overall_max'] = int(clean['len'].max())
clean[['label','len','words']].to_csv(OUT/'lengths.csv', index=False)
def toks(s): return re.findall(r'(?u)\b\w\w+\b', s.lower())
top = {}
for lab in ['ham','spam']:
    c = collections.Counter(t for s in clean[clean.label==lab].text for t in toks(s))
    top[lab] = c.most_common(25)
res['top_tokens'] = top
Xtr, Xv, Xte, ytr, yv, yte = split(clean)
res['split'] = {k: [int(len(y)), int(y.sum())] for k, y in [('train', ytr), ('val', yv), ('test', yte)]}
# per-run validation confusion matrices from logged models
runs = pd.read_csv('report/experiments.csv')
import mlflow
mlflow.set_tracking_uri('sqlite:///mlflow-data/mlflow.db')
from mlflow import MlflowClient
c = MlflowClient()
per = {}
for _, r in runs.iterrows():
    mid = None
    for m in c.search_logged_models(experiment_ids=['1'], filter_string=f"source_run_id='{r.run_id}'"):
        mid = m.model_id
    m = mlflow.sklearn.load_model(f'mlflow-data/artifacts/1/models/{mid}/artifacts')
    pv = m.predict(Xv)
    vocab = len(m.named_steps['features'].named_transformers_['tfidf'].vocabulary_)
    per[r.run_id] = {'n': int(r.n), 'model_id': mid, 'cm_val': confusion_matrix(yv, pv).tolist(), 'vocab': vocab}
    if int(r.n) in (8, 9):
        pt = m.predict(Xte)
        per[r.run_id]['test'] = {'f1': f1_score(yte, pt), 'prec': precision_score(yte, pt), 'rec': recall_score(yte, pt), 'acc': accuracy_score(yte, pt), 'cm': confusion_matrix(yte, pt).tolist()}
        if int(r.n) == 8:
            nb = m; vec = m.named_steps['features'].named_transformers_['tfidf']; clf = m.named_steps['clf']
            names = np.array(vec.get_feature_names_out())
            ratio = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
            o = np.argsort(ratio)
            res['nb_top_spam'] = [(names[i], round(float(ratio[i]),3)) for i in o[::-1][:20]]
            res['nb_top_ham'] = [(names[i], round(float(ratio[i]),3)) for i in o[:20]]
            proba = m.predict_proba(Xte)[:, 1]
            err = Xte.copy(); err['y'] = yte.values; err['p'] = pt; err['proba'] = proba
            res['test_fn'] = err[(err.y==1)&(err.p==0)].sort_values('proba')[['text','proba']].values.tolist()
            res['test_fp'] = err[(err.y==0)&(err.p==1)].sort_values('proba', ascending=False)[['text','proba']].values.tolist()
            pd.DataFrame({'y': yte.values, 'proba_v1': proba}).to_csv(OUT/'test_proba_v1.csv', index=False)
        else:
            vec = m.named_steps['features'].named_transformers_['tfidf']; clf = m.named_steps['clf']
            names = np.array(vec.get_feature_names_out()); w = clf.coef_[0]; o = np.argsort(w)
            res['lr_top_spam'] = [(names[i], round(float(w[i]),3)) for i in o[::-1][:20]]
            res['lr_top_ham'] = [(names[i], round(float(w[i]),3)) for i in o[:20]]
            pd.DataFrame({'y': yte.values, 'proba_v2': m.predict_proba(Xte)[:, 1]}).to_csv(OUT/'test_proba_v2.csv', index=False)
res['per_run'] = per
(OUT/'analysis.json').write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
print(json.dumps({k: v for k, v in res.items() if k not in ('top_tokens','per_run')}, ensure_ascii=False, default=str)[:6000])
print({v['n']: (v['cm_val'], v['vocab'], v.get('test')) for v in per.values()})
