import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

P = Path('/Users/sedooj/Downloads/ml/lab4/mlops_lab')
A = P / 'report/analysis'
OUT = P / 'report/figures'
OUT.mkdir(parents=True, exist_ok=True)
res = json.loads((A / 'analysis.json').read_text())
runs = pd.read_csv(P / 'report/experiments.csv')
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
SEQ = LinearSegmentedColormap.from_list('seq', ['#f4f8fe', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK, 'axes.titlesize': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'grid.linewidth': 0.6, 'legend.frameon': False, 'figure.dpi': 200, 'savefig.bbox': 'tight'})


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', facecolor=SURF)
    plt.close(fig)


# 1. Распределение классов до/после дедупликации
fig, ax = plt.subplots(figsize=(7, 2.6))
cats = ['Исходный набор', 'Удалённые дубликаты', 'После дедупликации']
ham = [res['raw_counts']['ham'], res['dup_counts']['ham'], res['clean_counts']['ham']]
spam = [res['raw_counts']['spam'], res['dup_counts']['spam'], res['clean_counts']['spam']]
y = np.arange(len(cats))
ax.barh(y, ham, color=BLUE, height=0.55, label='ham')
ax.barh(y, spam, left=ham, color=ORANGE, height=0.55, label='spam', edgecolor=SURF, linewidth=1.5)
for i in range(3):
    ax.text(ham[i] + spam[i] + 60, i, f'ham {ham[i]} / spam {spam[i]}', va='center', color=INK2, fontsize=9)
ax.set_yticks(y, cats); ax.invert_yaxis(); ax.set_xlim(0, 7200); ax.grid(axis='y', visible=False)
ax.set_xlabel('Число SMS'); ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
save(fig, 'f01_class_counts')

# 2. Длина сообщений по классам
L = pd.read_csv(A / 'lengths.csv')
fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.arange(0, 400, 10)
for lab, col in (('ham', BLUE), ('spam', ORANGE)):
    v = L[L.label == lab]['len'].clip(upper=399)
    ax.hist(v, bins=bins, density=True, histtype='step', linewidth=2, color=col, label=lab)
    ax.axvline(v.median(), color=col, linewidth=1, linestyle='--')
ax.set_xlabel('Длина сообщения, символов (значения > 400 отнесены к последнему интервалу)')
ax.set_ylabel('Плотность'); ax.legend()
ax.text(200, ax.get_ylim()[1] * 0.75, 'пунктир — медианы:\nham = 53 символа\nspam = 148 символов', color=INK2, fontsize=9)
save(fig, 'f02_lengths')

# 3. Частые токены
fig, axs = plt.subplots(1, 2, figsize=(7.5, 4.6))
for ax, lab, col in ((axs[0], 'ham', BLUE), (axs[1], 'spam', ORANGE)):
    t = res['top_tokens'][lab][:15]
    ax.barh([w for w, _ in t][::-1], [c for _, c in t][::-1], color=col, height=0.6)
    ax.set_title(f'Класс {lab}: 15 самых частых токенов', loc='left'); ax.grid(axis='y', visible=False)
    ax.set_xlabel('Число вхождений')
fig.tight_layout()
save(fig, 'f03_top_tokens')

# 4. Метрики validation по запускам
fig, ax = plt.subplots(figsize=(7.5, 3.6))
lab = [f"#{r.n} {r['params.algorithm'].upper()}\n" + (f"α={r['params.alpha']}" if r['params.algorithm'] == 'nb' else
       f"C={r['params.C']:g}" + (' bal' if r['params.class_weight'] == 'balanced' else '')) + f" {r['params.ngram_range']}"
       for _, r in runs.iterrows()][:7]
x = np.arange(7)
for k, (m, col, mk) in enumerate((('metrics.val_f1_spam', BLUE, 'o'), ('metrics.val_precision_spam', ORANGE, 's'),
                                  ('metrics.val_recall_spam', AQUA, '^'))):
    v = runs[m].values[:7]
    ax.plot(x + (k - 1) * 0.18, v, linestyle='none', marker=mk, markersize=8, color=col,
            markeredgecolor=SURF, markeredgewidth=1.5,
            label={'metrics.val_f1_spam': 'F1 (spam)', 'metrics.val_precision_spam': 'Precision (spam)',
                   'metrics.val_recall_spam': 'Recall (spam)'}[m])
ax.set_xticks(x, lab, fontsize=8); ax.set_ylim(0.55, 1.03); ax.set_ylabel('Значение на validation')
ax.legend(ncol=3, loc='lower right', fontsize=9)
for i, v in enumerate(runs['metrics.val_f1_spam'].values[:7]):
    ax.text(i - 0.18, v - 0.045, f'{v:.3f}', ha='center', fontsize=7.5, color=INK2)
save(fig, 'f04_val_metrics')

# 5. Матрицы ошибок validation для 7 запусков
per = sorted(res['per_run'].values(), key=lambda d: d['n'])[:7]
fig, axs = plt.subplots(2, 4, figsize=(8, 4.4))
for ax, d, t in zip(axs.flat, per, lab):
    cm = np.array(d['cm_val'])
    ax.imshow(np.log1p(cm), cmap=SEQ, vmin=0, vmax=np.log1p(904))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=10,
                    color='white' if cm[i, j] > 100 else INK)
    ax.set_xticks([0, 1], ['ham', 'spam'], fontsize=8); ax.set_yticks([0, 1], ['ham', 'spam'], fontsize=8)
    ax.set_title(t.replace('\n', ' '), fontsize=8); ax.grid(False)
axs.flat[-1].axis('off')
axs.flat[-1].text(0, 0.6, 'Строки — истинный класс,\nстолбцы — предсказанный.\nValidation: 904 ham, 130 spam', fontsize=9,
                  color=INK2, transform=axs.flat[-1].transAxes)
fig.tight_layout()
save(fig, 'f05_cm_val_all')

# 6. Precision–recall и распределение вероятностей на test (v1 и v2)
from sklearn.metrics import precision_recall_curve, average_precision_score
p1 = pd.read_csv(A / 'test_proba_v1.csv'); p2 = pd.read_csv(A / 'test_proba_v2.csv')
fig, ax = plt.subplots(figsize=(6, 4))
ap = {}
for df, c, col, name in ((p1, 'proba_v1', BLUE, 'Версия 1: NB α=0.1 (1,1)'), (p2, 'proba_v2', ORANGE, 'Версия 2: LR C=10 balanced (1,2)')):
    pr, rc, _ = precision_recall_curve(df.y, df[c]); ap[c] = average_precision_score(df.y, df[c])
    ax.plot(rc, pr, color=col, linewidth=2, label=f'{name}, AP={ap[c]:.3f}')
ax.plot([0.8779], [0.9664], marker='o', markersize=9, color=BLUE, markeredgecolor=SURF, markeredgewidth=2, linestyle='none')
ax.plot([0.9084], [0.9297], marker='o', markersize=9, color=ORANGE, markeredgecolor=SURF, markeredgewidth=2, linestyle='none')
ax.set_xlabel('Recall (spam)'); ax.set_ylabel('Precision (spam)'); ax.set_xlim(0.5, 1.01); ax.set_ylim(0.5, 1.02)
ax.legend(loc='lower left', fontsize=9)
ax.text(0.52, 0.62, 'Точки — рабочий порог 0.5', fontsize=9, color=INK2)
save(fig, 'f06_pr_test')
(A / 'ap.json').write_text(json.dumps(ap))

fig, ax = plt.subplots(figsize=(7, 3))
b = np.linspace(0, 1, 41)
ax.hist(p1[p1.y == 0].proba_v1, bins=b, color=BLUE, label='истинный ham (904)', log=True, edgecolor=SURF)
ax.hist(p1[p1.y == 1].proba_v1, bins=b, color=ORANGE, label='истинный spam (131)', log=True, edgecolor=SURF, alpha=0.9)
ax.axvline(0.5, color=INK2, linestyle='--', linewidth=1); ax.text(0.51, 300, 'порог 0.5', color=INK2, fontsize=9)
ax.set_xlabel('Вероятность spam по модели версии 1'); ax.set_ylabel('Число SMS (лог. шкала)'); ax.legend()
save(fig, 'f07_proba_hist')

# 7. Важные признаки моделей
fig, axs = plt.subplots(1, 2, figsize=(7.6, 4.8))
for ax, key_s, key_h, title, xl in ((axs[0], 'nb_top_spam', 'nb_top_ham', 'Версия 1 (NB)', 'log P(t|spam) − log P(t|ham)'),
                                    (axs[1], 'lr_top_spam', 'lr_top_ham', 'Версия 2 (LR)', 'коэффициент логит-модели')):
    s = res[key_s][:10]; h = res[key_h][:10]
    names = [w for w, _ in h][::-1] + [w for w, _ in s][::-1]
    vals = [v for _, v in h][::-1] + [v for _, v in s][::-1]
    ax.barh(range(len(vals)), vals, color=[BLUE if v < 0 else ORANGE for v in vals], height=0.65)
    ax.set_yticks(range(len(vals)), names, fontsize=8); ax.axvline(0, color=INK2, linewidth=0.8)
    ax.set_title(title, loc='left'); ax.set_xlabel(xl, fontsize=8); ax.grid(axis='y', visible=False)
axs[1].text(0.98, 0.02, 'синие — признаки ham,\nоранжевые — признаки spam', transform=axs[1].transAxes, ha='right',
            fontsize=8, color=INK2)
fig.tight_layout()
save(fig, 'f08_features')

# 8. Латентность от размера пакета
bench = json.loads((A / 'bench.json').read_text())
fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.2))
bs = [r['batch'] for r in bench]
axs[0].plot(bs, [r['client_median_ms'] for r in bench], marker='o', color=ORANGE, linewidth=2, markersize=7, label='клиент (HTTP round-trip)')
axs[0].plot(bs, [r['handler_median_ms'] for r in bench], marker='o', color=BLUE, linewidth=2, markersize=7, label='обработчик (latency_ms)')
axs[0].set_xlabel('SMS в запросе'); axs[0].set_ylabel('Медиана, мс'); axs[0].set_ylim(0, 7); axs[0].legend(fontsize=8, loc='upper left')
axs[0].set_title('Время ответа', loc='left')
axs[1].plot(bs, [r['per_message_ms'] for r in bench], marker='o', color=ORANGE, linewidth=2, markersize=7)
axs[1].set_xscale('log'); axs[1].set_yscale('log'); axs[1].set_xticks(bs, [str(b) for b in bs])
axs[1].set_xlabel('SMS в запросе'); axs[1].set_ylabel('мс на одно SMS'); axs[1].set_title('Время на сообщение', loc='left')
for b_, r in zip(bs, bench):
    axs[1].text(b_, r['per_message_ms'] * 1.25, f"{r['per_message_ms']}", ha='center', fontsize=8, color=INK2)
fig.tight_layout()
save(fig, 'f09_latency')

# 9. Размер образа по слоям
fig, ax = plt.subplots(figsize=(7, 2.6))
layers = [('pip install (зависимости)', 786), ('Debian trixie (база)', 101), ('Python 3.12 (база)', 43.9), ('модель + app.py', 0.35)]
ax.barh([n for n, _ in layers][::-1], [v for _, v in layers][::-1], color=BLUE, height=0.55)
for i, (n, v) in enumerate(layers[::-1]):
    ax.text(v + 8, i, f'{v:g} МБ', va='center', fontsize=9, color=INK2)
ax.set_xlim(0, 900); ax.set_xlabel('МБ'); ax.grid(axis='y', visible=False)
save(fig, 'f10_image_layers')
print(sorted(p.name for p in OUT.iterdir()), ap)
