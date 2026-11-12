# ЛР2. Классификация. Нейронные сети

Классификация тяжести ДТП (Великобритания): `Slight` (0) против `Serious`/`Fatal` (1). Датасет — `accident_data.csv` из `../lab1`.

## Содержимое

| Путь | Описание |
|---|---|
| `lab2_classification.ipynb` | полный отчёт: все 8 разделов, код, графики, выводы |
| `results/` | CSV с метриками, лучшими параметрами, исследованием гиперпараметров и экспериментами сети |
| `figures/` | ROC-кривые, столбчатые диаграммы метрик, матрицы ошибок, кривые обучения |
| `logs/` | логи TensorBoard (14 запусков сети) |
| `requirements.txt` | версии библиотек |

## Запуск

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace lab2_classification.ipynb
```

Файл `accident_data.csv` ищется рядом с ноутбуком и в `../lab1/`.

## TensorBoard

```bash
.venv/bin/tensorboard --logdir logs
```

Открыть http://localhost:6006 — вкладка Scalars (loss, accuracy, auc для обучения и валидации), Histograms (веса).

## Итоговое сравнение (тестовая выборка, 6000 объектов)

См. `results/comparison_all.csv` и раздел 7 ноутбука. Лучший метод — дерево решений (AUC 0.645, F1 0.319).
