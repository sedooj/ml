# Лабораторная работа №1 - полный листинг кода (из lab1_accident.ipynb)
# Запуск: из папки, где лежит needs/accident_data.csv

# Если какая-то библиотека не установлена, раскомментируйте строку и выполните ячейку:
# %pip install -q numpy pandas matplotlib seaborn scikit-learn statsmodels

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             mean_absolute_percentage_error, make_scorer)

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tools.tools import add_constant

import sklearn, statsmodels
print(f"Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, "
      f"scikit-learn {sklearn.__version__}, seaborn {sns.__version__}, statsmodels {statsmodels.__version__}")

sns.set_theme(style="whitegrid", context="notebook")
RANDOM_STATE = 42
N_JOBS = 2          # число параллельных процессов в GridSearchCV; 1 - если мало оперативной памяти (< 8 ГБ)
pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.precision", 4)

# ----------------------------------------------------------------------

# Файл ищется в типичных местах относительно папки, из которой запущен ноутбук
candidates = [Path("needs/accident_data.csv"), Path("accident_data.csv"),
              Path("../needs/accident_data.csv"), Path("../accident_data.csv")]
data_path = next((p for p in candidates if p.exists()), None)
if data_path is None:
    raise FileNotFoundError(
        "Не найден accident_data.csv. Положите файл рядом с ноутбуком или в папку needs/, "
        f"либо задайте путь вручную: data_path = Path('...'). Текущая папка: {Path.cwd()}")

# Параметры чтения: разделитель "," и кодировка UTF-8 стоят по умолчанию (файл стандартный CSV
# с заголовком в первой строке). Даты НЕ парсим при чтении: формат ДД-ММ-ГГГГ pandas
# мог бы распознать неверно (перепутать день и месяц), поэтому преобразуем явно позже.
df_raw = pd.read_csv(data_path)
print(f"Размер исходного датасета: {df_raw.shape[0]:,} строк и {df_raw.shape[1]} столбцов")
display(df_raw.head())

# ----------------------------------------------------------------------

print("Информация о столбцах (количество непустых значений и типы):")
df_raw.info()

print("\nОписательная статистика числовых признаков:")
display(df_raw.describe().T)

print("\nПропуски по столбцам:")
na_table = pd.DataFrame({
    "Количество пропусков": df_raw.isna().sum(),
    "Доля, %": df_raw.isna().mean() * 100,
})
display(na_table)

print(f"\nКоличество полностью повторяющихся строк: {df_raw.duplicated().sum()}")

print("\nУникальные значения категориальных столбцов:")
for col in ["Accident_Severity", "Light_Conditions", "Road_Surface_Conditions", "Road_Type",
            "Urban_or_Rural_Area", "Weather_Conditions", "Vehicle_Type"]:
    print(f"{col}: {sorted(df_raw[col].dropna().unique().tolist())}")
print(f"District Area: {df_raw['District Area'].nunique()} уникальных районов")

# ----------------------------------------------------------------------

# Проверка столбца Index: можно ли считать его идентификатором ДТП?
print(f"Уникальных значений Index: {df_raw['Index'].nunique():,} из {len(df_raw):,} строк")
sci = df_raw["Index"].str.contains("E+", regex=False)          # запись вида 2.01E+12 (формат Excel)
print(f"Строк, где Index записан в научной нотации: {sci.sum():,} ({sci.mean():.1%})")
print("Наиболее частые значения Index:")
display(df_raw["Index"].value_counts().head(5).rename("Количество строк").to_frame())

# Проверка диапазона дат и корректности формата
dates = pd.to_datetime(df_raw["Accident Date"], format="%d-%m-%Y", errors="coerce")
print(f"\nНеразобранных дат: {dates.isna().sum()};  период: {dates.min().date()} — {dates.max().date()}")
print("Число ДТП по годам:")
display(dates.dt.year.value_counts().sort_index().rename("ДТП").to_frame().T)

# Целевая переменная: диапазон и наличие нулей (важно для MAPE)
t = df_raw["Number_of_Casualties"]
print(f"\nNumber_of_Casualties: min = {t.min()}, max = {t.max()}, значений ≤ 0: {(t <= 0).sum()}")

# ----------------------------------------------------------------------

rows_before = len(df_raw)
df = df_raw.copy()

# 1. Пропуски (по всем столбцам исходной таблицы)
na_before = df.isna().sum()
df = df.dropna()
n_na = rows_before - len(df)

# 2. Служебные «значения-заглушки» в категориях
bad_values = {"Urban_or_Rural_Area": ["Unallocated"], "Vehicle_Type": ["Data missing or out of range"]}
mask_bad = pd.Series(False, index=df.index)
for col, bad in bad_values.items():
    mask_bad |= df[col].isin(bad)
n_bad = int(mask_bad.sum())
df = df.loc[~mask_bad]

# 3. Полные дубликаты ищем по ИСХОДНОЙ строке целиком (все 14 столбцов, включая дату и координаты),
#    чтобы не удалить разные ДТП, совпавшие лишь по оставшимся признакам.
n_dup = int(df.duplicated().sum())
df = df.drop_duplicates()

# 4. Признаки из даты. Сам столбец даты в модель не идёт: календарные признаки полезнее.
dates = pd.to_datetime(df["Accident Date"], format="%d-%m-%Y")
df["Month"] = dates.dt.month                                  # месяц 1..12
df["Is_Weekend"] = (dates.dt.dayofweek >= 5).astype(int)      # 1 - суббота/воскресенье

# 5. Столбцы, исключаемые из признаков
drop_cols = ["Index", "Accident Date", "District Area", "Accident_Severity"]
df = df.reset_index(drop=True)
severity = df["Accident_Severity"].copy()      # сохраняем ТОЛЬКО для контрольного эксперимента в конце
df = df.drop(columns=drop_cols)

# 6. Целевая переменная должна быть положительной (для MAPE)
assert (df["Number_of_Casualties"] >= 1).all()

print("Пропуски, удалённые по столбцам:")
display(na_before[na_before > 0].rename("Пропусков").to_frame())
print(f"Строк исходно:                         {rows_before:,}")
print(f"Удалено строк с пропусками:            {n_na:,}")
print(f"Удалено строк-«заглушек» категорий:    {n_bad:,}")
print(f"Удалено полных дубликатов:             {n_dup:,}")
print(f"Итоговый размер после предобработки:   {df.shape[0]:,} строк × {df.shape[1]} столбцов")
print(f"Осталось пропусков: {int(df.isna().sum().sum())};  доля сохранённых данных: {len(df)/rows_before:.1%}")
print("\nИсключены из признаков:", drop_cols)
print("Столбцы итогового датафрейма:", df.columns.tolist())

# ----------------------------------------------------------------------

target = "Number_of_Casualties"
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

counts = df[target].value_counts().sort_index()
axes[0].bar(counts.index[:12], counts.values[:12], color="steelblue", edgecolor="black")
axes[0].set_title("Распределение числа пострадавших (значения 1–12)")
axes[0].set_xlabel("Number_of_Casualties")
axes[0].set_ylabel("Количество ДТП")
axes[0].set_xticks(range(1, 13))

axes[1].bar(counts.index, counts.values, color="steelblue", edgecolor="black")
axes[1].set_yscale("log")
axes[1].set_title("То же, весь диапазон, логарифмическая ось Y")
axes[1].set_xlabel("Number_of_Casualties")
axes[1].set_ylabel("Количество ДТП (log)")
plt.tight_layout()
plt.show()

stats = df[target].describe(percentiles=[.5, .75, .9, .99, .999])
display(stats.rename("Number_of_Casualties").to_frame().T)
share = (counts / counts.sum() * 100).round(2)
print("Доля ДТП по числу пострадавших, %:")
display(share.head(8).rename("Доля, %").to_frame().T)
print(f"Среднее = {df[target].mean():.3f}, медиана = {df[target].median():.0f}, "
      f"дисперсия = {df[target].var():.3f}, асимметрия = {df[target].skew():.2f}")

# ----------------------------------------------------------------------

numeric_features = ["Number_of_Vehicles", "Latitude", "Longitude", "Month", "Is_Weekend"]
categorical_features = ["Light_Conditions", "Road_Surface_Conditions", "Road_Type",
                        "Urban_or_Rural_Area", "Weather_Conditions", "Vehicle_Type"]

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
fig.suptitle("Распределения числовых признаков", fontsize=20)
for ax, col in zip(axes.flat, numeric_features):
    if col in ("Is_Weekend", "Number_of_Vehicles", "Month"):
        vc = df[col].value_counts().sort_index()
        vc = vc[vc.index <= 8] if col == "Number_of_Vehicles" else vc
        ax.bar(vc.index.astype(int), vc.values, color="seagreen", edgecolor="black")
        ax.set_xticks(vc.index.astype(int))
    else:
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color="seagreen")
    ax.set_title(col)
    ax.set_xlabel(col)
axes.flat[-1].axis("off")
plt.tight_layout()
plt.show()

display(df[numeric_features].describe().T)

# ----------------------------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle("Распределение категориальных признаков", fontsize=18)
for ax, col in zip(axes.flat, categorical_features):
    df[col].value_counts().plot(kind="bar", ax=ax, color="salmon", edgecolor="black")
    ax.set_title(col)
    ax.set_ylabel("Количество")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.5)
plt.tight_layout()
plt.show()

for col in categorical_features:
    freq = df[col].value_counts()
    print(f"{col} ({len(freq)} категорий):")
    display(pd.DataFrame({"Количество": freq, "Доля, %": freq / len(df) * 100}).round(2))

# ----------------------------------------------------------------------

# Связь целевой переменной с признаками: средние значения по группам
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for ax, col in zip(axes.flat, categorical_features):
    order = df.groupby(col)[target].mean().sort_values().index
    try:
        sns.barplot(data=df, x=col, y=target, order=order, ax=ax, color="lightsteelblue",
                    errorbar=None, edgecolor="black")
    except TypeError:                      # seaborn < 0.12 не знает параметра errorbar
        sns.barplot(data=df, x=col, y=target, order=order, ax=ax, color="lightsteelblue",
                    ci=None, edgecolor="black")
    ax.axhline(df[target].mean(), color="red", linestyle="--", label="среднее по выборке")
    ax.set_title(f"Среднее число пострадавших по {col}")
    ax.tick_params(axis="x", rotation=60)
    ax.set_xlabel("")
axes.flat[0].legend()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
by_veh = df[df["Number_of_Vehicles"] <= 8].groupby("Number_of_Vehicles")[target].mean()
axes[0].plot(by_veh.index, by_veh.values, marker="o", color="darkorange")
axes[0].set_title("Среднее число пострадавших по числу транспортных средств")
axes[0].set_xlabel("Number_of_Vehicles")
axes[0].set_ylabel("Среднее Number_of_Casualties")
sev = df_raw.loc[df_raw["Number_of_Casualties"] >= 1].groupby("Accident_Severity")[target].mean()
sev = sev.reindex(["Slight", "Serious", "Fatal"])
axes[1].bar(sev.index, sev.values, color="indianred", edgecolor="black")
axes[1].set_title("Среднее число пострадавших по тяжести ДТП (в признаки не входит)")
axes[1].set_ylabel("Среднее Number_of_Casualties")
plt.tight_layout()
plt.show()

print("Среднее число пострадавших по числу ТС (1–8):")
display(by_veh.round(3).rename("Среднее").to_frame().T)
print("Среднее число пострадавших по тяжести ДТП:")
display(sev.round(3).rename("Среднее").to_frame().T)
for col in ["Light_Conditions", "Road_Type", "Urban_or_Rural_Area", "Vehicle_Type"]:
    print(f"\n{col}:")
    display(df.groupby(col)[target].agg(["count", "mean", "median"]).sort_values("mean", ascending=False).round(3))

# ----------------------------------------------------------------------

if "df" not in globals():
    raise NameError("Нет датафрейма df: сначала выполните ячейки загрузки и очистки данных (запуск сверху вниз).")

outlier_features = ["Number_of_Casualties", "Number_of_Vehicles", "Latitude", "Longitude", "Month"]

# Boxplot построен средствами matplotlib (не зависит от версии seaborn)
fig, axes = plt.subplots(1, 5, figsize=(22, 5))
for ax, col in zip(axes, outlier_features):
    ax.boxplot(df[col].to_numpy(), patch_artist=True,
               boxprops=dict(facecolor="lightblue"),
               flierprops=dict(marker="o", markersize=2, alpha=0.4))
    ax.set_title(f"Выбросы: {col}")
    ax.set_xticks([])
plt.suptitle("Boxplot числовых столбцов после очистки", fontsize=15)
plt.tight_layout()
plt.show()

iqr_rows = []
for col in outlier_features:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (df[col] < lower) | (df[col] > upper)
    iqr_rows.append({"Признак": col, "Q1": q1, "Q3": q3, "IQR": iqr,
                     "Нижняя граница": lower, "Верхняя граница": upper,
                     "Значений вне границ": int(mask.sum()), "Доля, %": mask.mean() * 100,
                     "Максимум": df[col].max()})
iqr_summary = pd.DataFrame(iqr_rows).set_index("Признак")
display(iqr_summary.round(3))

print("Хвосты целевой переменной и числа ТС:")
for col in ["Number_of_Casualties", "Number_of_Vehicles"]:
    for thr in (5, 10, 20):
        share = (df[col] > thr).mean() * 100
        print(f"  {col} > {thr}: {int((df[col] > thr).sum()):>6,} строк ({share:.3f} %)")
print("\nГеография (UK: широта ≈ 49.9–60.9, долгота ≈ −8.7…1.8):")
print(f"  Latitude  [{df['Latitude'].min():.3f}; {df['Latitude'].max():.3f}]")
print(f"  Longitude [{df['Longitude'].min():.3f}; {df['Longitude'].max():.3f}]")

# ----------------------------------------------------------------------

# Кодирование для корреляций и VIF: one-hot. Из каждой группы dummy исключается одна БАЗОВАЯ категория
# (иначе столбцы линейно зависимы: сумма dummy группы = 1 = константа). Базой выбрана САМАЯ ЧАСТАЯ категория:
# так коэффициенты читаются как отличие от «типичного» ДТП (Car, Daylight, Dry, ...), а VIF не искажается редкой базой.
X_all = df.drop(columns=target)
base = {col: df[col].value_counts().idxmax() for col in categorical_features}
X_enc = pd.get_dummies(X_all, columns=categorical_features, dtype=float)
X_enc = X_enc.drop(columns=[f"{col}_{cat}" for col, cat in base.items()])
df_enc = pd.concat([X_enc, df[target]], axis=1)

# Для сравнения: вариант «база = первая по алфавиту» (поведение drop_first=True)
X_enc_alpha = pd.get_dummies(X_all, columns=categorical_features, drop_first=True, dtype=float)

print(f"Признаков после one-hot кодирования: {X_enc.shape[1]}  (числовых {len(numeric_features)} + dummy {X_enc.shape[1] - len(numeric_features)})")
print("Базовые (исключённые) категории:")
for col, cat in base.items():
    print(f"  {col}: {cat}  ({(df[col] == cat).mean():.1%} наблюдений)")
display(df_enc.head())

# ----------------------------------------------------------------------

# Корреляционная матрица Пирсона: числовые признаки + целевая переменная
num_cols = numeric_features + [target]
corr_num = df[num_cols].corr(method="pearson")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_num, annot=True, cmap="coolwarm", fmt=".3f", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Корреляционная матрица (Пирсон): числовые признаки и целевая переменная")
plt.tight_layout()
plt.show()

# Сравнение Пирсона и Спирмена для связи с целевой переменной (целевая переменная дискретна и скошена)
cmp_corr = pd.DataFrame({
    "Пирсон": df[num_cols].corr(method="pearson")[target],
    "Спирмен": df[num_cols].corr(method="spearman")[target],
}).drop(target)
display(cmp_corr.round(4))

# ----------------------------------------------------------------------

# Полная матрица: все столбцы после кодирования (признаки + dummy + целевая переменная)
corr_matrix = df_enc.corr(method="pearson")
plt.figure(figsize=(17, 14))
sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True, linewidths=0.2,
            cbar_kws={"shrink": 0.7}, xticklabels=True, yticklabels=True)
plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.title(f"Полная корреляционная матрица ({corr_matrix.shape[0]} столбцов, включая целевую переменную)")
plt.tight_layout()
plt.show()

# Самые сильные пары между ПРИЗНАКАМИ (без целевой переменной)
feat_corr = corr_matrix.drop(index=target, columns=target)
pairs = (feat_corr.where(np.triu(np.ones(feat_corr.shape, dtype=bool), k=1))
         .stack().rename("Корреляция").reset_index())
pairs.columns = ["Признак 1", "Признак 2", "Корреляция"]
pairs["|r|"] = pairs["Корреляция"].abs()
print("Топ-12 пар признаков по модулю корреляции:")
display(pairs.sort_values("|r|", ascending=False).head(12).drop(columns="|r|").reset_index(drop=True).round(4))
print(f"Пар с |r| > 0.7: {(pairs['|r|'] > 0.7).sum()};  с |r| > 0.5: {(pairs['|r|'] > 0.5).sum()}")

price_corr = corr_matrix[target].drop(target).sort_values(key=lambda s: s.abs(), ascending=False)
print("\nКорреляция признаков с целевой переменной Number_of_Casualties (топ-12 по модулю):")
display(price_corr.head(12).rename("Корреляция с целевой").to_frame().round(4))

# ----------------------------------------------------------------------

def compute_vif(features):
    """VIF каждого столбца: 1 / (1 - R²) регрессии этого столбца на все остальные."""
    X_c = add_constant(features)                         # константа нужна, иначе VIF считается неверно
    vif = [variance_inflation_factor(X_c.values, i) for i in range(1, X_c.shape[1])]
    return (pd.DataFrame({"Признак": X_c.columns[1:], "VIF": vif})
            .sort_values("VIF", ascending=False).reset_index(drop=True))

# На вход подаются ВСЕ признаки после one-hot (без целевой переменной)
vif_table = compute_vif(X_enc)
print(f"VIF (база = самая частая категория) рассчитан для {len(vif_table)} признаков. Топ-15:")
display(vif_table.head(15).round(3))
print(f"Максимальный VIF: {vif_table['VIF'].max():.2f};  признаков с VIF > 5: {(vif_table['VIF'] > 5).sum()};  "
      f"с VIF > 10: {(vif_table['VIF'] > 10).sum()}")

# Сравнение: при редкой базовой категории (первая по алфавиту) VIF dummy-столбцов раздувается искусственно
vif_alpha = compute_vif(X_enc_alpha)
print(f"\nVIF при алфавитной базе (drop_first=True): максимум {vif_alpha['VIF'].max():.2f}, "
      f"признаков с VIF > 10: {(vif_alpha['VIF'] > 10).sum()}. Топ-6:")
display(vif_alpha.head(6).round(3))

fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharex=False)
for ax, tbl, ttl in zip(axes, [vif_table, vif_alpha], ["база = самая частая категория (используется в моделях)", "база = первая по алфавиту (для сравнения)"]):
    sns.barplot(data=tbl, y="Признак", x="VIF", color="steelblue", ax=ax)
    ax.axvline(5, color="orange", linestyle="--", label="VIF = 5")
    ax.axvline(10, color="red", linestyle="--", label="VIF = 10")
    ax.set_title(f"VIF: {ttl}", fontsize=11)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# Предобработка: числовые - StandardScaler, категориальные - OneHotEncoder (база - самая частая категория)
# drop = список базовых категорий (самых частых) в порядке categorical_features
def make_ohe(**kwargs):
    """OneHotEncoder с плотным выводом: параметр называется sparse_output (sklearn >= 1.2) или sparse (старее)."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, **kwargs)

one_hot_encoder = make_ohe(drop=[base[c] for c in categorical_features])
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("categorical", one_hot_encoder, categorical_features),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

print(f"Всего объектов:       {len(X):,}")
print(f"Обучающая выборка:    {X_train.shape[0]:,} объектов × {X_train.shape[1]} исходных признаков")
print(f"Тестовая выборка:     {X_test.shape[0]:,} объектов × {X_test.shape[1]} исходных признаков")
print(f"Доля тестовой выборки: {len(X_test) / len(X):.1%}")
print(f"Целевая переменная минимум: {y.min()} (MAPE без деления на ноль: {(y > 0).all()})")
print(f"Среднее целевой: train = {y_train.mean():.4f}, test = {y_test.mean():.4f};  "
      f"доли ДТП с 1 пострадавшим: train = {(y_train == 1).mean():.3f}, test = {(y_test == 1).mean():.3f}")

print(f"Std целевой: train = {y_train.std():.4f}, test = {y_test.std():.4f};  максимум: train = {y_train.max()}, test = {y_test.max()}")

feature_names = preprocessor.fit(X_train).get_feature_names_out()      # fit ТОЛЬКО на train
print(f"\nЧисло признаков после кодирования: {len(feature_names)}")
print(list(feature_names))

# ----------------------------------------------------------------------

# Демонстрация fit / transform / fit_transform и отсутствия утечки данных из тестовой выборки
demo_cols = ["Number_of_Vehicles", "Latitude"]
scaler = StandardScaler()

scaler.fit(X_train[demo_cols])                         # fit: запоминает среднее и std ТОЛЬКО обучающей выборки
print("fit: mean_ =", scaler.mean_.round(4), " scale_ (std) =", scaler.scale_.round(4))

train_scaled = scaler.transform(X_train[demo_cols])    # transform: применяет уже запомненные параметры
test_scaled = scaler.transform(X_test[demo_cols])      # то же правило к тесту (параметры НЕ пересчитываются)

same = StandardScaler().fit_transform(X_train[demo_cols])   # fit_transform = fit + transform на одних данных
print("fit_transform == fit + transform на train:", np.allclose(train_scaled, same))

print("\nСреднее после масштабирования, train:", train_scaled.mean(axis=0).round(6), "(строго 0 - так и должно быть)")
print("Среднее после масштабирования, test :", test_scaled.mean(axis=0).round(4),
      "(≠ 0: тест масштабируется по параметрам train - утечки нет)")

# ----------------------------------------------------------------------

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    "RMSE": "neg_root_mean_squared_error",
    "R2": "r2",
    "MAE": "neg_mean_absolute_error",
    "MAPE": make_scorer(mean_absolute_percentage_error, greater_is_better=False),
}

# Сетка параметра регуляризации alpha (логарифмическая, 10 значений) - общая для Ridge и Lasso
ALPHA_VALUES = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5])

regressors = {
    "Линейная регрессия": LinearRegression(),
    "Ridge": Ridge(),                                  # остальные параметры - по умолчанию
    "Lasso": Lasso(max_iter=10000, random_state=RANDOM_STATE),
}

def make_regression_pipeline(regressor):
    """Pipeline: кодирование/масштабирование -> стандартизация всех столбцов -> регрессор."""
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("feature_scaler", StandardScaler()),
        ("regressor", clone(regressor)),
    ])

def evaluate(y_true, y_pred):
    """Метрики качества. RMSE считается через MSE: RMSE = sqrt(MSE). MAPE переводится в проценты."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE, %": mean_absolute_percentage_error(y_true, y_pred) * 100,
    }

def fit_and_score(model_name, regressor, pipeline_factory, X_tr, y_tr, X_te, y_te, searches):
    """Обучение одной модели: Ridge/Lasso - через GridSearchCV по alpha, обычная МНК - через cross_validate."""
    pipeline = pipeline_factory(regressor)
    if model_name in ("Ridge", "Lasso"):
        search = GridSearchCV(pipeline, param_grid={"regressor__alpha": ALPHA_VALUES},
                              scoring=scoring, refit="RMSE", cv=cv, n_jobs=N_JOBS)
        search.fit(X_tr, y_tr)
        searches[model_name] = search
        fitted = search.best_estimator_
        i = search.best_index_
        res = search.cv_results_
        cv_part = {"CV RMSE, среднее": -res["mean_test_RMSE"][i], "CV RMSE, std": res["std_test_RMSE"][i],
                   "CV R², среднее": res["mean_test_R2"][i], "CV MAE, среднее": -res["mean_test_MAE"][i],
                   "CV MAPE, %": -res["mean_test_MAPE"][i] * 100}
        alpha = search.best_params_["regressor__alpha"]
    else:
        scores = cross_validate(pipeline, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=N_JOBS)
        fitted = clone(pipeline).fit(X_tr, y_tr)
        cv_part = {"CV RMSE, среднее": (-scores["test_RMSE"]).mean(), "CV RMSE, std": (-scores["test_RMSE"]).std(ddof=1),
                   "CV R², среднее": scores["test_R2"].mean(), "CV MAE, среднее": (-scores["test_MAE"]).mean(),
                   "CV MAPE, %": (-scores["test_MAPE"]).mean() * 100}
        alpha = "не применяется"
    y_tr_pred = fitted.predict(X_tr)                    # прогноз на обучающей выборке
    y_te_pred = fitted.predict(X_te)                    # прогноз на тестовой выборке (итоговые метрики - по нему)
    tr, te = evaluate(y_tr, y_tr_pred), evaluate(y_te, y_te_pred)
    row = {"Модель": model_name, "Выбранная alpha": alpha, **cv_part,
           **{f"Train {k}": v for k, v in tr.items()}, **{f"Тест {k}": v for k, v in te.items()}}
    return fitted, row

# Базовый ориентир: «модель», всегда предсказывающая среднее обучающей выборки
dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
base_metrics = evaluate(y_test, dummy.predict(X_test))
print("Ориентир (прогноз = среднее train) на тесте:", {k: round(v, 4) for k, v in base_metrics.items()})
print(f"MAPE при прогнозе «всегда 1 пострадавший»: {mean_absolute_percentage_error(y_test, np.ones(len(y_test))) * 100:.2f} %")

# ----------------------------------------------------------------------

baseline_models, baseline_searches, baseline_rows = {}, {}, []
for model_name, regressor in regressors.items():
    fitted, row = fit_and_score(model_name, regressor, make_regression_pipeline,
                                X_train, y_train, X_test, y_test, baseline_searches)
    baseline_models[model_name] = fitted
    baseline_rows.append(row)

baseline_results = pd.DataFrame(baseline_rows).set_index("Модель")
print("Качество моделей на исходных признаках (alpha для Ridge/Lasso выбрана по CV):")
display(baseline_results[["Выбранная alpha", "CV RMSE, среднее", "CV RMSE, std", "CV R², среднее", "CV MAE, среднее", "CV MAPE, %"]].round(4))
display(baseline_results[[c for c in baseline_results.columns if c.startswith(("Train", "Тест"))]].round(4))

# ----------------------------------------------------------------------

# Как выбирается alpha: средний CV RMSE для каждого значения сетки
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
alpha_rows = []
for ax, name in zip(axes, ["Ridge", "Lasso"]):
    res = baseline_searches[name].cv_results_
    alphas = res["param_regressor__alpha"].astype(float)
    rmse = -res["mean_test_RMSE"]
    ax.errorbar(alphas, rmse, yerr=res["std_test_RMSE"], marker="o", capsize=3)
    ax.axvline(baseline_searches[name].best_params_["regressor__alpha"], color="red", linestyle="--", label="выбранная alpha")
    ax.set_xscale("log")
    ax.set_title(f"{name}: CV RMSE в зависимости от alpha")
    ax.set_xlabel("alpha (log)")
    ax.set_ylabel("CV RMSE")
    ax.legend()
    for a, r in zip(alphas, rmse):
        alpha_rows.append({"Модель": name, "alpha": a, "CV RMSE": r})
plt.tight_layout()
plt.show()
display(pd.DataFrame(alpha_rows).pivot(index="alpha", columns="Модель", values="CV RMSE").round(5))

# ----------------------------------------------------------------------

baseline_feature_names = baseline_models["Линейная регрессия"].named_steps["preprocessor"].get_feature_names_out()
coef_frame = pd.DataFrame({name: model.named_steps["regressor"].coef_ for name, model in baseline_models.items()},
                          index=baseline_feature_names)
print("Коэффициенты моделей (признаки стандартизованы: изменение прогноза при росте признака на 1 std):")
display(coef_frame.round(4))

summary_rows = []
for model_name, model in baseline_models.items():
    coef = pd.Series(model.named_steps["regressor"].coef_, index=baseline_feature_names)
    top = coef.abs().sort_values(ascending=False).head(5).index
    summary_rows.append({
        "Модель": model_name,
        "Свободный член": model.named_steps["regressor"].intercept_,
        "Ненулевых коэффициентов": int(np.count_nonzero(np.abs(coef.values) > 1e-8)),
        "Нулевых коэффициентов": int(np.sum(np.abs(coef.values) <= 1e-8)),
        "Топ-5 по |коэффициенту|": ", ".join(f"{f} ({coef[f]:.3f})" for f in top),
    })
display(pd.DataFrame(summary_rows).set_index("Модель"))

zero_lasso = coef_frame.index[np.abs(coef_frame["Lasso"]) <= 1e-8].tolist()
print("Признаки, обнулённые Lasso:", zero_lasso)

# Уравнение линейной регрессии (в стандартизованных признаках): топ-6 слагаемых
lin = coef_frame["Линейная регрессия"].sort_values(key=abs, ascending=False)
intercept = baseline_models["Линейная регрессия"].named_steps["regressor"].intercept_
terms = " ".join(f"{c:+.3f}·z({n})" for n, c in lin.head(6).items())
print(f"\nŷ = {intercept:.3f} {terms} + ... (ещё {len(lin) - 6} слагаемых), где z(·) - стандартизованный признак")

# ----------------------------------------------------------------------

def residual_plot(models, title, X_te, y_te, ylim=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (model_name, fitted_model) in zip(axes, models.items()):
        y_pred = fitted_model.predict(X_te)
        residuals = y_te.to_numpy() - y_pred
        ax.scatter(y_pred, residuals, s=5, alpha=0.08)
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(model_name)
        ax.set_xlabel("Предсказанное число пострадавших")
        ax.grid(alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel("Остаток: факт − прогноз")
    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.show()

residual_plot(baseline_models, "Графики остатков моделей на тестовой выборке (исходные признаки)", X_test, y_test)
residual_plot(baseline_models, "То же, увеличенный масштаб по оси остатков (−4 … 10)", X_test, y_test, ylim=(-4, 10))

# Количественная проверка гомоскедастичности: разброс остатков по интервалам прогноза
lin_model = baseline_models["Линейная регрессия"]
pred = lin_model.predict(X_test)
res = y_test.to_numpy() - pred
bins = pd.qcut(pred, q=5, duplicates="drop")
res_by_bin = (pd.DataFrame({"Прогноз (квантильный интервал)": bins, "Остаток": res})
              .groupby("Прогноз (квантильный интервал)", observed=True)["Остаток"]
              .agg(["count", "mean", "std"]).rename(columns={"count": "Объектов", "mean": "Средний остаток", "std": "Std остатка"}))
print("Остатки линейной регрессии по пяти квантильным интервалам прогноза:")
display(res_by_bin.round(4))

# Тест Бройша-Пагана: H0 - дисперсия остатков постоянна (гомоскедастичность)
bp_rows = []
for model_name, fitted_model in baseline_models.items():
    Z = fitted_model[:-1].transform(X_test)                 # признаки теста после preprocessor + scaler
    r = y_test.to_numpy() - fitted_model.predict(X_test)
    lm, lm_p, f_stat, f_p = het_breuschpagan(r, add_constant(Z))
    bp_rows.append({"Модель": model_name, "LM-статистика": lm, "p-value (LM)": lm_p, "F-статистика": f_stat, "p-value (F)": f_p})
display(pd.DataFrame(bp_rows).set_index("Модель"))
print(f"Средний остаток: {res.mean():.4f}; доля отрицательных прогнозов: {(pred < 0).mean():.2%}; "
      f"минимальный прогноз: {pred.min():.3f}; максимальный прогноз: {pred.max():.3f}")

# ----------------------------------------------------------------------

# Шаги: one-hot + стандартизация -> дополнительная стандартизация -> полный PCA.
# Все преобразования обучаются ТОЛЬКО на обучающей выборке, тест лишь преобразуется.
pca_preprocessor = clone(preprocessor)
X_train_encoded = pca_preprocessor.fit_transform(X_train)
X_test_encoded = pca_preprocessor.transform(X_test)

pca_feature_scaler = StandardScaler()
X_train_scaled_for_pca = pca_feature_scaler.fit_transform(X_train_encoded)
X_test_scaled_for_pca = pca_feature_scaler.transform(X_test_encoded)

pca_full = PCA(random_state=RANDOM_STATE)
X_train_pca_full = pca_full.fit_transform(X_train_scaled_for_pca)
X_test_pca_full = pca_full.transform(X_test_scaled_for_pca)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)          # накопленная доля = частичные суммы ряда
n_components_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)

pca_variance_table = pd.DataFrame({
    "Главная компонента": np.arange(1, len(explained_variance) + 1),
    "Собственное значение": pca_full.explained_variance_,
    "Объяснённая дисперсия, %": explained_variance * 100,
    "Накопленная дисперсия, %": cumulative_variance * 100,
})
print(f"Число признаков до PCA: {X_train_scaled_for_pca.shape[1]}")
print(f"Размер train после полного PCA: {X_train_pca_full.shape};  test: {X_test_pca_full.shape}")
print(f"Компонент для объяснения ≥ 95 % дисперсии: {n_components_95}")
print(f"Размер train при {n_components_95} компонентах: {X_train_pca_full[:, :n_components_95].shape};  "
      f"test: {X_test_pca_full[:, :n_components_95].shape}")
display(pca_variance_table.round(3).set_index("Главная компонента"))

# ----------------------------------------------------------------------

components = np.arange(1, len(explained_variance) + 1)

# Точка излома: начало «полки» - первая компонента, после которой собственное значение
# перестаёт заметно падать (разность соседних λ < 0.05)
eig = pca_full.explained_variance_
drops = -np.diff(eig)                                  # падение λ при переходе k -> k+1
elbow = int(np.argmax(drops < 0.05) + 1)
print("Падение λ между соседними компонентами (первые 8):", np.round(drops[:8], 3))
print(f"Собственные значения > 1 (критерий Кайзера): {(eig > 1).sum()} компонент(ы)")
print(f"Точка излома (начало полки, падение λ < 0.05): компонента {elbow}")

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# График каменистой осыпи (scree plot): собственные значения по номерам компонент
axes[0].plot(components, eig, marker="o", color="darkorange", linewidth=2)
axes[0].axhline(1, color="gray", linestyle="--", label="λ = 1 (критерий Кайзера)")
axes[0].axvline(elbow, color="red", linestyle=":", label=f"излом ≈ PC{elbow}")
axes[0].set_title("График каменистой осыпи (scree plot)")
axes[0].set_xlabel("Номер главной компоненты")
axes[0].set_ylabel("Собственное значение λ")
axes[0].set_xticks(components)
axes[0].legend()

axes[1].bar(components, explained_variance * 100, alpha=0.5, label="Доля дисперсии компоненты")
axes[1].set_xlabel("Номер главной компоненты")
axes[1].set_ylabel("Объяснённая дисперсия одной компоненты, %")
axes[1].set_xticks(components)
ax2 = axes[1].twinx()
ax2.plot(components, cumulative_variance * 100, marker="o", color="steelblue", linewidth=2, label="Накопленная дисперсия")
ax2.axhline(95, color="red", linestyle="--", label="Порог 95 %")
ax2.axvline(n_components_95, color="black", linestyle=":", label=f"Выбрано компонент: {n_components_95}")
ax2.set_ylabel("Накопленная объяснённая дисперсия, %")
ax2.set_ylim(0, 103)
h1, l1 = axes[1].get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
axes[1].legend(h1 + h2, l1 + l2, loc="center right")
axes[1].set_title("Объяснённая и накопленная дисперсия")
plt.tight_layout()
plt.show()

for thr in (0.80, 0.90, 0.95, 0.99):
    print(f"Для ≥ {thr:.0%} дисперсии нужно компонент: {int(np.argmax(cumulative_variance >= thr) + 1)}")

# ----------------------------------------------------------------------

# Нагрузки компонент: коэффициенты линейных комбинаций исходных (стандартизованных) признаков
pc_cols = [f"PC{i}" for i in range(1, n_components_95 + 1)]
loadings = pd.DataFrame(pca_full.components_[:n_components_95].T, index=feature_names, columns=pc_cols)

for pc in ["PC1", "PC2", "PC3", "PC4"]:
    top = loadings[pc].abs().sort_values(ascending=False).head(6).index
    print(f"{pc} (объясняет {explained_variance[int(pc[2:]) - 1] * 100:.2f} % дисперсии) - сильнейшие нагрузки:")
    display(loadings.loc[top, [pc]].round(3).T)

plt.figure(figsize=(14, 10))
sns.heatmap(loadings.iloc[:, :8], cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Нагрузки первых 8 главных компонент на исходные признаки")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------

# Проверка устранения мультиколлинеарности: корреляции и VIF сохранённых компонент (train)
Z_pc = X_train_pca_full[:, :n_components_95]
pc_corr = np.corrcoef(Z_pc, rowvar=False)
off_diag = pc_corr - np.eye(n_components_95)
max_abs_corr = np.abs(off_diag).max()

pc_names = [f"PC{i}" for i in range(1, n_components_95 + 1)]
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(pc_corr, index=pc_names, columns=pc_names).where(lambda d: d.abs() >= 0.005, 0.0),
            annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True,
            linewidths=0.4, annot_kws={"size": 6}, cbar_kws={"shrink": 0.8})
plt.title(f"Матрица корреляций {n_components_95} сохранённых главных компонент (train)")
plt.tight_layout()
plt.show()

vif_pc = compute_vif(pd.DataFrame(Z_pc, columns=pc_names))
print(f"Число компонент в матрице: {n_components_95}")
print(f"Максимальная |корреляция| между разными компонентами: {max_abs_corr:.3e}")
print(f"VIF компонент: min = {vif_pc['VIF'].min():.6f}, max = {vif_pc['VIF'].max():.6f}")
print(f"Для сравнения, VIF исходных признаков: max = {vif_table['VIF'].max():.3f}")

# ----------------------------------------------------------------------

def make_pca_pipeline(regressor):
    """preprocessor -> стандартизация -> PCA (95 % дисперсии) -> регрессор. Всё обучается заново в каждом фолде CV."""
    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("feature_scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
        ("regressor", clone(regressor)),
    ])

pca_models, pca_searches, pca_rows = {}, {}, []
for model_name, regressor in regressors.items():
    fitted, row = fit_and_score(model_name, regressor, make_pca_pipeline,
                                X_train, y_train, X_test, y_test, pca_searches)
    row = {"Модель": row.pop("Модель"), "Компонент": fitted.named_steps["pca"].n_components_, **row}
    pca_models[model_name] = fitted
    pca_rows.append(row)

pca_results = pd.DataFrame(pca_rows).set_index("Модель")
print("Качество моделей на главных компонентах (alpha для Ridge/Lasso выбрана по CV):")
display(pca_results[["Выбранная alpha", "Компонент", "CV RMSE, среднее", "CV RMSE, std", "CV R², среднее", "CV MAE, среднее", "CV MAPE, %"]].round(4))
display(pca_results[[c for c in pca_results.columns if c.startswith(("Train", "Тест"))]].round(4))

pca_coef_rows = []
for model_name, model in pca_models.items():
    reg = model.named_steps["regressor"]
    row = {"Модель": model_name, "Свободный член": reg.intercept_}
    row.update({f"PC{i + 1}": v for i, v in enumerate(reg.coef_)})
    pca_coef_rows.append(row)
print("Коэффициенты регрессий в координатах главных компонент:")
display(pd.DataFrame(pca_coef_rows).set_index("Модель").round(4))

# ----------------------------------------------------------------------

residual_plot(pca_models, "Графики остатков моделей на тестовой выборке (главные компоненты)", X_test, y_test)

bp_rows = []
for model_name, fitted_model in pca_models.items():
    Z = fitted_model[:-1].transform(X_test)
    r = y_test.to_numpy() - fitted_model.predict(X_test)
    lm, lm_p, f_stat, f_p = het_breuschpagan(r, add_constant(Z))
    bp_rows.append({"Модель": model_name, "LM-статистика": lm, "p-value (LM)": lm_p, "F-статистика": f_stat, "p-value (F)": f_p})
display(pd.DataFrame(bp_rows).set_index("Модель"))

# ----------------------------------------------------------------------

comparison_rows = []
for representation, results_df in [("Исходные признаки", baseline_results), ("Главные компоненты PCA", pca_results)]:
    for model_name, row in results_df.iterrows():
        comparison_rows.append({
            "Представление признаков": representation, "Модель": model_name,
            "CV RMSE": row["CV RMSE, среднее"], "Тест RMSE": row["Тест RMSE"],
            "CV R²": row["CV R², среднее"], "Тест R²": row["Тест R2"],
            "Тест MAE": row["Тест MAE"], "CV MAPE, %": row["CV MAPE, %"], "Тест MAPE, %": row["Тест MAPE, %"],
        })
comparison_results = pd.DataFrame(comparison_rows)
print("Сводное сравнение всех шести моделей:")
display(comparison_results.round(4))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (metric_col, plot_title) in zip(axes, [("Тест RMSE", "RMSE на тесте"), ("Тест R²", "R² на тесте"), ("Тест MAPE, %", "MAPE на тесте, %")]):
    sns.barplot(data=comparison_results, x="Модель", y=metric_col, hue="Представление признаков", ax=ax)
    ax.set_title(plot_title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    if ax is not axes[0] and ax.get_legend() is not None:
        ax.get_legend().remove()
axes[0].legend(title="Признаки", loc="best")
plt.suptitle("Сравнение качества до PCA и после PCA", fontsize=15)
plt.tight_layout()
plt.show()

# Изменение метрик при переходе к PCA (на тесте)
delta = []
for model_name in regressors:
    b, p = baseline_results.loc[model_name], pca_results.loc[model_name]
    delta.append({"Модель": model_name,
                  "ΔRMSE (PCA − исх.)": p["Тест RMSE"] - b["Тест RMSE"],
                  "ΔR²": p["Тест R2"] - b["Тест R2"],
                  "ΔMAPE, п.п.": p["Тест MAPE, %"] - b["Тест MAPE, %"],
                  "Вердикт по RMSE": "PCA лучше" if p["Тест RMSE"] < b["Тест RMSE"] else "PCA хуже"})
display(pd.DataFrame(delta).set_index("Модель").round(5))

# ----------------------------------------------------------------------

# Проверка: PCA без отбрасывания компонент - это лишь поворот осей, качество МНК не должно измениться
full_pca_pipe = Pipeline([("preprocessor", clone(preprocessor)), ("feature_scaler", StandardScaler()),
                          ("pca", PCA(n_components=None)), ("regressor", LinearRegression())]).fit(X_train, y_train)
m_full = evaluate(y_test, full_pca_pipe.predict(X_test))
m_base = evaluate(y_test, baseline_models["Линейная регрессия"].predict(X_test))
print(f"Линейная регрессия на исходных признаках, тест:      RMSE = {m_base['RMSE']:.6f}, R² = {m_base['R2']:.6f}")
print(f"Линейная регрессия на ВСЕХ {full_pca_pipe.named_steps['pca'].n_components_} компонентах, тест: RMSE = {m_full['RMSE']:.6f}, R² = {m_full['R2']:.6f}")

# Как меняется качество с числом сохранённых компонент (МНК, тест)
k_rows = []
Z_tr_all, Z_te_all = X_train_pca_full, X_test_pca_full
for k in [1, 2, 3, 5, 8, 10, 15, n_components_95, 20, 25, 30, Z_tr_all.shape[1]]:
    if k > Z_tr_all.shape[1]:
        continue
    lr = LinearRegression().fit(Z_tr_all[:, :k], y_train)
    m = evaluate(y_test, lr.predict(Z_te_all[:, :k]))
    k_rows.append({"Компонент": k, "Накопл. дисперсия, %": cumulative_variance[k - 1] * 100, "Тест RMSE": m["RMSE"], "Тест R²": m["R2"], "Тест MAPE, %": m["MAPE, %"]})
k_table = pd.DataFrame(k_rows).drop_duplicates("Компонент").set_index("Компонент")
display(k_table.round(4))

plt.figure(figsize=(9, 5))
plt.plot(k_table.index, k_table["Тест R²"], marker="o")
plt.axvline(n_components_95, color="black", linestyle=":", label=f"выбрано {n_components_95} (95 % дисперсии)")
plt.xlabel("Число сохранённых главных компонент")
plt.ylabel("R² на тесте (МНК)")
plt.title("Качество регрессии в зависимости от числа компонент")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# ----------------------------------------------------------------------

# Правило выбора лучшей модели: минимальный CV RMSE (тест в выборе не участвует).
# При равенстве в пределах шума (разность < std между фолдами) предпочитаем более простую модель.
all_results = pd.concat({"Исходные признаки": baseline_results, "PCA": pca_results}, names=["Представление", "Модель"])
ranking = all_results[["CV RMSE, среднее", "CV RMSE, std", "Тест RMSE", "Тест R2", "Тест MAPE, %"]].sort_values("CV RMSE, среднее")
display(ranking.round(4))

best_key = ranking.index[0]
best_cv = ranking.iloc[0]["CV RMSE, среднее"]
noise = ranking.iloc[0]["CV RMSE, std"]
tied = ranking[ranking["CV RMSE, среднее"] - best_cv < noise]
print(f"Лучшая по CV RMSE: {best_key}  (CV RMSE = {best_cv:.4f}; разброс между фолдами ≈ {noise:.4f})")
print("Модели, неотличимые от лучшей в пределах разброса фолдов:", [k for k in tied.index])

# Контрольная проверка: сравнение с ориентиром «прогноз = среднее»
best_te = ranking.iloc[0]
print(f"\nУлучшение RMSE относительно прогноза средним: {base_metrics['RMSE']:.4f} → {best_te['Тест RMSE']:.4f} "
      f"({(1 - best_te['Тест RMSE'] / base_metrics['RMSE']) * 100:.2f} %)")

# ----------------------------------------------------------------------

# Дополнительный эксперимент (за рамками задания): как можно повысить точность?
from sklearn.linear_model import PoissonRegressor

def run_variant(name, X_tr, X_te, regressor, num, cat):
    pre = ColumnTransformer([("n", StandardScaler(), num), ("c", make_ohe(), cat)])
    pipe = Pipeline([("pre", pre), ("reg", regressor)]).fit(X_tr, y_train)
    pred = pipe.predict(X_te)
    return {"Вариант": name, **evaluate(y_test, pred), "Макс. прогноз": pred.max()}

rows = [run_variant("Линейная регрессия (базовая)", X_train, X_test, LinearRegression(), numeric_features, categorical_features),
        run_variant("Пуассоновская регрессия (целевая - счётчик)", X_train, X_test, PoissonRegressor(alpha=1e-4, max_iter=500), numeric_features, categorical_features)]

# Проверка гипотезы об утечке: добавим тяжесть ДТП (она известна только ПОСЛЕ ДТП - в основной модели не используется)
X_sev = X.assign(Accident_Severity=severity)
Xs_tr, Xs_te = X_sev.loc[X_train.index], X_sev.loc[X_test.index]
rows.append(run_variant("Линейная + Accident_Severity (утечка, только для сравнения)", Xs_tr, Xs_te, LinearRegression(),
                        numeric_features, categorical_features + ["Accident_Severity"]))
display(pd.DataFrame(rows).set_index("Вариант").round(4))

# ----------------------------------------------------------------------

