"""Загрузка SMS Spam Collection и фиксированное разбиение 60/20/20.

Все скрипты (train, export_model, evaluate) используют только этот модуль,
поэтому состав train/validation/test одинаков везде.
"""
import csv
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / 'data' / 'raw' / 'SMSSpamCollection'
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/228/sms+spam+collection'
SEED = 42
CLASSES = ['ham', 'spam']          # 0 -> ham, 1 -> spam
TEXT_COL = 'text'


def file_sha256(path: Path = RAW_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw() -> pd.DataFrame:
    # Исходный файл: TSV без заголовка, UTF-8, разделитель \t, без кавычек.
    df = pd.read_csv(RAW_PATH, sep='\t', header=None, names=['label', TEXT_COL],
                     quoting=csv.QUOTE_NONE, encoding='utf-8')
    return df


def load_clean() -> pd.DataFrame:
    """Документированная предобработка: удаляем точные дубликаты текста
    (одинаковые SMS в train и test дали бы утечку), кодируем метку spam=1."""
    df = load_raw()
    df = df.drop_duplicates(subset=TEXT_COL, keep='first').reset_index(drop=True)
    df['target'] = (df['label'] == 'spam').astype(int)
    return df


def split(df: pd.DataFrame | None = None):
    """20% test, затем 25% оставшегося — validation (итого 60/20/20), stratify."""
    if df is None:
        df = load_clean()
    X, y = df[[TEXT_COL]], df['target']
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=.25, random_state=SEED,
                                                  stratify=ydev)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest
