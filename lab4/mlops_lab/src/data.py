import csv
import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / 'data' / 'raw' / 'SMSSpamCollection'
SOURCE_URL = 'https://archive.ics.uci.edu/dataset/228/sms+spam+collection'
SEED = 42
CLASSES = ['ham', 'spam']
TEXT_COL = 'text'


def file_sha256(path: Path = RAW_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep='\t', header=None, names=['label', TEXT_COL],
                     quoting=csv.QUOTE_NONE, encoding='utf-8')
    return df


def load_clean() -> pd.DataFrame:
    df = load_raw()
    df = df.drop_duplicates(subset=TEXT_COL, keep='first').reset_index(drop=True)
    df['target'] = (df['label'] == 'spam').astype(int)
    return df


def split(df: pd.DataFrame | None = None):
    if df is None:
        df = load_clean()
    X, y = df[[TEXT_COL]], df['target']
    Xdev, Xtest, ydev, ytest = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    Xtrain, Xval, ytrain, yval = train_test_split(Xdev, ydev, test_size=.25, random_state=SEED,
                                                  stratify=ydev)
    return Xtrain, Xval, Xtest, ytrain, yval, ytest
