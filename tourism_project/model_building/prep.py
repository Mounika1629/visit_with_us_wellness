"""
Data preparation
----------------
1. loads tourism_project/data/tourism.csv from the repository,
2. cleans the data and drops unnecessary columns,
3. splits it into train / test sets (stratified on the target),
4. saves Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv in the current working directory.

The GitHub Actions workflow uploads these four files as an artifact and the training job
downloads them - nothing is pushed to any external store.
"""
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
OUTPUT_DIR = Path(os.environ.get("PREP_OUTPUT_DIR", "."))   # repo root when run from the workflow
TARGET = "ProdTaken"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH}  shape={df.shape}")

    # ---- drop unnecessary columns --------------------------------------------------------
    # 'Unnamed: 0' is a leftover row index, 'CustomerID' is only an identifier -> no predictive value
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")

    # ---- cleaning ------------------------------------------------------------------------
    # trim stray whitespace in text columns
    text_cols = [c for c in df.columns if c not in df.select_dtypes(include="number").columns]
    for col in text_cols:
        df[col] = df[col].str.strip()
    # 'Fe Male' is a typo for 'Female' (155 rows)
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    # a row without a target label cannot be used for supervised learning
    before = len(df)
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)
    print(f"Rows without target removed: {before - len(df)}")
    print("Gender values after cleaning:", sorted(df["Gender"].unique()))
    print(f"Remaining missing values: {int(df.isna().sum().sum())} (numeric gaps are imputed inside the model pipeline)")

    # ---- split ---------------------------------------------------------------------------
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Xtrain.to_csv(OUTPUT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(OUTPUT_DIR / "Xtest.csv", index=False)
    ytrain.to_csv(OUTPUT_DIR / "ytrain.csv", index=False)
    ytest.to_csv(OUTPUT_DIR / "ytest.csv", index=False)

    print(f"\nXtrain {Xtrain.shape} | Xtest {Xtest.shape}")
    print(f"Purchase rate  train={ytrain.mean():.3f}  test={ytest.mean():.3f}")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv ->", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
