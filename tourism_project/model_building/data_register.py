"""
Data registration
-----------------
Registers the raw tourism dataset that lives inside the repository
(tourism_project/data/tourism.csv). The script
  1. loads the CSV straight from the repo,
  2. checks that every expected column is present (and that the target is valid),
  3. prints a short summary so the dataset version used by a pipeline run is visible in the logs.

Exit code is non-zero when validation fails, so the GitHub Actions job stops early.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

# <repo>/tourism_project/data/tourism.csv  (independent of the folder the script is run from)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]
# Column that pandas.to_csv() adds when the index is written - tolerated, removed later in prep.py
TOLERATED_EXTRA_COLUMNS = ["Unnamed: 0"]


def main() -> int:
    print(f"Dataset path : {DATA_PATH}")
    if not DATA_PATH.exists():
        print("ERROR: dataset file not found. Add tourism.csv to tourism_project/data/.")
        return 1

    df = pd.read_csv(DATA_PATH)
    errors = []

    # ---- 1. expected columns -------------------------------------------------------------
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS + TOLERATED_EXTRA_COLUMNS]
    if missing:
        errors.append(f"Missing expected columns: {missing}")
    if extra:
        print(f"WARNING: unexpected extra columns (kept as-is): {extra}")

    # ---- 2. basic content checks ---------------------------------------------------------
    if df.empty:
        errors.append("Dataset is empty.")
    if "ProdTaken" in df.columns and not set(df["ProdTaken"].dropna().unique()) <= {0, 1}:
        errors.append("Target column 'ProdTaken' must only contain 0/1.")
    if "CustomerID" in df.columns and not df["CustomerID"].is_unique:
        errors.append("CustomerID is not unique.")

    # ---- 3. summary ----------------------------------------------------------------------
    print("\n===== DATASET SUMMARY =====")
    print(f"Rows x Columns : {df.shape[0]} x {df.shape[1]}")
    print(f"File size      : {DATA_PATH.stat().st_size / 1024:.1f} KB")
    print(f"SHA-256        : {hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()[:16]}...")
    print("\nColumn types:")
    print(df.dtypes.astype(str).to_string())
    print("\nMissing values per column (only columns with gaps):")
    na = df.isna().sum()
    print(na[na > 0].to_string() if (na > 0).any() else "none")
    if "ProdTaken" in df.columns:
        print("\nTarget distribution (ProdTaken):")
        print(pd.concat([df["ProdTaken"].value_counts().rename("count"),
                         df["ProdTaken"].value_counts(normalize=True).round(3).rename("share")], axis=1).to_string())
    print(f"\nDuplicate rows (all columns): {int(df.duplicated().sum())}")

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        return 1
    print("\nVALIDATION PASSED: all expected columns are present - dataset registered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
