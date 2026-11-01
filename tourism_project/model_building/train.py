"""
Model building with experiment tracking
---------------------------------------
1. loads the train / test splits produced by prep.py (Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv),
2. defines a Random Forest inside a preprocessing + model pipeline and a hyper-parameter grid,
3. tunes it with 5-fold cross-validated GridSearchCV,
4. logs EVERY tuned parameter combination (one nested MLflow run each) plus the best parameters
   and the train / test metrics to MLflow,
5. picks the decision threshold on out-of-fold training predictions (no test-set leakage),
6. saves the best model to tourism_project/deployment/ so the workflow can commit it to the repo.

MLflow: the workflow starts an MLflow server on http://localhost:5000. If no server is reachable
(e.g. when the script is run locally / in Colab) the script automatically falls back to a local
./mlruns folder, so it always runs.
"""
import os
import time
import urllib.request
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------- configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", "."))                      # where the split files are
DEPLOY_DIR = Path(__file__).resolve().parents[1] / "deployment"        # tourism_project/deployment
MODEL_PATH = DEPLOY_DIR / "best_tourism_model_v1.joblib"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "tourism-package-prediction"
RANDOM_STATE = 42
CV_FOLDS = 5
SCORING = "f1"            # purchases are only ~19 % of customers -> optimise F1 of the positive class


# ----------------------------------------------------------------------------- MLflow helpers
def server_is_up(uri: str, attempts: int = 3, wait: int = 2) -> bool:
    """True when an MLflow tracking server answers on <uri>/health."""
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(wait)
    return False


def configure_mlflow() -> str:
    uri = TRACKING_URI
    if uri.startswith("http") and not server_is_up(uri):
        print(f"MLflow server {uri} not reachable -> using local file store ./mlruns")
        uri = "file:./mlruns"
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
    except Exception as exc:                       # server up but misbehaving -> local fallback
        print(f"Could not use {uri} ({exc}); falling back to ./mlruns")
        uri = "file:./mlruns"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
    print("MLflow tracking URI:", uri)
    return uri


# ----------------------------------------------------------------------------- evaluation
def evaluate(model, X, y, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def main() -> None:
    # ---- 1. load data --------------------------------------------------------------------
    Xtrain = pd.read_csv(DATA_DIR / "Xtrain.csv")
    Xtest = pd.read_csv(DATA_DIR / "Xtest.csv")
    ytrain = pd.read_csv(DATA_DIR / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(DATA_DIR / "ytest.csv").squeeze("columns")
    print(f"Train {Xtrain.shape} | Test {Xtest.shape} | purchase rate {ytrain.mean():.3f}")

    numeric_features = Xtrain.select_dtypes(include="number").columns.tolist()
    categorical_features = [c for c in Xtrain.columns if c not in numeric_features]
    print("Numeric features    :", numeric_features)
    print("Categorical features:", categorical_features)

    # ---- 2. model + parameter grid -------------------------------------------------------
    # tree based model -> no scaling needed; impute so that the app never fails on a gap
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "model__n_estimators": [100, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2],
        "model__max_features": ["sqrt", 0.5],
        "model__class_weight": [None, "balanced_subsample"],
    }
    n_candidates = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"\nGrid search: {n_candidates} parameter combinations x {CV_FOLDS}-fold CV, scoring='{SCORING}'")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=SCORING, n_jobs=-1, refit=True)

    configure_mlflow()
    with mlflow.start_run(run_name="random_forest_tuning"):
        mlflow.set_tag("model_family", "RandomForestClassifier")
        mlflow.log_params({"cv_folds": CV_FOLDS, "scoring": SCORING,
                           "n_candidates": n_candidates, "random_state": RANDOM_STATE,
                           "sklearn_version": sklearn.__version__})

        # ---- 3. tune ---------------------------------------------------------------------
        t0 = time.time()
        grid.fit(Xtrain, ytrain)
        print(f"Tuning finished in {time.time() - t0:.0f}s")

        # ---- 4a. log every tuned parameter combination (nested run per candidate) -------
        results = pd.DataFrame(grid.cv_results_)
        for i, row in results.iterrows():
            with mlflow.start_run(run_name=f"candidate_{i:03d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in row["params"].items()})
                mlflow.log_metrics({f"cv_mean_{SCORING}": float(row["mean_test_score"]),
                                    f"cv_std_{SCORING}": float(row["std_test_score"]),
                                    "cv_rank": int(row["rank_test_score"])})
        print(f"Logged {len(results)} tuning runs to MLflow")

        best_model = grid.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in grid.best_params_.items()}
        print("\nBest parameters:", best_params)
        print(f"Best CV {SCORING}: {grid.best_score_:.4f}")

        # ---- 5. decision threshold from out-of-fold TRAIN predictions --------------------
        oof = cross_val_predict(best_model, Xtrain, ytrain, cv=cv, method="predict_proba")[:, 1]
        grid_thr = np.round(np.arange(0.20, 0.71, 0.05), 2)
        threshold = float(grid_thr[int(np.argmax([f1_score(ytrain, oof >= t) for t in grid_thr]))])
        print(f"Selected classification threshold: {threshold:.2f}")

        # ---- 6. evaluate -----------------------------------------------------------------
        train_metrics = evaluate(best_model, Xtrain, ytrain, threshold)
        test_metrics = evaluate(best_model, Xtest, ytest, threshold)
        test_pred = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)
        train_pred = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
        print("\n=== TRAIN classification report ===")
        print(classification_report(ytrain, train_pred, target_names=["No purchase", "Purchase"]))
        print("=== TEST classification report ===")
        print(classification_report(ytest, test_pred, target_names=["No purchase", "Purchase"]))
        print("Metrics summary:")
        print(pd.DataFrame({"train": train_metrics, "test": test_metrics}).round(4).to_string())

        # ---- 4b. log best params + metrics ----------------------------------------------
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_param("classification_threshold", threshold)
        mlflow.log_metric(f"best_cv_{SCORING}", float(grid.best_score_))
        mlflow.log_metrics({f"train_{k}": float(v) for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in test_metrics.items()})

        # ---- 7. save the best model for deployment --------------------------------------
        DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": best_model,                       # preprocessing + Random Forest
            "threshold": threshold,                       # probability cut-off used by the app
            "feature_names": Xtrain.columns.tolist(),     # exact input columns / order
            "best_params": best_params,
            "metrics": {"train": {k: float(v) for k, v in train_metrics.items()},
                        "test": {k: float(v) for k, v in test_metrics.items()}},
            "sklearn_version": sklearn.__version__,
        }
        joblib.dump(bundle, MODEL_PATH, compress=3)
        print(f"\nBest model saved -> {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")
        try:
            mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        except Exception as exc:                          # never fail the pipeline because of artifact upload
            print("Could not upload model artifact to MLflow:", exc)


if __name__ == "__main__":
    main()
