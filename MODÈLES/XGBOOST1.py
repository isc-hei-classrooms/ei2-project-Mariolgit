"""
Entraînement XGBoost pour la prédiction de charge
===================================================
Charge features.csv, entraîne un modèle XGBoost, évalue sur le test set,
et compare aux baselines.

Usage:
    python train_xgboost.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

# Split temporel (doit être identique à feature_engineering.py)
TEST_START = "2024-10-01"

# Features (même liste que feature_engineering.py)
FEATURE_COLS = [
    # Temporelles
    "hour_sin",
    "hour_cos",
    "weekday",
    "is_weekend",
    "month_sin",
    "month_cos",
    "is_holiday",
    # Météo (prévisions)
    "pred_radiation",
    "pred_temperature",
    "pred_humidity",
    "pred_wind_speed",
    "pred_radiation_spread",
    "pred_temperature_spread",
    # Lags
    "load_lag_24h",
    "load_lag_48h",
    "load_lag_168h",
    # Rolling
    "load_rolling_24h",
    "load_rolling_168h",
]

TARGET_COL = "load"

# Hyperparamètres XGBoost (point de départ raisonnable)
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,  # L1
    "reg_lambda": 1.0,  # L2
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 30,
}


# ─────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge features.csv et split train/test."""
    path = DATA_DIR / "features.csv"
    log.info(f"Chargement: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    log.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")

    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]
    log.info(f"  → Train: {len(train)} | Test: {len(test)}")

    return train, test


# ─────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule MAE, RMSE et nRMSE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    amplitude = y_true.max() - y_true.min()
    nrmse = rmse / amplitude * 100

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "nrmse_pct": round(nrmse, 2),
    }


# ─────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────


def train_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[xgb.XGBRegressor, dict]:
    """Entraîne un XGBoost avec early stopping sur le test set."""
    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test = test[FEATURE_COLS]
    y_test = test[TARGET_COL]

    log.info(f"Entraînement XGBoost ({len(FEATURE_COLS)} features)...")
    log.info(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")

    model = xgb.XGBRegressor(**XGB_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )

    # Prédictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Métriques
    metrics_train = compute_metrics(y_train.values, y_pred_train)
    metrics_test = compute_metrics(y_test.values, y_pred_test)

    log.info(
        f"  Train — MAE={metrics_train['mae']:.4f} | RMSE={metrics_train['rmse']:.4f} | nRMSE={metrics_train['nrmse_pct']:.2f}%"
    )
    log.info(
        f"  Test  — MAE={metrics_test['mae']:.4f} | RMSE={metrics_test['rmse']:.4f} | nRMSE={metrics_test['nrmse_pct']:.2f}%"
    )

    best_iteration = (
        model.best_iteration if hasattr(model, "best_iteration") else XGB_PARAMS["n_estimators"]
    )
    log.info(f"  Best iteration: {best_iteration}")

    results = {
        "train": metrics_train,
        "test": metrics_test,
        "best_iteration": best_iteration,
        "params": {k: v for k, v in XGB_PARAMS.items() if k != "n_jobs"},
    }

    return model, results


# ─────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────


def analyse_feature_importance(model: xgb.XGBRegressor) -> pd.DataFrame:
    """Extrait et affiche le classement des features par importance."""
    importance = model.feature_importances_
    fi = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    fi["rank"] = range(1, len(fi) + 1)
    fi["pct"] = (fi["importance"] / fi["importance"].sum() * 100).round(1)

    log.info("Feature importance (top 10):")
    for _, row in fi.head(10).iterrows():
        bar = "█" * int(row["pct"])
        log.info(f"  {row['rank']:2.0f}. {row['feature']:30s} {row['pct']:5.1f}% {bar}")

    return fi


# ─────────────────────────────────────────────
# Comparaison aux baselines
# ─────────────────────────────────────────────


def compare_baselines(test: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Compare le modèle aux baselines sur le test set."""
    y_true = test[TARGET_COL].values

    comparisons = []

    # XGBoost
    m = compute_metrics(y_true, y_pred)
    comparisons.append({"model": "XGBoost", **m})

    # Baseline: Persistence 24h
    if "load_lag_24h" in test.columns:
        m = compute_metrics(y_true, test["load_lag_24h"].values)
        comparisons.append({"model": "Persistence 24h", **m})

    # Baseline: Persistence 168h
    if "load_lag_168h" in test.columns:
        m = compute_metrics(y_true, test["load_lag_168h"].values)
        comparisons.append({"model": "Persistence 168h", **m})

    # Baseline: Prévision Oiken (charger depuis dataset_clean)
    try:
        df_clean = pd.read_csv(DATA_DIR / "dataset_clean.csv", index_col=0, parse_dates=True)
        oiken = df_clean["load_forecast"].reindex(test.index).dropna()
        common = test.index.intersection(oiken.index)
        if len(common) > 100:
            m = compute_metrics(
                test.loc[common, TARGET_COL].values,
                oiken.loc[common].values,
            )
            comparisons.append({"model": "Prévision Oiken", **m})
    except Exception as e:
        log.warning(f"  Impossible de charger la prévision Oiken: {e}")

    comp_df = pd.DataFrame(comparisons)

    log.info("=" * 65)
    log.info("COMPARAISON AUX BASELINES (test set)")
    log.info("=" * 65)
    for _, row in comp_df.iterrows():
        marker = "→" if row["model"] == "XGBoost" else " "
        log.info(
            f"  {marker} {row['model']:25s} | MAE={row['mae']:.4f} | "
            f"RMSE={row['rmse']:.4f} | nRMSE={row['nrmse_pct']:.2f}%"
        )
    log.info("=" * 65)

    # Amélioration vs Oiken
    xgb_rmse = comp_df.loc[comp_df["model"] == "XGBoost", "rmse"].values[0]
    oiken_row = comp_df.loc[comp_df["model"] == "Prévision Oiken"]
    if len(oiken_row) > 0:
        oiken_rmse = oiken_row["rmse"].values[0]
        improvement = (1 - xgb_rmse / oiken_rmse) * 100
        if improvement > 0:
            log.info(f"  XGBoost bat Oiken de {improvement:.1f}% (RMSE)")
        else:
            log.info(f"  XGBoost est {-improvement:.1f}% moins bon qu'Oiken (RMSE)")

    return comp_df


# ─────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────


def run_training():
    """Exécute le pipeline complet : chargement → entraînement → évaluation."""
    log.info("=" * 60)
    log.info("Entraînement XGBoost — Prédiction de charge Oiken")
    log.info("=" * 60)

    # Charger les features
    train, test = load_features()

    # Entraîner le modèle
    model, results = train_model(train, test)

    # Feature importance
    fi = analyse_feature_importance(model)

    # Prédictions sur le test set
    y_pred = model.predict(test[FEATURE_COLS])

    # Comparaison aux baselines
    comp = compare_baselines(test, y_pred)

    # Sauvegarder
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Modèle
    model_path = MODEL_DIR / "xgboost_load.json"
    model.save_model(str(model_path))
    log.info(f"Modèle sauvegardé: {model_path}")

    # Feature importance
    fi_path = MODEL_DIR / "feature_importance.csv"
    fi.to_csv(fi_path, index=False)

    # Résultats
    results_path = MODEL_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Comparaison baselines
    comp_path = MODEL_DIR / "baseline_comparison.csv"
    comp.to_csv(comp_path, index=False)

    # Prédictions test (pour visualisation)
    preds_df = pd.DataFrame(
        {
            "timestamp": test.index,
            "y_true": test[TARGET_COL].values,
            "y_pred": y_pred,
        }
    )
    preds_path = MODEL_DIR / "predictions_test.csv"
    preds_df.to_csv(preds_path, index=False)
    log.info(f"Prédictions test sauvegardées: {preds_path}")

    log.info("=" * 60)
    log.info("Terminé.")
    log.info("=" * 60)

    return model, results, fi, comp


if __name__ == "__main__":
    run_training()
