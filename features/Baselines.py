"""Baselines pour la prédiction de charge.

4 baselines simples pour établir un point de comparaison avant le modèle XGBoost.

Usage:
    python baselines.py
"""

import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Split temporel : dernière année = test
TEST_START = "2024-10-01"


# ─────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Charge dataset_clean.csv et prépare l'index."""
    path = DATA_DIR / "dataset_clean.csv"
    log.info(f"Chargement: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Ne garder que les lignes où load est disponible
    df = df.dropna(subset=["load"])
    log.info(f"  → {len(df)} lignes avec load non-NaN")
    log.info(f"  → Période: {df.index.min()} → {df.index.max()}")
    return df


# ─────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────


def baseline_persistence_24h(df: pd.DataFrame) -> pd.Series:
    """Baseline 1 — Persistence naïve (t-24h).

    Prédit que la charge sera identique à celle de la même heure hier.
    C'est le minimum absolu à battre.
    """
    return df["load"].shift(24)


def baseline_persistence_168h(df: pd.DataFrame) -> pd.Series:
    """Baseline 2 — Persistence hebdomadaire (t-168h).

    Prédit que la charge sera identique à celle de la même heure,
    le même jour de la semaine, la semaine précédente.
    Capte le pattern hebdomadaire (lundi ≠ dimanche).
    """
    return df["load"].shift(168)


def baseline_rolling_4weeks(df: pd.DataFrame) -> pd.Series:
    """Baseline 3 — Moyenne glissante 4 semaines (même heure, même jour).

    Pour chaque heure, prend la moyenne des 4 dernières occurrences
    du même jour de semaine à la même heure.
    Plus robuste que la persistence simple car elle lisse le bruit.
    """
    df_tmp = df[["load"]].copy()
    dt_idx = cast(pd.DatetimeIndex, df_tmp.index)
    df_tmp["weekday"] = dt_idx.day_of_week
    df_tmp["hour"] = dt_idx.hour

    predictions = pd.Series(np.nan, index=df.index, name="pred_rolling_4w")

    for idx in cast(pd.DatetimeIndex, df_tmp.index):
        wd = idx.day_of_week
        hr = idx.hour
        # Chercher les 4 mêmes (jour, heure) précédents
        mask = (df_tmp.index < idx) & (df_tmp["weekday"] == wd) & (df_tmp["hour"] == hr)
        past = pd.Series(df_tmp.loc[mask, "load"]).tail(4)
        if len(past) >= 1:
            predictions.at[idx] = past.mean()

    return predictions


def baseline_rolling_4weeks_fast(df: pd.DataFrame) -> pd.Series:
    """Baseline 3 (version rapide) — Moyenne des 4 dernières semaines.

    Version vectorisée : moyenne de shift(168), shift(336), shift(504), shift(672).
    Chaque shift correspond à 1, 2, 3 et 4 semaines en arrière.
    """
    shifts = pd.concat(
        [df["load"].shift(168 * w) for w in range(1, 5)],
        axis=1,
    )
    return shifts.mean(axis=1)


def baseline_oiken_forecast(df: pd.DataFrame) -> pd.Series:
    """Baseline 4 — Prévision existante d'Oiken (load_forecast).

    C'est la baseline business : la prévision que fait déjà Oiken.
    Le modèle XGBoost doit faire mieux que ça pour justifier son existence.
    """
    return df["load_forecast"]


# ─────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────


def evaluate(y_true: pd.Series, y_pred: pd.Series, name: str) -> dict:
    """Calcule les métriques d'évaluation sur les valeurs non-NaN communes."""
    mask = y_true.notna() & y_pred.notna()
    y_t = y_true[mask]
    y_p = y_pred[mask]

    n = len(y_t)
    if n == 0:
        log.warning(f"  {name}: aucune valeur commune pour évaluer")
        return {"name": name, "n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan}

    errors = y_t - y_p
    mae = errors.abs().mean()
    rmse = np.sqrt((errors**2).mean())

    # MAPE : attention aux valeurs proches de 0 (charge standardisée)
    # On utilise le MAPE relatif à l'amplitude du signal
    amplitude = y_t.max() - y_t.min()
    nrmse = rmse / amplitude * 100  # Normalized RMSE en %

    return {
        "name": name,
        "n": n,
        "mae": mae,
        "rmse": rmse,
        "nrmse_pct": nrmse,
    }


def run_baselines():
    """Exécute et compare les 4 baselines."""
    df = load_data()

    # Split temporel
    train = df[df.index < TEST_START]
    test = df[df.index >= TEST_START]
    log.info(f"Train: {len(train)} lignes ({train.index.min()} → {train.index.max()})")
    log.info(f"Test:  {len(test)} lignes ({test.index.min()} → {test.index.max()})")

    # Calculer les baselines sur tout le dataset (le shift gère naturellement le passé)
    log.info("Calcul des baselines...")
    baselines = {
        "1. Persistence 24h": baseline_persistence_24h(df),
        "2. Persistence 168h (hebdo)": baseline_persistence_168h(df),
        "3. Moyenne 4 semaines": baseline_rolling_4weeks_fast(df),
        "4. Prévision Oiken": baseline_oiken_forecast(df),
    }

    # Évaluer sur le test set uniquement
    log.info("=" * 65)
    log.info("RÉSULTATS SUR LE TEST SET")
    log.info(f"Période: {test.index.min()} → {test.index.max()}")
    log.info("=" * 65)

    results = []
    y_true = test["load"]

    for name, preds in baselines.items():
        # Aligner les prédictions sur le test set
        preds_test = preds.reindex(test.index)
        res = evaluate(y_true, preds_test, name)
        results.append(res)
        log.info(
            f"  {name:35s} | MAE={res['mae']:.4f} | RMSE={res['rmse']:.4f} | "
            f"nRMSE={res['nrmse_pct']:.2f}% | n={res['n']}"
        )

    log.info("=" * 65)

    # Identifier la meilleure baseline
    valid = [r for r in results if not np.isnan(r["rmse"])]
    if valid:
        best = min(valid, key=lambda r: r["rmse"])
        log.info(f"Meilleure baseline: {best['name']} (RMSE={best['rmse']:.4f})")
        log.info("Le modèle XGBoost devra battre ce score.")

    # Sauvegarder les résultats
    results_df = pd.DataFrame(results)
    output_path = DATA_DIR / "baseline_results.csv"
    results_df.to_csv(output_path, index=False)
    log.info(f"Résultats sauvegardés: {output_path}")

    return results_df


if __name__ == "__main__":
    run_baselines()
