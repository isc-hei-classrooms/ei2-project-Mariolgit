"""
Feature engineering pour la prédiction de charge
=================================================
Transforme dataset_clean.csv en matrice X/y prête pour XGBoost.

Catégories de features :
1. Temporelles (heure, jour, mois, jours fériés)
2. Météo (prévisions COSMO best run + spread)
3. Lags de charge (t-24, t-48, t-168)
4. Rolling windows (moyenne glissante 24h, 168h)

Cible : load (charge standardisée)

Usage:
    python feature_engineering.py
    python feature_engineering.py --output data/features.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Split temporel
TEST_START = "2024-10-01"

# ─────────────────────────────────────────────
# Jours fériés du Valais (Suisse)
# ─────────────────────────────────────────────


def get_valais_holidays(years: list[int]) -> set[str]:
    """Retourne les jours fériés valaisans pour les années données.

    Inclut : fériés fédéraux + cantonaux (VS).
    Les dates de Pâques sont calculées avec l'algorithme de Gauss.
    Format retourné : set de strings 'YYYY-MM-DD'.
    """
    holidays = set()

    for year in years:
        # Fériés fixes
        fixed = [
            (1, 1),  # Nouvel An
            (1, 2),  # Saint-Berchtold
            (3, 19),  # Saint-Joseph
            (5, 1),  # Fête du travail
            (8, 1),  # Fête nationale
            (8, 15),  # Assomption
            (9, 25),  # Saint-Nicolas de Flüe (VS)
            (11, 1),  # Toussaint
            (12, 8),  # Immaculée Conception
            (12, 25),  # Noël
            (12, 26),  # Saint-Étienne
        ]
        for m, d in fixed:
            holidays.add(f"{year}-{m:02d}-{d:02d}")

        # Pâques (algorithme de Gauss)
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter = pd.Timestamp(year, month, day)

        # Fériés mobiles basés sur Pâques
        mobile = {
            "Vendredi Saint": easter - pd.Timedelta(days=2),
            "Lundi de Pâques": easter + pd.Timedelta(days=1),
            "Ascension": easter + pd.Timedelta(days=39),
            "Lundi de Pentecôte": easter + pd.Timedelta(days=50),
            "Fête-Dieu": easter + pd.Timedelta(days=60),
        }
        for name, date in mobile.items():
            holidays.add(date.strftime("%Y-%m-%d"))

    return holidays


# ─────────────────────────────────────────────
# Création des features
# ─────────────────────────────────────────────


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features temporelles.

    - hour_sin, hour_cos : encodage cyclique de l'heure (24h)
    - weekday : jour de la semaine (0=lundi, 6=dimanche)
    - is_weekend : booléen samedi/dimanche
    - month_sin, month_cos : encodage cyclique du mois (12 mois)
    - is_holiday : jour férié valaisan
    """
    log.info("Features temporelles...")
    idx = df.index

    # Heure — encodage cyclique
    hour = idx.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Jour de la semaine
    df["weekday"] = idx.weekday
    df["is_weekend"] = (idx.weekday >= 5).astype(int)

    # Mois — encodage cyclique
    month = idx.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Jours fériés valaisans
    years = sorted(idx.year.unique())
    holidays = get_valais_holidays(years)
    df["is_holiday"] = idx.strftime("%Y-%m-%d").isin(holidays).astype(int)

    n_holidays = df["is_holiday"].sum()
    log.info(f"  → {n_holidays} heures marquées comme jours fériés")
    log.info(
        "  → Colonnes ajoutées: hour_sin, hour_cos, weekday, is_weekend, month_sin, month_cos, is_holiday"
    )
    return df


def add_meteo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne les features météo retenues après analyse de corrélation.

    Features retenues (best run) :
    - pred_radiation (r=0.91 avec pv_total, r=-0.51 avec load)
    - pred_temperature (r=-0.77 avec load)
    - pred_humidity (r=0.32 avec load, r=-0.52 avec pv_total)
    - pred_wind_speed (r=-0.16 avec load, mais info complémentaire)

    Features retenues (spread = incertitude) :
    - pred_radiation_spread (r=0.43 avec pv_total)
    - pred_temperature_spread (info sur la fiabilité de la prév.)

    Exclues après analyse :
    - pred_sunshine (r=0.85 avec pred_radiation → redondant)
    - pred_precipitation (r≈0 avec load et pv_total)
    - pred_pressure (r≈0 avec load)
    - pred_wind_dir (peu informatif pour la charge)
    """
    log.info("Features météo...")

    meteo_features = [
        "pred_radiation",
        "pred_temperature",
        "pred_humidity",
        "pred_wind_speed",
        "pred_radiation_spread",
        "pred_temperature_spread",
    ]

    present = [f for f in meteo_features if f in df.columns]
    missing = [f for f in meteo_features if f not in df.columns]

    if missing:
        log.warning(f"  Features météo manquantes: {missing}")

    log.info(f"  → {len(present)} features météo retenues: {present}")
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les lags de charge (valeurs passées).

    Lags retenus :
    - load_t-24 : même heure hier (meilleure baseline naïve, MAE=0.258)
    - load_t-48 : même heure avant-hier
    - load_t-168 : même heure, même jour semaine précédente

    Ces lags ne causent PAS de data leakage car ils utilisent
    uniquement des valeurs passées connues au moment de la prédiction.
    Pour le Niveau 1 (prédiction J+1 à 11h), load_t-24 correspond
    à la charge d'hier qui est connue (données Oiken disponibles
    à ~2-3h du matin pour J-1).
    """
    log.info("Features de lags...")

    df["load_lag_24h"] = df["load"].shift(24)
    df["load_lag_48h"] = df["load"].shift(48)
    df["load_lag_168h"] = df["load"].shift(168)

    log.info("  → Colonnes ajoutées: load_lag_24h, load_lag_48h, load_lag_168h")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les moyennes glissantes sur la charge.

    - load_rolling_24h : moyenne des 24 dernières heures
      (capture le niveau moyen récent de consommation)
    - load_rolling_168h : moyenne de la dernière semaine
      (capture la tendance hebdomadaire)

    min_periods assure qu'on ne calcule pas de moyenne
    sur trop peu de valeurs en début de série.
    """
    log.info("Features rolling...")

    df["load_rolling_24h"] = (
        df["load"]
        .shift(1)  # exclure l'heure courante (anti-leakage)
        .rolling(window=24, min_periods=12)
        .mean()
    )
    df["load_rolling_168h"] = df["load"].shift(1).rolling(window=168, min_periods=84).mean()

    log.info("  → Colonnes ajoutées: load_rolling_24h, load_rolling_168h")
    return df


# ─────────────────────────────────────────────
# Assemblage de la matrice X/y
# ─────────────────────────────────────────────

# Liste des features dans l'ordre
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


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Construit la matrice X/y et split train/test.

    Retourne (df_full, df_train, df_test) avec les colonnes
    FEATURE_COLS + TARGET_COL, sans NaN.
    """
    log.info("Assemblage de la matrice X/y...")

    # Vérifier que toutes les features sont présentes
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.error(f"  Features manquantes dans le DataFrame: {missing}")
        raise ValueError(f"Features manquantes: {missing}")

    # Sélectionner features + cible
    cols = available + [TARGET_COL]
    df_xy = df[cols].copy()

    # Supprimer les lignes avec NaN (lags en début de série, etc.)
    n_before = len(df_xy)
    df_xy = df_xy.dropna()
    n_dropped = n_before - len(df_xy)
    log.info(f"  → {n_dropped} lignes supprimées (NaN dans lags/rolling), {len(df_xy)} restantes")

    # Split temporel
    df_train = df_xy[df_xy.index < TEST_START]
    df_test = df_xy[df_xy.index >= TEST_START]

    log.info(
        f"  → Train: {len(df_train)} lignes ({df_train.index.min()} → {df_train.index.max()})"
    )
    log.info(f"  → Test:  {len(df_test)} lignes ({df_test.index.min()} → {df_test.index.max()})")

    return df_xy, df_train, df_test


# ─────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────


def run_feature_engineering(output_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exécute le pipeline de feature engineering complet."""
    log.info("=" * 60)
    log.info("Feature engineering")
    log.info("=" * 60)

    # Charger dataset_clean
    path = DATA_DIR / "dataset_clean.csv"
    log.info(f"Chargement: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    log.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")

    # Créer les features
    df = add_temporal_features(df)
    df = add_meteo_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Assemblage et split
    df_full, df_train, df_test = build_dataset(df)

    # Résumé des features
    log.info("=" * 60)
    log.info(f"RÉSUMÉ — {len(FEATURE_COLS)} features + 1 cible ({TARGET_COL})")
    log.info("-" * 60)
    for i, col in enumerate(FEATURE_COLS, 1):
        if col in df_train.columns:
            vals = df_train[col]
            log.info(
                f"  {i:2d}. {col:30s} | min={vals.min():8.3f} | max={vals.max():8.3f} | mean={vals.mean():8.3f}"
            )
    log.info("-" * 60)
    target = df_train[TARGET_COL]
    log.info(
        f"  Cible: {TARGET_COL:30s} | min={target.min():8.3f} | max={target.max():8.3f} | mean={target.mean():8.3f}"
    )
    log.info("=" * 60)

    # Export
    if output_path is None:
        output_path = str(DATA_DIR / "features.csv")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(output)
    log.info(f"Dataset sauvegardé: {output}")

    return df_train, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feature engineering pour la prédiction de charge"
    )
    parser.add_argument("--output", default=None, help="Chemin du CSV de sortie")
    args = parser.parse_args()

    df_train, df_test = run_feature_engineering(args.output)
