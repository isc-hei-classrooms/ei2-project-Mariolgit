"""
Feature Engineering v2 — Matrice X pour XGBoost
=================================================
Transforme dataset_clean.csv + sion_meteo_reelle_*.csv en matrice X/y
prête pour la prédiction de charge J+1.

18 features + 1 target (load) :
  - 6 calendaires  (heure locale Europe/Zurich, day-of-year, day-of-week, férié)
  - 5 prévisions météo J+1 (temp, irad, precip par heure + somme irad + clear-sky ratio)
  - 3 météo observée J-1 (temp, radiation, précip agrégés sur J-1)
  - 3 load historique (mean J-1, same-hour J-1, mean J-7)
  - 2 engineered (ratio load/irad, delta temp prédit vs observé)

Colonnes supplémentaires (benchmark, pas utilisées comme features) :
  - load_forecast : prévision Oiken (baseline de comparaison)

Anti-leakage :
  - J-1 = 2 jours avant J+1 (le load de J n'est PAS disponible à 11h J)
  - Toutes les features historiques utilisent des données strictement antérieures

Usage:
    python features/FEATURES2.py
    python features/FEATURES2.py --output data/features_v2.csv
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

# Split temporel (chronologique uniquement)
# Train: oct 2022 → sept 2024 (~2 ans, toutes saisons)
# Gap: 1 semaine (anti-leakage)
# Test: oct 2024 → sept 2025 (~1 an complet)
TRAIN_END = "2024-10-07"


# ─────────────────────────────────────────────
# Jours fériés du Valais (Suisse)
# ─────────────────────────────────────────────


def get_valais_holidays(years: list[int]) -> set[str]:
    """Retourne les jours fériés valaisans pour les années données.

    Inclut : fériés fédéraux + cantonaux (VS).
    Les dates de Pâques sont calculées avec l'algorithme de Gauss.
    Format retourné : set de strings 'YYYY-MM-DD'.
    """
    holidays: set[str] = set()

    for year in years:
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
        d_var = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d_var - g + 15) % 30
        i = c // 4
        k = c % 4
        ll = (32 + 2 * e + 2 * i - h - k) % 7
        mm = (a + 11 * h + 22 * ll) // 451
        month = (h + ll - 7 * mm + 114) // 31
        day = ((h + ll - 7 * mm + 114) % 31) + 1
        easter = pd.Timestamp(year, month, day)

        mobile = {
            "Vendredi Saint": easter - pd.Timedelta(days=2),
            "Lundi de Pâques": easter + pd.Timedelta(days=1),
            "Ascension": easter + pd.Timedelta(days=39),
            "Lundi de Pentecôte": easter + pd.Timedelta(days=50),
            "Fête-Dieu": easter + pd.Timedelta(days=60),
        }
        for ts in mobile.values():
            holidays.add(ts.strftime("%Y-%m-%d"))

    return holidays


# ─────────────────────────────────────────────
# 1. Features calendaires
# ─────────────────────────────────────────────


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute 6 features calendaires en heure locale (Europe/Zurich).

    - time_of_day_sin / _cos : encodage cyclique de l'heure locale (24h)
    - day_of_year_sin / _cos : encodage cyclique du jour de l'année (365.25j)
    - day_of_week            : 1=lundi … 7=dimanche (catégoriel)
    - is_holiday             : jour férié valaisan (0/1)
    """
    log.info("Features calendaires (heure locale Europe/Zurich)...")
    idx = pd.DatetimeIndex(df.index)
    ts_local = idx.tz_convert("Europe/Zurich")

    hour_frac = ts_local.hour + ts_local.minute / 60
    df["time_of_day_sin"] = np.sin(2 * np.pi * hour_frac / 24)
    df["time_of_day_cos"] = np.cos(2 * np.pi * hour_frac / 24)

    doy = ts_local.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * doy / 365.25)

    df["day_of_week"] = ts_local.dayofweek + 1  # 1=lundi, 7=dimanche

    years = sorted(ts_local.year.unique().tolist())
    holidays = get_valais_holidays(years)
    df["is_holiday"] = ts_local.strftime("%Y-%m-%d").isin(holidays).astype(int)

    log.info(f"  → {df['is_holiday'].sum()} heures marquées comme jours fériés")
    return df


# ─────────────────────────────────────────────
# 2. Prévisions météo J+1
# ─────────────────────────────────────────────


def add_forecast_features(df: pd.DataFrame, meteo_obs: pd.DataFrame) -> pd.DataFrame:
    """Ajoute 5 features issues des prévisions COSMO pour l'heure J+1.

    - temp_pred_h       : température prédite pour cette heure (pred_temperature)
    - irad_pred_h       : irradiance prédite pour cette heure (pred_radiation)
    - precip_risk_h     : précipitations prédites pour cette heure (pred_precipitation)
    - irad_pred_daily_sum : somme irradiance prédite sur toute la journée J+1
    - clear_sky_ratio   : irad_pred_daily_sum / p95 somme journalière radiation_obs (par mois)

    NaN attendus après ~19h locale J+1 (couverture 33h depuis 09 UTC).
    """
    log.info("Features prévisions météo J+1...")

    df["temp_pred_h"] = df["pred_temperature"]
    df["irad_pred_h"] = df["pred_radiation"]
    df["precip_risk_h"] = df["pred_precipitation"]

    # Somme irradiance prédite par jour J+1 (groupby sur la date UTC)
    idx = pd.DatetimeIndex(df.index)
    date_j1 = idx.normalize()
    df["irad_pred_daily_sum"] = df.groupby(date_j1)["pred_radiation"].transform("sum")

    # Référence clear-sky : p95 des sommes journalières observées, par mois
    daily_obs_sum = meteo_obs["radiation_obs"].resample("D").sum()
    obs_dti = pd.DatetimeIndex(daily_obs_sum.index)
    monthly_p95 = daily_obs_sum.groupby(obs_dti.month).quantile(0.95)
    month_series = pd.Series(idx.month, index=df.index).map(monthly_p95).clip(lower=1.0)
    df["clear_sky_ratio"] = df["irad_pred_daily_sum"] / month_series

    log.info(
        "  → Colonnes ajoutées: temp_pred_h, irad_pred_h, precip_risk_h, "
        "irad_pred_daily_sum, clear_sky_ratio"
    )
    return df


# ─────────────────────────────────────────────
# 3. Météo observée J-1
# ─────────────────────────────────────────────


def add_obs_weather_features(df: pd.DataFrame, meteo_obs: pd.DataFrame) -> pd.DataFrame:
    """Ajoute 3 agrégats journaliers de la météo observée sur J-1.

    J-1 = 2 jours avant J+1 en UTC.
    Ces features sont identiques pour toutes les heures d'un même J+1.

    - temp_obs_J1_mean       : température moyenne J-1
    - radiation_obs_J1_mean  : radiation moyenne J-1
    - precip_obs_J1_sum      : cumul précipitations J-1
    """
    log.info("Features météo observée J-1...")

    daily_meteo = meteo_obs.resample("D").agg(
        temp_obs_J1_mean=("temp_obs", "mean"),
        radiation_obs_J1_mean=("radiation_obs", "mean"),
        precip_obs_J1_sum=("precip_obs", "sum"),
    )
    # Retirer la timezone de l'index pour le mapping (daily_meteo index = date naive UTC)
    daily_meteo.index = pd.DatetimeIndex(daily_meteo.index).tz_localize(None)

    idx = pd.DatetimeIndex(df.index)
    date_jminus1 = (idx.normalize() - pd.Timedelta(days=2)).tz_localize(None)

    s_jm1 = pd.Series(date_jminus1, index=df.index)
    df["temp_obs_J1_mean"] = s_jm1.map(daily_meteo["temp_obs_J1_mean"])
    df["radiation_obs_J1_mean"] = s_jm1.map(daily_meteo["radiation_obs_J1_mean"])
    df["precip_obs_J1_sum"] = s_jm1.map(daily_meteo["precip_obs_J1_sum"])

    log.info("  → Colonnes ajoutées: temp_obs_J1_mean, radiation_obs_J1_mean, precip_obs_J1_sum")
    return df


# ─────────────────────────────────────────────
# 4. Load historique
# ─────────────────────────────────────────────


def add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute 3 features de charge historique (sans leakage).

    J-1 = 2 jours avant J+1 → shift(48) pour la même heure.

    - load_J1_mean       : charge moyenne journalière J-1
    - load_J1_same_hour  : charge à la même heure sur J-1 (shift 48h)
    - load_J7_mean       : charge moyenne journalière J-7 (même jour de semaine)
    """
    log.info("Features load historique...")

    # Même heure J-1 = 48 h avant
    df["load_J1_same_hour"] = df["load"].shift(48)

    # Moyennes journalières via resample
    daily_load = df["load"].resample("D").mean()
    daily_load.index = pd.DatetimeIndex(daily_load.index).tz_localize(None)

    idx = pd.DatetimeIndex(df.index)
    date_jminus1 = (idx.normalize() - pd.Timedelta(days=2)).tz_localize(None)
    date_jminus7 = (idx.normalize() - pd.Timedelta(days=8)).tz_localize(None)

    df["load_J1_mean"] = pd.Series(date_jminus1, index=df.index).map(daily_load)
    df["load_J7_mean"] = pd.Series(date_jminus7, index=df.index).map(daily_load)

    log.info("  → Colonnes ajoutées: load_J1_mean, load_J1_same_hour, load_J7_mean")
    return df


# ─────────────────────────────────────────────
# 5. Features engineered
# ─────────────────────────────────────────────


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute 2 features d'interaction.

    - ratio_load_irad_J1    : load_J1_mean / radiation_obs_J1_mean (clip lower=1 anti-div0)
    - delta_temp_pred_vs_obs : temp_pred_h - temp_obs_J1_mean (changement de régime météo)
    """
    log.info("Features engineered...")

    df["ratio_load_irad_J1"] = df["load_J1_mean"] / df["radiation_obs_J1_mean"].clip(lower=1.0)
    df["delta_temp_pred_vs_obs"] = df["temp_pred_h"] - df["temp_obs_J1_mean"]

    log.info("  → Colonnes ajoutées: ratio_load_irad_J1, delta_temp_pred_vs_obs")
    return df


# ─────────────────────────────────────────────
# Assemblage final
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Calendaire
    "time_of_day_sin",
    "time_of_day_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "day_of_week",
    "is_holiday",
    # Prévisions météo J+1
    "temp_pred_h",
    "irad_pred_h",
    "precip_risk_h",
    "irad_pred_daily_sum",
    "clear_sky_ratio",
    # Météo observée J-1
    "temp_obs_J1_mean",
    "radiation_obs_J1_mean",
    "precip_obs_J1_sum",
    # Load historique
    "load_J1_mean",
    "load_J1_same_hour",
    "load_J7_mean",
    # Engineered
    "ratio_load_irad_J1",
    "delta_temp_pred_vs_obs",
]

# Colonnes de benchmark (conservées dans le CSV mais PAS utilisées comme features)
BENCHMARK_COLS = ["load_forecast"]

TARGET_COL = "load"


def build_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sélectionne features + cible + benchmarks, drop NaN et split train/test."""
    log.info("Assemblage de la matrice X/y...")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes dans le DataFrame: {missing}")

    # Inclure les colonnes de benchmark si elles existent
    benchmark_present = [c for c in BENCHMARK_COLS if c in df.columns]
    if benchmark_present:
        log.info(f"  → Colonnes benchmark incluses: {benchmark_present}")
    else:
        log.info("  → Aucune colonne benchmark trouvée (load_forecast absent)")

    keep_cols = FEATURE_COLS + [TARGET_COL] + benchmark_present
    df_xy = df[keep_cols].copy()

    n_before = len(df_xy)
    df_xy = df_xy.dropna(subset=["load_J1_same_hour", "load_J1_mean", "load_J7_mean", TARGET_COL])
    log.info(
        f"  → {n_before - len(df_xy)} lignes supprimées (NaN load/lags), {len(df_xy)} restantes"
    )

    df_train = df_xy[df_xy.index < TRAIN_END]
    df_test = df_xy[df_xy.index >= TRAIN_END]

    log.info(
        f"  → Train: {len(df_train)} lignes ({df_train.index.min()} → {df_train.index.max()})"
    )
    log.info(f"  → Test:  {len(df_test)} lignes ({df_test.index.min()} → {df_test.index.max()})")

    return df_xy, df_train, df_test


# ─────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────


def run_feature_engineering(output_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("=" * 60)
    log.info("Feature Engineering v2")
    log.info("=" * 60)

    # ── Chargement dataset_clean ──
    clean_path = DATA_DIR / "dataset_clean.csv"
    log.info(f"Chargement: {clean_path}")
    df = pd.read_csv(clean_path, index_col=0, parse_dates=True)
    # S'assurer que l'index est UTC
    df_idx = pd.DatetimeIndex(df.index)
    if df_idx.tz is None:
        df.index = df_idx.tz_localize("UTC")
    df = df.sort_index()
    log.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")

    # ── Chargement météo observée ──
    meteo_files = sorted(DATA_DIR.glob("sion_meteo_reelle_*.csv"))
    if not meteo_files:
        raise FileNotFoundError("Aucun fichier sion_meteo_reelle_*.csv trouvé dans data/")
    meteo_path = meteo_files[-1]
    log.info(f"Chargement météo observée: {meteo_path.name}")
    meteo_obs = pd.read_csv(meteo_path, parse_dates=["timestamp"])
    meteo_obs = meteo_obs.set_index("timestamp").sort_index()
    meteo_idx = pd.DatetimeIndex(meteo_obs.index)
    if meteo_idx.tz is None:
        meteo_obs.index = meteo_idx.tz_localize("UTC")
    log.info(f"  → {len(meteo_obs)} lignes ({meteo_obs.index.min()} → {meteo_obs.index.max()})")

    # ── Construction des features ──
    df = add_calendar_features(df)
    df = add_forecast_features(df, meteo_obs)
    df = add_obs_weather_features(df, meteo_obs)
    df = add_load_features(df)
    df = add_engineered_features(df)

    # ── Assemblage & split ──
    df_full, df_train, df_test = build_dataset(df)

    # ── Résumé ──
    log.info("=" * 60)
    log.info(f"RÉSUMÉ — {len(FEATURE_COLS)} features + 1 cible ({TARGET_COL})")
    benchmark_present = [c for c in BENCHMARK_COLS if c in df_full.columns]
    if benchmark_present:
        log.info(f"  + colonnes benchmark: {benchmark_present}")
    log.info(f"Shape finale : {df_full.shape}")
    log.info("-" * 60)
    log.info("NaN par colonne (sur le dataset complet) :")
    nan_counts = df_full.isnull().sum()
    for col, n in nan_counts.items():
        pct = 100 * n / len(df_full)
        if n > 0:
            log.info(f"  {col:35s} {n:5d} NaN ({pct:.1f}%)")
    if nan_counts.sum() == 0:
        log.info("  Aucun NaN.")
    log.info("-" * 60)
    for i, col in enumerate(FEATURE_COLS, 1):
        vals = df_train[col].dropna()
        log.info(
            f"  {i:2d}. {col:35s} | min={vals.min():8.3f} | max={vals.max():8.3f} | mean={vals.mean():8.3f}"
        )
    log.info("-" * 60)
    t = df_train[TARGET_COL]
    log.info(
        f"  Cible: {TARGET_COL:35s} | min={t.min():8.3f} | max={t.max():8.3f} | mean={t.mean():8.3f}"
    )
    log.info("=" * 60)

    # ── Export ──
    if output_path is None:
        output_path = str(DATA_DIR / "features_v2.csv")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_csv(out)
    log.info(f"Dataset sauvegardé: {out}")

    return df_train, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature Engineering v2 — matrice X pour XGBoost")
    parser.add_argument(
        "--output", default=None, help="Chemin du CSV de sortie (défaut: data/features_v2.csv)"
    )
    args = parser.parse_args()

    df_train, df_test = run_feature_engineering(args.output)
