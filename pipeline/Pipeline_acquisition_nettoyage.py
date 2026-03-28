"""Pipeline d'acquisition et de nettoyage des données (v2 — sans météo réelle).

Projet : Prédiction de la courbe de charge Oiken
Sources : données Oiken (charge + PV) + prévisions météo COSMO/ICON

⚠️  Les données météo réelles (observations) ont été retirées pour éviter
    le data leakage : en production, seules les prévisions sont disponibles
    au moment de la prédiction.

Usage:
    python pipeline.py --oiken data/oiken.csv \
                       --forecast data/prevision_data.csv \
                       --output data/dataset_clean.csv
"""

import argparse
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Configuration & logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Valeur sentinelle indiquant donnée manquante dans les prévisions
SENTINEL = -99999.0

# Résolution temporelle cible (doit correspondre au pas des prévisions COSMO)
TARGET_FREQ = "1h"

# Heures de nuit (UTC) où la production solaire doit être nulle
# En Suisse : lever ~4h-6h UTC en été, coucher ~18h-20h UTC
# On prend des bornes larges pour ne pas couper de vrais signaux
NIGHT_START_UTC = 21  # 21h UTC = nuit certaine toute l'année
NIGHT_END_UTC = 5  # 5h UTC  = nuit certaine toute l'année

# Seuil z-score pour détection des valeurs aberrantes
ZSCORE_THRESHOLD = 4.0

# Saut maximum entre deux pas de temps consécutifs (pour les prévisions)
MAX_DELTA = {
    "temperature": 8.0,  # °C par heure
    "pressure": 5.0,  # hPa par heure
}


# ═════════════════════════════════════════════
# FONCTIONS UTILITAIRES DE NETTOYAGE
# ═════════════════════════════════════════════


def is_night(index: pd.DatetimeIndex) -> pd.Series:
    """Retourne True si l'heure UTC est en pleine nuit.

    Nuit = [NIGHT_START_UTC, 23] ∪ [0, NIGHT_END_UTC]
    """
    hours = index.to_series().dt.hour
    return (hours >= NIGHT_START_UTC) | (hours <= NIGHT_END_UTC)


def clip_negatives(df: pd.DataFrame, columns: list, name: str = "") -> pd.DataFrame:
    """Force les valeurs négatives à 0 pour les colonnes spécifiées.

    Applicable à : production PV, radiation, précipitations, ensoleillement.
    """
    for col in columns:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                log.warning(f"  {name}{col}: {n_neg} valeurs négatives → 0")
                df[col] = df[col].clip(lower=0)
    return df


def enforce_night_zero(df: pd.DataFrame, columns: list, name: str = "") -> pd.DataFrame:
    """Force à 0 les colonnes solaires pendant la nuit."""
    night_mask = is_night(cast(pd.DatetimeIndex, df.index))
    for col in columns:
        if col in df.columns:
            night_nonzero = (df.loc[night_mask, col] > 0).sum()
            if night_nonzero > 0:
                log.info(f"  {name}{col}: {night_nonzero} valeurs nocturnes → 0")
                df.loc[night_mask, col] = 0.0
    return df


def apply_physical_bounds(df: pd.DataFrame, bounds: dict, name: str = "") -> pd.DataFrame:
    """Remplace par NaN les valeurs hors bornes physiques réalistes."""
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_out = mask.sum()
            if n_out > 0:
                log.warning(f"  {name}{col}: {n_out} outliers hors [{lo}, {hi}] → NaN")
                df.loc[mask, col] = np.nan
    return df


def detect_zscore_outliers(
    df: pd.DataFrame,
    columns: list,
    threshold: float = ZSCORE_THRESHOLD,
    window: int = 168,
    min_periods: int = 24,
    name: str = "",
) -> pd.DataFrame:
    """Détecte et remplace les outliers statistiques (z-score glissant) par NaN.

    Fenêtre glissante centrée pour capturer la saisonnalité locale.
    """
    for col in columns:
        if col not in df.columns:
            continue
        rolling_mean = df[col].rolling(window=window, center=True, min_periods=min_periods).mean()
        rolling_std = df[col].rolling(window=window, center=True, min_periods=min_periods).std()
        rolling_std = rolling_std.replace(0, np.nan)
        z_scores = (df[col] - rolling_mean) / rolling_std
        mask = z_scores.abs() > threshold
        n_out = mask.sum()
        if n_out > 0:
            log.warning(f"  {name}{col}: {n_out} outliers statistiques (|z| > {threshold}) → NaN")
            df.loc[mask, col] = np.nan
    return df


def detect_spikes(df: pd.DataFrame, max_deltas: dict, name: str = "") -> pd.DataFrame:
    """Détecte les variations trop brutales entre pas de temps consécutifs."""
    for col, max_delta in max_deltas.items():
        if col not in df.columns:
            continue
        delta = df[col].diff().abs()
        mask = delta > max_delta
        n_spikes = mask.sum()
        if n_spikes > 0:
            log.warning(f"  {name}{col}: {n_spikes} sauts brutaux (Δ > {max_delta}) → NaN")
            df.loc[mask, col] = np.nan
    return df


def check_cross_consistency(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Vérifie la cohérence entre variables liées dans les prévisions.

    Règle : si radiation == 0, alors sunshine devrait être ~0 (tolérance 1 min/sec).
    """
    # Cohérence sur les best runs
    pairs = [
        ("GLOB_best", "DURSUN_best"),
        ("pred_radiation", "pred_sunshine"),
    ]
    for rad_col, sun_col in pairs:
        if rad_col in df.columns and sun_col in df.columns:
            mask = (df[rad_col] == 0) & (df[sun_col] > 1)
            n = mask.sum()
            if n > 0:
                log.info(f"  {name}Incohérence {rad_col}=0 mais {sun_col}>1: {n} → 0")
                df.loc[mask, sun_col] = 0
    return df


def interpolate_gaps(df: pd.DataFrame, max_gap: int = 6, name: str = "") -> pd.DataFrame:
    """Interpole les NaN avec une limite de trou maximum.

    - Trous ≤ max_gap : interpolation linéaire temporelle
    - Trous > max_gap : laissés en NaN (on ne veut pas inventer de données)
    """
    n_before = df.isna().sum().sum()
    if n_before > 0:
        df = df.interpolate(method="time", limit=max_gap)
        # Forward/backward fill pour les bords uniquement (max 2 pas)
        df = df.ffill(limit=2).bfill(limit=2)
        n_after = df.isna().sum().sum()
        n_filled = n_before - n_after
        log.info(
            f"  {name}Interpolation: {n_filled}/{n_before} NaN comblés "
            f"(max_gap={max_gap}), {n_after} NaN restants"
        )
    return df


def detect_temporal_gaps(df: pd.DataFrame, expected_freq: str, name: str = "") -> None:
    """Journalise les trous dans la série temporelle."""
    expected = pd.tseries.frequencies.to_offset(expected_freq)
    gaps = df.index.to_series().diff()
    big_gaps = gaps[gaps > expected]
    if len(big_gaps) > 0:
        log.warning(f"  {name}{len(big_gaps)} trous temporels détectés:")
        for idx, gap in big_gaps.head(10).items():
            prev = idx - gap
            log.warning(f"    {prev} → {idx} (trou de {gap})")


# ═════════════════════════════════════════════
# ÉTAPE 1 — Chargement des données brutes
# ═════════════════════════════════════════════


def load_oiken(path: str) -> pd.DataFrame:
    """Charge les données Oiken (courbe de charge + production PV).

    Format attendu : CSV avec colonnes timestamp + valeurs numériques.
    Pas temporel d'origine : 15 minutes.
    """
    log.info(f"Chargement Oiken: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    rename_map = {
        "standardised load [-]": "load",
        "standardised forecast load [-]": "load_forecast",
        "central valais solar production [kWh]": "pv_central_valais",
        "sion area solar production [kWh]": "pv_sion",
        "sierre area production [kWh]": "pv_sierre",
        "remote solar production [kWh]": "pv_remote",
    }
    df = df.rename(columns=rename_map)
    df = df.set_index("timestamp")

    log.info(f"  → {len(df)} lignes, période: {df.index.min()} → {df.index.max()}")
    return df


def load_forecast(path: str) -> pd.DataFrame:
    """Charge les prévisions météo COSMO/ICON.

    Format attendu : CSV séparateur ';', décimale ',', index 'time_utc'.
    Colonnes : {VAR}_ctrl_lt{1..33} pour chaque variable et lead time.
    """
    log.info(f"Chargement prévisions météo: {path}")
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        parse_dates=["time_utc"],
        dayfirst=True,
    )
    df = df.set_index("time_utc")
    df.index.name = "timestamp"

    log.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")
    log.info(f"  → période: {df.index.min()} → {df.index.max()}")
    return df


# ═════════════════════════════════════════════
# ÉTAPE 2 — Nettoyage de chaque source
# ═════════════════════════════════════════════


def clean_oiken(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données Oiken.

    Traitements appliqués :
    1. Localisation UTC et suppression doublons (changements d'heure été/hiver)
    2. Détection des trous temporels
    3. Forcer valeurs PV négatives à 0
    4. Forcer production PV à 0 la nuit
    5. Détection des outliers statistiques sur la charge (z-score glissant)
    6. Bornes physiques sur la production PV
    7. Interpolation NaN (max 6 pas de temps = 1h30)
    8. Resampling 15min → 1h (charge: moyenne, PV: somme)
    9. Création de la colonne PV total
    """
    log.info("Nettoyage Oiken...")
    n_before = len(df)

    # 1. Timezone et doublons
    _idx = cast(pd.DatetimeIndex, df.index)
    if _idx.tz is None:
        df.index = _idx.tz_localize("UTC")
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        log.warning(f"  {n_dup} doublons de timestamp supprimés")
        df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # 2. Détection des trous temporels
    detect_temporal_gaps(df, "15min", name="[Oiken] ")

    # 3. Forcer les valeurs PV négatives à 0
    pv_cols = [c for c in df.columns if c.startswith("pv_")]
    df = clip_negatives(df, pv_cols, name="[Oiken] ")

    # 4. Forcer la production PV à 0 la nuit
    df = enforce_night_zero(df, pv_cols, name="[Oiken] ")

    # 5. Outliers statistiques sur la charge (z-score glissant, fenêtre 1 semaine)
    load_cols = [c for c in ["load", "load_forecast"] if c in df.columns]
    window_15min = 672  # 7 jours × 24h × 4 (pas de 15min)
    for col in load_cols:
        rolling_mean = df[col].rolling(window=window_15min, center=True, min_periods=96).mean()
        rolling_std = df[col].rolling(window=window_15min, center=True, min_periods=96).std()
        rolling_std = rolling_std.replace(0, np.nan)
        z = (df[col] - rolling_mean) / rolling_std
        mask = z.abs() > ZSCORE_THRESHOLD
        n_out = mask.sum()
        if n_out > 0:
            log.warning(f"  [Oiken] {col}: {n_out} outliers (|z| > {ZSCORE_THRESHOLD}) → NaN")
            df.loc[mask, col] = np.nan

    # 6. Bornes physiques PV (pas de production > seuil réaliste)
    pv_bounds = {
        "pv_central_valais": (0, 5000),  # kWh/15min
        "pv_sion": (0, 1500),
        "pv_sierre": (0, 2000),
        "pv_remote": (0, 80000),
    }
    df = apply_physical_bounds(df, pv_bounds, name="[Oiken] ")

    # 7. Interpolation des NaN (max 6 pas = 1h30 pour données 15min)
    df = interpolate_gaps(df, max_gap=6, name="[Oiken] ")

    # 8. Resampling 15min → 1h
    #    - Charge (standardisée, sans unité) : moyenne sur l'heure
    #    - PV (kWh sur 15min) : somme sur l'heure = kWh sur 1h
    agg_rules = {}
    for col in ["load", "load_forecast"]:
        if col in df.columns:
            agg_rules[col] = "mean"
    for col in pv_cols:
        if col in df.columns:
            agg_rules[col] = "sum"
    df = df.resample(TARGET_FREQ).agg(agg_rules)

    # 9. Colonne PV total
    pv_cols_after = [c for c in df.columns if c.startswith("pv_")]
    if pv_cols_after:
        df["pv_total"] = df[pv_cols_after].sum(axis=1)

    log.info(f"  → {n_before} lignes (15min) → {len(df)} lignes (1h)")
    return df


def extract_best_spread(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Extrait best run, spread et n_runs pour une variable de prévision.

    Pour chaque heure cible (ligne), parmi les lead times disponibles (non-NaN) :
    - best      = valeur du lead time le plus petit (= prévision la plus fraîche)
    - spread    = écart-type entre toutes les valeurs disponibles
    - n_runs    = nombre de lead times non-NaN
    - lead_time = horizon du best run (en heures)

    Args:
        df: DataFrame brut avec colonnes {prefix}_lt1 à {prefix}_lt33
        prefix: ex. "GLOB_ctrl", "T_2M_ctrl"

    Returns
    -------
        DataFrame avec colonnes: {short}_best, {short}_spread,
        {short}_n_runs, {short}_lead_time
    """
    lt_cols = sorted(
        [c for c in df.columns if c.startswith(prefix + "_lt")],
        key=lambda c: int(c.split("_lt")[1]),
    )
    if not lt_cols:
        return pd.DataFrame(index=df.index)

    lead_times = [int(c.split("_lt")[1]) for c in lt_cols]
    short = prefix.replace("_ctrl", "")
    sub = df[lt_cols].copy()

    # Best = valeur du lead time le plus petit non-NaN
    best = pd.Series(np.nan, index=df.index, name=f"{short}_best")
    best_lt = pd.Series(np.nan, index=df.index, name=f"{short}_lead_time")
    for lt, col in zip(lead_times, lt_cols):
        mask = best.isna() & sub[col].notna()
        best[mask] = sub.loc[mask, col]
        best_lt[mask] = lt

    # Spread = écart-type entre les lead times disponibles
    spread = sub.std(axis=1, skipna=True).fillna(0)
    spread.name = f"{short}_spread"

    # N_runs = nombre de lead times non-NaN
    n_runs = sub.notna().sum(axis=1)
    n_runs.name = f"{short}_n_runs"

    result = pd.concat([best, spread, n_runs, best_lt], axis=1)
    log.info(
        f"  {short}: best/spread/n_runs extraits "
        f"(n_runs médian={n_runs.median():.0f}, "
        f"lead_time médian={best_lt.median():.0f}h)"
    )
    return result


def clean_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les prévisions météo et extrait best run + spread.

    Traitements appliqués :
    1. Remplacement des sentinelles -99999 par NaN
    2. Localisation UTC et suppression doublons
    3. Extraction best/spread/n_runs par variable
    4. Bornes physiques sur les best runs et spreads
    5. Forcer radiation/sunshine à 0 la nuit
    6. Forcer les non-négatifs à 0
    7. Conversion pression Pa → hPa
    8. Détection des sauts et outliers statistiques
    9. Cohérence croisée radiation ↔ sunshine
    10. Interpolation des NaN (sauf n_runs et lead_time)
    """
    log.info("Nettoyage prévisions météo...")

    # 1. Sentinelles -99999 → NaN
    n_sentinel = (df == SENTINEL).sum().sum()
    if n_sentinel > 0:
        log.info(f"  {n_sentinel} valeurs sentinelles (-99999) → NaN")
        df = df.replace(SENTINEL, np.nan)

    # 2. Timezone et doublons
    _idx = cast(pd.DatetimeIndex, df.index)
    if _idx.tz is None:
        df.index = _idx.tz_localize("UTC")
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        log.warning(f"  {n_dup} doublons de timestamp supprimés")
        df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    detect_temporal_gaps(df, TARGET_FREQ, name="[Prév] ")

    # 3. Extraction best/spread/n_runs par variable
    variable_prefixes = {
        "T_2M_ctrl": "pred_temperature",
        "GLOB_ctrl": "pred_radiation",
        "TOT_PREC_ctrl": "pred_precipitation",
        "DURSUN_ctrl": "pred_sunshine",
        "PS_ctrl": "pred_pressure",
        "RELHUM_2M_ctrl": "pred_humidity",
        "FF_10M_ctrl": "pred_wind_speed",
        "DD_10M_ctrl": "pred_wind_dir",
    }

    result = pd.DataFrame(index=df.index)
    for prefix, _legacy_name in variable_prefixes.items():
        extracted = extract_best_spread(df, prefix)
        if not extracted.empty:
            result = pd.concat([result, extracted], axis=1)

    # Alias pour compatibilité avec les noms pred_*
    alias_map = {
        "T_2M_best": "pred_temperature",
        "GLOB_best": "pred_radiation",
        "TOT_PREC_best": "pred_precipitation",
        "DURSUN_best": "pred_sunshine",
        "PS_best": "pred_pressure",
        "RELHUM_2M_best": "pred_humidity",
        "FF_10M_best": "pred_wind_speed",
        "DD_10M_best": "pred_wind_dir",
        "T_2M_spread": "pred_temperature_spread",
        "GLOB_spread": "pred_radiation_spread",
        "TOT_PREC_spread": "pred_precipitation_spread",
        "DURSUN_spread": "pred_sunshine_spread",
        "PS_spread": "pred_pressure_spread",
        "RELHUM_2M_spread": "pred_humidity_spread",
        "FF_10M_spread": "pred_wind_speed_spread",
        "DD_10M_spread": "pred_wind_dir_spread",
    }
    for src, dst in alias_map.items():
        if src in result.columns:
            result[dst] = result[src]

    log.info(f"  → {len(result.columns)} colonnes extraites (best + spread + n_runs + aliases)")

    # 4. Bornes physiques sur les best runs
    physical_bounds = {
        "GLOB_best": (0, 1200),  # W/m²
        "T_2M_best": (-35, 45),  # °C
        "PS_best": (85000, 105000),  # Pa (avant conversion)
        "TOT_PREC_best": (0, 50),  # mm/h
        "DURSUN_best": (0, 3600),  # secondes
        "RELHUM_2M_best": (0, 100),  # %
        "FF_10M_best": (0, 50),  # m/s
        "DD_10M_best": (0, 360),  # degrés
        "pred_radiation": (0, 1200),
        "pred_temperature": (-35, 45),
        "pred_pressure": (85000, 105000),
        "pred_precipitation": (0, 50),
        "pred_sunshine": (0, 3600),
        "pred_humidity": (0, 100),
        "pred_wind_speed": (0, 50),
        "pred_wind_dir": (0, 360),
    }
    result = apply_physical_bounds(result, physical_bounds, name="[Prév] ")

    # Bornes sur les spreads
    spread_bounds = {
        "GLOB_spread": (0, 600),
        "T_2M_spread": (0, 20),
        "TOT_PREC_spread": (0, 30),
        "DURSUN_spread": (0, 3600),
        "PS_spread": (0, 5000),
        "pred_radiation_spread": (0, 600),
        "pred_temperature_spread": (0, 20),
        "pred_precipitation_spread": (0, 30),
        "pred_sunshine_spread": (0, 3600),
        "pred_pressure_spread": (0, 5000),
        "pred_humidity_spread": (0, 50),
        "pred_wind_speed_spread": (0, 30),
        "pred_wind_dir_spread": (0, 360),
    }
    result = apply_physical_bounds(result, spread_bounds, name="[Prév spread] ")

    # 5. Forcer radiation et sunshine à 0 la nuit
    solar_cols = [
        c
        for c in result.columns
        if any(s in c for s in ["GLOB", "DURSUN", "pred_radiation", "pred_sunshine"])
        and "n_runs" not in c
        and "lead_time" not in c
    ]
    result = enforce_night_zero(result, solar_cols, name="[Prév] ")

    # 6. Forcer les non-négatifs
    non_neg = [
        c
        for c in result.columns
        if any(
            s in c
            for s in [
                "GLOB",
                "TOT_PREC",
                "DURSUN",
                "pred_radiation",
                "pred_precipitation",
                "pred_sunshine",
            ]
        )
        and "lead_time" not in c
    ]
    result = clip_negatives(result, non_neg, name="[Prév] ")

    # 7. Conversion pression Pa → hPa
    pressure_cols = [
        c
        for c in result.columns
        if ("PS" in c or "pred_pressure" in c) and "n_runs" not in c and "lead_time" not in c
    ]
    for col in pressure_cols:
        if col in result.columns and result[col].median() > 10000:
            result[col] = result[col] / 100.0
    if pressure_cols:
        log.info(f"  Pression convertie Pa → hPa ({len(pressure_cols)} colonnes)")
        # Mettre à jour les bornes après conversion pour les étapes suivantes
        # (les bornes physiques ont déjà été appliquées en Pa, les prochaines
        #  détections travailleront sur des valeurs en hPa)

    # 8. Sauts et outliers statistiques sur les best runs
    pred_deltas = {
        "T_2M_best": MAX_DELTA["temperature"],
        "PS_best": MAX_DELTA["pressure"],
        "pred_temperature": MAX_DELTA["temperature"],
        "pred_pressure": MAX_DELTA["pressure"],
    }
    result = detect_spikes(result, pred_deltas, name="[Prév] ")

    zscore_cols = [
        c
        for c in ["T_2M_best", "PS_best", "pred_temperature", "pred_pressure"]
        if c in result.columns
    ]
    result = detect_zscore_outliers(
        result, zscore_cols, window=168, min_periods=24, name="[Prév] "
    )

    # 9. Cohérence croisée radiation ↔ sunshine
    result = check_cross_consistency(result, name="[Prév] ")

    # 10. Interpolation (sauf n_runs et lead_time qui sont des métadonnées entières)
    interp_cols = [c for c in result.columns if "n_runs" not in c and "lead_time" not in c]
    non_interp_cols = [c for c in result.columns if c not in interp_cols]
    result_interp = interpolate_gaps(result[interp_cols], max_gap=6, name="[Prév] ")
    result = pd.concat([result_interp, result[non_interp_cols]], axis=1)

    log.info(f"  → {len(result)} lignes, {len(result.columns)} variables")
    return result


# ═════════════════════════════════════════════
# ÉTAPE 3 — Fusion et export
# ═════════════════════════════════════════════


def merge_datasets(
    oiken: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne Oiken + prévisions sur l'index temporel commun.

    Utilise un outer join pour conserver toutes les heures couvertes
    par au moins une source. Les colonnes sans données restent en NaN.
    """
    log.info("Fusion des datasets...")

    for name, d in [("oiken", oiken), ("forecast", forecast)]:
        _idx = cast(pd.DatetimeIndex, d.index)
        if _idx.tz is None:
            log.warning(f"  {name}: index sans timezone, localisation en UTC")
            d.index = _idx.tz_localize("UTC")

    merged = oiken.join(forecast, how="outer")

    detect_temporal_gaps(merged, TARGET_FREQ, name="[Fusionné] ")

    n_total = len(merged)
    log.info("  Couverture des colonnes:")
    for col in merged.columns:
        coverage = merged[col].notna().sum() / n_total * 100
        log.info(f"    {col}: {coverage:.1f}%")

    log.info(f"  → Dataset fusionné: {len(merged)} lignes, {len(merged.columns)} colonnes")
    log.info(f"  → Période: {merged.index.min()} → {merged.index.max()}")
    return merged


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════


def run_pipeline(
    oiken_path: str,
    forecast_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Exécute le pipeline complet : chargement → nettoyage → fusion → export."""
    log.info("=" * 60)
    log.info("Pipeline d'acquisition v2 (sans météo réelle)")
    log.info("=" * 60)

    # Étape 1 — Chargement
    oiken_raw = load_oiken(oiken_path)
    forecast_raw = load_forecast(forecast_path)

    # Étape 2 — Nettoyage
    oiken_clean = clean_oiken(oiken_raw)
    forecast_clean = clean_forecast(forecast_raw)

    # Étape 3 — Fusion
    dataset = merge_datasets(oiken_clean, forecast_clean)

    # Export
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output)
    log.info(f"Dataset sauvegardé: {output}")
    log.info(f"  → {len(dataset)} lignes, {len(dataset.columns)} colonnes")

    # Résumé rapide
    log.info("=" * 60)
    log.info("Colonnes finales:")
    for col in sorted(dataset.columns):
        dtype = dataset[col].dtype
        na_pct = dataset[col].isna().mean() * 100
        log.info(f"  {col}: {dtype}, {na_pct:.1f}% NaN")
    log.info("=" * 60)

    return dataset


if __name__ == "__main__":
    # Résoudre les chemins par rapport à la racine du projet
    # (le script est dans code_data/, les données dans data/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    def _find_latest(pattern: str) -> str | None:
        matches = sorted(
            DATA_DIR.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return str(matches[0]) if matches else None

    parser = argparse.ArgumentParser(
        description="Pipeline de nettoyage des données (v2 — sans météo réelle)"
    )
    parser.add_argument(
        "--oiken",
        default=_find_latest("oiken_*.csv"),
        help="Chemin du CSV Oiken",
    )
    parser.add_argument(
        "--forecast",
        default=str(DATA_DIR / "prevision_data.csv"),
        help="Chemin du CSV prévisions COSMO/ICON",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_DIR / "dataset_clean.csv"),
        help="Chemin du CSV de sortie",
    )
    args = parser.parse_args()

    run_pipeline(args.oiken, args.forecast, args.output)
