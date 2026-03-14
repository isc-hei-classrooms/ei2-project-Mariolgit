"""Pipeline d'acquisition et de nettoyage data."""

import argparse
import logging
from pathlib import Path

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

# Sentinel value used in forecast data to indicate missing/invalid
SENTINEL = -99999.0

# Target temporal resolution
TARGET_FREQ = "1h"

# Night hours (UTC) where solar production/radiation MUST be 0
# Conservative: safe for Sion all year round (sunrise never before 05h UTC,
# sunset never after 21h UTC, even in summer)
NIGHT_START_UTC = 21  # 21h UTC = earliest possible full dark
NIGHT_END_UTC = 5  # 5h UTC  = latest possible still dark

# Z-score threshold for statistical outlier detection
ZSCORE_THRESHOLD = 4.0

# Maximum plausible change per time step (for spike detection)
MAX_DELTA = {
    "temperature": 8.0,  # °C per hour (extreme fronts)
    "pressure": 5.0,  # hPa per hour
    "radiation": 600.0,  # W/m² per hour (sunrise/sunset transitions)
    "precipitation": 15.0,  # mm per hour
}


# ═════════════════════════════════════════════
# FONCTIONS UTILITAIRES DE NETTOYAGE
# ═════════════════════════════════════════════


def is_night(index: pd.DatetimeIndex) -> pd.Series:
    """Retourne un masque booléen : True si l'heure est en pleine nuit (UTC).

    Nuit = [NIGHT_START_UTC, 23] ∪ [0, NIGHT_END_UTC]
    """
    hours = index.hour  # type: ignore[attr-defined]
    return (hours >= NIGHT_START_UTC) | (hours <= NIGHT_END_UTC)


def clip_negatives(df: pd.DataFrame, columns: list, name: str = "") -> pd.DataFrame:
    """Force les valeurs négatives à 0 pour les colonnes spécifiées.

    Concerne les variables physiquement non-négatives :
    production PV, radiation, précipitations, ensoleillement.
    """
    for col in columns:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                log.warning(f"  {name}{col}: {n_neg} valeurs négatives → 0")
                df[col] = df[col].clip(lower=0)
    return df


def enforce_night_zero(df: pd.DataFrame, columns: list, name: str = "") -> pd.DataFrame:
    """Force à 0 les colonnes solaires pendant la nuit.

    La production PV et la radiation sont physiquement impossibles la nuit.
    Toute valeur > 0 la nuit est une erreur de mesure ou un bruit d'onduleur.
    """
    night_mask = is_night(df.index)  # type: ignore[arg-type]
    for col in columns:
        if col in df.columns:
            night_nonzero = (df.loc[night_mask, col] > 0).sum()
            if night_nonzero > 0:
                log.info(f"  {name}{col}: {night_nonzero} valeurs nocturnes → 0")
                df.loc[night_mask, col] = 0.0
    return df


def apply_physical_bounds(df: pd.DataFrame, bounds: dict, name: str = "") -> pd.DataFrame:
    """Remplace par NaN les valeurs hors des bornes physiques réalistes.

    Les bornes sont définies par variable et correspondent aux limites
    physiquement possibles pour le site de Sion (Valais, ~480m alt.).
    """
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
    """Détecte et remplace les outliers statistiques (z-score) par NaN.

    Utilise une fenêtre glissante pour calculer la moyenne et l'écart-type
    locaux, ce qui respecte la saisonnalité journalière et hebdomadaire.
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
    """Détecte les variations trop brutales entre pas de temps consécutifs.

    Un saut de température de 20°C en 1h est physiquement impossible
    et indique une erreur de capteur. Ces points sont remplacés par NaN.
    """
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
    """Vérifie la cohérence entre variables liées.

    Règles :
    - Si radiation == 0, alors sunshine devrait être ~0 (tolérance 1 min)
    - Si sunshine > 0, alors radiation devrait être > 0
    """
    if "radiation" in df.columns and "sunshine" in df.columns:
        mask = (df["radiation"] == 0) & (df["sunshine"] > 1)
        n = mask.sum()
        if n > 0:
            log.info(f"  {name}Incohérence radiation=0 mais sunshine>1: {n} → sunshine=0")
            df.loc[mask, "sunshine"] = 0

    if "pred_radiation" in df.columns and "pred_sunshine" in df.columns:
        mask = (df["pred_radiation"] == 0) & (df["pred_sunshine"] > 1)
        n = mask.sum()
        if n > 0:
            log.info(f"  {name}Incohérence pred_radiation=0 mais pred_sunshine>1: {n} → 0")
            df.loc[mask, "pred_sunshine"] = 0

    return df


def interpolate_gaps(df: pd.DataFrame, max_gap: int = 6, name: str = "") -> pd.DataFrame:
    """Interpole les NaN avec une limite de trou maximum.

    - Trous ≤ max_gap pas de temps : interpolation linéaire temporelle
    - Trous > max_gap pas de temps : laissés en NaN (données manquantes réelles)

    Cela évite d'inventer des données sur de longues périodes.
    """
    n_before = df.isna().sum().sum()
    if n_before > 0:
        df = df.interpolate(method="time", limit=max_gap)
        # Forward/backward fill uniquement pour les bords (max 2 pas)
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
        for idx, gap in big_gaps.head(10).items():  # type: ignore[attr-defined]
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


def load_meteo(path: str) -> pd.DataFrame:
    """Charge les données météo réelles (observations au pas 10 min)."""
    log.info(f"Chargement météo réelle: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])

    rename_map = {
        "Air temperature 2m above ground (current value)_0": "temperature",
        "Atmospheric pressure at barometric altitude_0": "pressure",
        "Global radiation (ten minutes mean)_0": "radiation",
        "Precipitation (ten minutes total)_0": "precipitation",
        "Sunshine duration (ten minutes total)_0": "sunshine",
    }
    df = df.rename(columns=rename_map)
    df = df.set_index("timestamp")

    log.info(f"  → {len(df)} lignes, période: {df.index.min()} → {df.index.max()}")
    return df


def load_forecast(path: str) -> pd.DataFrame:
    """Charge les prévisions météo (ensemble de runs, pas 1h)."""
    log.info(f"Chargement prévisions météo: {path}")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")

    log.info(f"  → {len(df)} lignes, {len(df.columns)} colonnes")
    log.info(f"  → période: {df.index.min()} → {df.index.max()}")
    return df


# ═════════════════════════════════════════════
# ÉTAPE 2 — Nettoyage de chaque source
# ═════════════════════════════════════════════


def clean_oiken(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données Oiken.

    Traitements appliqués :
    1. Localisation UTC et suppression des doublons
    2. Détection des trous temporels
    3. Forcer les valeurs PV négatives à 0
    4. Forcer la production PV à 0 pendant la nuit
    5. Détection des outliers statistiques (z-score glissant) sur la charge
    6. Bornes physiques sur la production PV
    7. Interpolation des NaN (max 6 pas de temps)
    8. Resampling 15min → 1h
    9. Création de la colonne PV total
    """
    log.info("Nettoyage Oiken...")
    n_before = len(df)

    # 1. Timezone et doublons
    if df.index.tz is None:  # type: ignore[union-attr]
        df.index = df.index.tz_localize("UTC")  # type: ignore[union-attr]
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        log.warning(f"  {n_dup} doublons de timestamp supprimés")
        df = df[~df.index.duplicated(keep="first")]  # type: ignore[assignment]
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

    # 8. Resample 15min → 1h
    agg_rules = {"load": "mean", "load_forecast": "mean"}
    for col in pv_cols:
        agg_rules[col] = "sum"
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    df = df.resample(TARGET_FREQ).agg(agg_rules)  # type: ignore[assignment]

    # 9. Colonne PV total
    pv_cols_after = [c for c in df.columns if c.startswith("pv_")]
    if pv_cols_after:
        df["pv_total"] = df[pv_cols_after].sum(axis=1)

    log.info(f"  → {n_before} lignes (15min) → {len(df)} lignes (1h)")
    return df


def clean_meteo(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les données météo réelles.

    Traitements appliqués :
    1. Suppression des doublons, tri, détection trous
    2. Bornes physiques réalistes (Sion, Valais, ~480m)
    3. Forcer radiation et sunshine à 0 la nuit
    4. Forcer les valeurs non-négatives à 0 (radiation, précipitations, sunshine)
    5. Détection des sauts brutaux (spikes de capteur)
    6. Détection des outliers statistiques (z-score glissant)
    7. Cohérence croisée radiation ↔ sunshine
    8. Interpolation des NaN (max 6 pas)
    9. Resampling 10min → 1h
    """
    log.info("Nettoyage météo réelle...")
    n_before = len(df)

    # 1. Doublons et tri
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        log.warning(f"  {n_dup} doublons supprimés")
        df = df[~df.index.duplicated(keep="first")]  # type: ignore[assignment]
    df = df.sort_index()
    detect_temporal_gaps(df, "10min", name="[Météo] ")

    # 2. Bornes physiques réalistes pour Sion
    physical_bounds = {
        "temperature": (-35, 45),  # °C (record Sion : -22°C / +38°C, avec marge)
        "pressure": (920, 980),  # hPa (Sion ~480m altitude)
        "radiation": (0, 1200),  # W/m² (max théorique ~1100 en été)
        "precipitation": (0, 20),  # mm/10min (événement extrême)
        "sunshine": (0, 10),  # minutes sur 10min
    }
    df = apply_physical_bounds(df, physical_bounds, name="[Météo] ")

    # 3. Forcer radiation et sunshine à 0 la nuit
    solar_cols = [c for c in ["radiation", "sunshine"] if c in df.columns]
    df = enforce_night_zero(df, solar_cols, name="[Météo] ")

    # 4. Forcer les valeurs non-négatives
    non_neg_cols = [c for c in ["radiation", "precipitation", "sunshine"] if c in df.columns]
    df = clip_negatives(df, non_neg_cols, name="[Météo] ")

    # 5. Détection des sauts brutaux (seuils adaptés au pas 10min)
    #    Sion en vallée du Rhône : le foehn peut créer des variations de 2-3°C/10min
    max_deltas_10min = {
        "temperature": 4.0,  # °C/10min (foehn, inversions thermiques)
        "pressure": 1.5,  # hPa/10min
        "radiation": 300.0,  # W/m²/10min (passages nuageux rapides)
        "precipitation": 5.0,  # mm/10min
    }
    df = detect_spikes(df, max_deltas_10min, name="[Météo] ")

    # 6. Outliers statistiques (z-score glissant, fenêtre 1 semaine = 1008 pas de 10min)
    zscore_cols = [c for c in ["temperature", "pressure"] if c in df.columns]
    df = detect_zscore_outliers(df, zscore_cols, window=1008, min_periods=144, name="[Météo] ")

    # 7. Cohérence croisée radiation ↔ sunshine
    df = check_cross_consistency(df, name="[Météo] ")

    # 8. Interpolation (max 6 pas = 1h pour données 10min)
    df = interpolate_gaps(df, max_gap=6, name="[Météo] ")

    # 9. Resample 10min → 1h
    agg_rules = {
        "temperature": "mean",
        "pressure": "mean",
        "radiation": "mean",
        "precipitation": "sum",
        "sunshine": "sum",
    }
    agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    df = df.resample(TARGET_FREQ).agg(agg_rules)  # type: ignore[assignment]

    log.info(f"  → {n_before} lignes (10min) → {len(df)} lignes (1h)")
    return df


def clean_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les prévisions météo (ensemble).

    Traitements appliqués :
    1. Remplacement des sentinelles -99999 par NaN
    2. Exclusion des runs trop incomplets (> 50% NaN)
    3. Moyenne d'ensemble par variable
    4. Bornes physiques
    5. Forcer radiation/sunshine à 0 la nuit
    6. Forcer les non-négatifs à 0
    7. Conversion pression Pa → hPa
    8. Détection des sauts et outliers statistiques
    9. Cohérence croisée radiation ↔ sunshine
    10. Interpolation des NaN
    """
    log.info("Nettoyage prévisions météo...")

    # 1. Sentinelles -99999 → NaN
    n_sentinel = (df == SENTINEL).sum().sum()
    if n_sentinel > 0:
        log.info(f"  {n_sentinel} valeurs sentinelles (-99999) → NaN")
        df = df.replace(SENTINEL, np.nan)

    # 2-3. Moyenne d'ensemble par variable
    variable_map = {
        "PRED_T_2M_ctrl": "pred_temperature",
        "PRED_GLOB_ctrl": "pred_radiation",
        "PRED_TOT_PREC_ctrl": "pred_precipitation",
        "PRED_DURSUN_ctrl": "pred_sunshine",
        "PRED_PS_q10": "pred_pressure",
    }

    result = pd.DataFrame(index=df.index)
    for prefix, new_name in variable_map.items():
        run_cols = [c for c in df.columns if c.startswith(prefix + "_")]
        if not run_cols:
            log.warning(f"  Aucune colonne trouvée pour {prefix}")
            continue

        threshold = len(df) * 0.5
        valid_cols = [c for c in run_cols if df[c].isna().sum() < threshold]
        excluded = set(run_cols) - set(valid_cols)
        if excluded:
            log.info(f"  {new_name}: exclusion de {len(excluded)} runs (trop de NaN)")

        if valid_cols:
            result[new_name] = df[valid_cols].mean(axis=1)
            log.info(f"  {new_name}: moyenne de {len(valid_cols)} runs")
        else:
            log.warning(f"  {new_name}: aucun run valide !")

    # 4. Bornes physiques
    physical_bounds = {
        "pred_temperature": (-35, 45),
        "pred_pressure": (85000, 105000),  # Pa
        "pred_radiation": (-1, 1200),
        "pred_precipitation": (0, 50),
        "pred_sunshine": (0, 61),
    }
    result = apply_physical_bounds(result, physical_bounds, name="[Prév] ")

    # 5. Forcer radiation et sunshine à 0 la nuit
    solar_cols = [c for c in ["pred_radiation", "pred_sunshine"] if c in result.columns]
    result = enforce_night_zero(result, solar_cols, name="[Prév] ")

    # 6. Forcer les non-négatifs
    non_neg = [
        c for c in ["pred_radiation", "pred_precipitation", "pred_sunshine"] if c in result.columns
    ]
    result = clip_negatives(result, non_neg, name="[Prév] ")

    # 7. Conversion pression Pa → hPa
    if "pred_pressure" in result.columns:
        result["pred_pressure"] = result["pred_pressure"] / 100.0
        log.info("  Pression convertie Pa → hPa")

    # 8. Sauts et outliers statistiques
    pred_deltas = {
        "pred_temperature": MAX_DELTA["temperature"],
        "pred_pressure": MAX_DELTA["pressure"],
    }
    result = detect_spikes(result, pred_deltas, name="[Prév] ")
    zscore_cols = [c for c in ["pred_temperature", "pred_pressure"] if c in result.columns]
    result = detect_zscore_outliers(
        result, zscore_cols, window=168, min_periods=24, name="[Prév] "
    )

    # 9. Cohérence croisée
    result = check_cross_consistency(result, name="[Prév] ")

    # 10. Interpolation
    result = interpolate_gaps(result, max_gap=6, name="[Prév] ")

    log.info(f"  → {len(result)} lignes, {len(result.columns)} variables")
    return result


# ═════════════════════════════════════════════
# ÉTAPE 3 — Fusion et export
# ═════════════════════════════════════════════


def merge_datasets(
    oiken: pd.DataFrame,
    meteo: pd.DataFrame,
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Fusionne les 3 sources sur l'index temporel commun (outer join)."""
    log.info("Fusion des datasets...")

    for name, d in [("oiken", oiken), ("meteo", meteo), ("forecast", forecast)]:
        if d.index.tz is None:  # type: ignore[union-attr]
            log.warning(f"  {name}: index sans timezone, localisation en UTC")
            d.index = d.index.tz_localize("UTC")  # type: ignore[union-attr]

    merged = oiken.join(meteo, how="outer")
    merged = merged.join(forecast, how="outer")

    detect_temporal_gaps(merged, TARGET_FREQ, name="[Fusionné] ")

    n_total = len(merged)
    log.info("  Couverture des colonnes:")
    for col in merged.columns:
        coverage = merged[col].notna().sum() / n_total * 100
        log.info(f"    {col}: {coverage:.1f}%")

    log.info(f"  → Dataset fusionné: {len(merged)} lignes, {len(merged.columns)} colonnes")
    log.info(f"  → Période: {merged.index.min()} → {merged.index.max()}")
    return merged


def generate_report(df: pd.DataFrame) -> str:
    """Génère un rapport textuel résumant le dataset nettoyé."""
    lines = [
        "=" * 70,
        "RAPPORT DU PIPELINE — Dataset nettoyé",
        "=" * 70,
        f"Période      : {df.index.min()} → {df.index.max()}",
        f"Pas temporel : {TARGET_FREQ}",
        f"Lignes       : {len(df)}",
        f"Colonnes     : {len(df.columns)}",
        "",
        "─── Couverture par colonne ───",
    ]
    for col in df.columns:
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        lines.append(f"  {col:<30s} {n_valid:>6d}/{len(df)} ({pct:.1f}%)")

    lines.append("")
    lines.append("─── Contrôles qualité ───")

    # Vérification PV nuit
    pv_cols = [c for c in df.columns if c.startswith("pv_")]
    night_mask = is_night(df.index)  # type: ignore[arg-type]
    for col in pv_cols:
        if col in df.columns:
            night_prod = (df.loc[night_mask, col] > 0).sum()
            lines.append(f"  {col} production nocturne > 0: {night_prod}")

    # Radiation nuit
    for col in ["radiation", "pred_radiation"]:
        if col in df.columns:
            night_rad = (df.loc[night_mask, col] > 0).sum()
            lines.append(f"  {col} radiation nocturne > 0: {night_rad}")

    # Valeurs négatives
    non_neg_cols = pv_cols + [
        c
        for c in [
            "radiation",
            "precipitation",
            "sunshine",
            "pred_radiation",
            "pred_precipitation",
            "pred_sunshine",
        ]
        if c in df.columns
    ]
    for col in non_neg_cols:
        if col in df.columns:
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                lines.append(f"  ⚠️ {col}: {n_neg} valeurs négatives restantes")

    # NaN restants
    lines.append("")
    lines.append("─── NaN restants par colonne ───")
    for col in df.columns:
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            lines.append(f"  {col}: {n_nan} NaN ({n_nan / len(df) * 100:.1f}%)")

    lines.extend(
        [
            "",
            "─── Statistiques descriptives ───",
            df.describe().round(3).to_string(),
        ]
    )
    return "\n".join(lines)


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════


def run_pipeline(
    oiken_path: str,
    meteo_path: str,
    forecast_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Exécute le pipeline complet."""
    log.info("🚀 Démarrage du pipeline d'acquisition")

    # Étape 1 — Chargement
    oiken_raw = load_oiken(oiken_path)
    meteo_raw = load_meteo(meteo_path)
    forecast_raw = load_forecast(forecast_path)

    # Étape 2 — Nettoyage
    oiken_clean = clean_oiken(oiken_raw)
    meteo_clean = clean_meteo(meteo_raw)
    forecast_clean = clean_forecast(forecast_raw)

    # Étape 3 — Fusion
    dataset = merge_datasets(oiken_clean, meteo_clean, forecast_clean)

    # Export
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output)
    log.info(f"✅ Dataset sauvegardé: {output}")

    # Rapport
    report = generate_report(dataset)
    report_path = output.with_suffix(".report.txt")
    report_path.write_text(report, encoding="utf-8")
    log.info(f"📊 Rapport sauvegardé: {report_path}")
    print("\n" + report)

    return dataset


if __name__ == "__main__":

    def _find_latest(pattern: str) -> str | None:
        matches = sorted(Path("data").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(matches[0]) if matches else None

    parser = argparse.ArgumentParser(description="Pipeline de nettoyage des données")
    parser.add_argument("--oiken", default=_find_latest("oiken_*.csv"), help="Chemin du CSV Oiken")
    parser.add_argument(
        "--meteo", default=_find_latest("sion_meteo_*.csv"), help="Chemin du CSV météo réelle"
    )
    parser.add_argument(
        "--forecast", default=_find_latest("sion_prevision_*.csv"), help="Chemin du CSV prévisions"
    )
    parser.add_argument(
        "--output", default="data/dataset_clean.csv", help="Chemin du CSV de sortie"
    )
    args = parser.parse_args()

    run_pipeline(args.oiken, args.meteo, args.forecast, args.output)
