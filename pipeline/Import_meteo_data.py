"""Import des données météo réelles observées depuis InfluxDB.

Télécharge par morceaux mensuels (Oct 2022 → Oct 2025) pour éviter la
troncature silencieuse d'InfluxDB sur les grandes plages temporelles.

Post-traitement :
  - Pivot long → wide
  - Renommage des colonnes (noms courts *_obs)
  - Resampling 10 min → 1 h (alignement avec COSMO)
  - Rapport qualité (NaN, trous)

Usage :
    python pipeline/Import_meteo_data.py
"""

import certifi
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MEASUREMENTS = [
    "Air temperature 2m above ground (current value)",
    "Atmospheric pressure at barometric altitude",
    "Global radiation (ten minutes mean)",
    "Precipitation (ten minutes total)",
    "Sunshine duration (ten minutes total)",
]

# Colonnes _0 conservées après pivot + noms courts
COLUMN_RENAME = {
    "Air temperature 2m above ground (current value)_0": "temp_obs",
    "Atmospheric pressure at barometric altitude_0": "pressure_obs",
    "Global radiation (ten minutes mean)_0": "radiation_obs",
    "Precipitation (ten minutes total)_0": "precip_obs",
    "Sunshine duration (ten minutes total)_0": "sunshine_obs",
}

# Agrégation lors du resampling 10 min → 1 h
RESAMPLE_AGG = {
    "temp_obs": "mean",
    "pressure_obs": "mean",
    "radiation_obs": "mean",
    "precip_obs": "sum",
    "sunshine_obs": "sum",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def query_meteo_chunk(
    client: InfluxDBClient,
    org: str,
    bucket: str,
    start_date: datetime,
    stop_date: datetime,
) -> list[dict]:
    """Requête InfluxDB pour une période donnée (un mois)."""
    measurement_set = ", ".join(f'"{m}"' for m in MEASUREMENTS)
    start_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
    stop_str = stop_date.strftime("%Y-%m-%dT00:00:00Z")

    query = f"""
from(bucket: "{bucket}")
  |> range(start: {start_str}, stop: {stop_str})
  |> filter(fn: (r) => r.Site == "Sion")
  |> filter(fn: (r) => contains(value: r._measurement, set: [{measurement_set}]))
  |> filter(fn: (r) => r._field == "Value")
"""
    tables = client.query_api().query(org=org, query=query)
    records = []
    for table in tables:
        for record in table.records:
            records.append(
                {
                    "timestamp": record["_time"],
                    "measurement": record["_measurement"],
                    "value": record["_value"],
                }
            )
    return records


def pivot_records(records: list[dict]) -> pd.DataFrame:
    """Convertit la liste de records en DataFrame large (un col par mesure)."""
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df["dup_idx"] = df.groupby(["timestamp", "measurement"]).cumcount()
    df = df.pivot_table(
        index="timestamp",
        columns=["measurement", "dup_idx"],
        values="value",
    ).sort_index()
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    df = df.reset_index()
    return df


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Conserve uniquement les colonnes _0 et les renomme."""
    keep = [c for c in df.columns if c == "timestamp" or c in COLUMN_RENAME]
    df = df[keep].rename(columns=COLUMN_RENAME)
    return df


def resample_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resampling 10 min → 1 h avec les agrégations appropriées."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.resample("1h").agg(RESAMPLE_AGG)
    df = df.reset_index()
    return df


def quality_report(df: pd.DataFrame) -> None:
    """Affiche un rapport sur les NaN et les trous > 1 h."""
    obs_cols = [c for c in df.columns if c != "timestamp"]
    print("\n─── Rapport qualité ───────────────────────────────────────")
    print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Lignes  : {len(df)}")
    print("\n  NaN par colonne :")
    for col in obs_cols:
        n_nan = df[col].isna().sum()
        pct = 100 * n_nan / len(df)
        print(f"    {col:<20} {n_nan:>5}  ({pct:.1f}%)")

    # Trous : lignes consécutives toutes NaN
    all_nan = df[obs_cols].isna().all(axis=1)
    in_gap = False
    gap_start = None
    gaps = []
    for ts, is_gap in zip(df["timestamp"], all_nan):
        if is_gap and not in_gap:
            gap_start = ts
            in_gap = True
        elif not is_gap and in_gap:
            gaps.append((gap_start, ts))
            in_gap = False
    if in_gap:
        gaps.append((gap_start, df["timestamp"].iloc[-1]))

    if gaps:
        print(f"\n  Trous complets ({len(gaps)}) :")
        for g_start, g_end in gaps:
            print(f"    {g_start} → {g_end}")
    else:
        print("\n  Aucun trou complet détecté.")
    print("───────────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    org = os.getenv("INFLUXDB_ORG", "")
    bucket = os.getenv("INFLUXDB_BUCKET", "")
    token = os.getenv("INFLUXDB_TOKEN", "")

    client = InfluxDBClient(
        url="https://timeseries.hevs.ch",
        token=token,
        org=org,
        ssl_ca_cert=certifi.where(),
        timeout=1_000_000,
    )

    date_start = datetime(2022, 10, 1)
    date_end = datetime(2025, 10, 1)

    total_months = (date_end.year - date_start.year) * 12 + date_end.month - date_start.month
    print(f"Téléchargement météo Sion : {date_start.date()} → {date_end.date()}")
    print(f"  ({total_months} mois à charger)\n")

    all_records: list[dict] = []
    current = date_start
    month_idx = 0

    while current < date_end:
        next_month = current + relativedelta(months=1)
        if next_month > date_end:
            next_month = date_end

        month_idx += 1
        print(
            f"  [{month_idx}/{total_months}] {current.strftime('%Y-%m')}...",
            end=" ",
            flush=True,
        )

        records = query_meteo_chunk(client, org, bucket, current, next_month)
        all_records.extend(records)
        print(f"{len(records)} enregistrements")

        current = next_month

    client.close()

    print(f"\nTotal brut : {len(all_records)} enregistrements")

    # Post-traitement
    df = pivot_records(all_records)

    if df.empty:
        print("Aucune donnée reçue — vérifier la connexion / les credentials.")
    else:
        df = select_and_rename(df)
        df = resample_to_hourly(df)
        quality_report(df)

        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = data_dir / f"sion_meteo_reelle_{timestamp_str}.csv"
        df.to_csv(filename, index=False)

        print(f"Sauvegardé : {filename}")
        print(f"  {len(df)} lignes × {len(df.columns)} colonnes")
