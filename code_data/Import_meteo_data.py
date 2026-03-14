"""Module for importing meteo data."""

import certifi
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from influxdb_client.client.influxdb_client import InfluxDBClient

load_dotenv()

if __name__ == "__main__":
    org = os.getenv("INFLUXDB_ORG", "")
    bucket = os.getenv("INFLUXDB_BUCKET", "")
    token = os.getenv("INFLUXDB_TOKEN", "")
    url = os.getenv("INFLUXDB_URL", "")
    client = InfluxDBClient(
        url="https://timeseries.hevs.ch",
        token=token,
        org=org,
        ssl_ca_cert=certifi.where(),
        timeout=1000000,
    )

    measurements = [
        "Air temperature 2m above ground (current value)",
        "Atmospheric pressure at barometric altitude",
        "Global radiation (ten minutes mean)",
        "Precipitation (ten minutes total)",
        "Sunshine duration (ten minutes total)",
    ]
    measurement_set = ", ".join(f'"{measurement}"' for measurement in measurements)
    query = f'''
from(bucket: "{bucket}")
  |> range(start: -12d)
  |> filter(fn: (r) => r.Site == "Sion")
  |> filter(fn: (r) => contains(value: r._measurement, set: [{measurement_set}]))
  |> filter(fn: (r) => r._field == "Value")
'''
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
    client.close()

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = data_dir / f"sion_meteo_{timestamp}.csv"

    # Save to CSV (one column per measurement)
    df = pd.DataFrame(records)
    if not df.empty:
        # Add an index for duplicate timestamp-measurement pairs
        df["dup_idx"] = df.groupby(["timestamp", "measurement"]).cumcount()
        df = df.pivot_table(
            index="timestamp",
            columns=["measurement", "dup_idx"],
            values="value",
        ).sort_index()
        # Flatten multi-level columns: PRED_T_2M_ctrl_0, PRED_T_2M_ctrl_1, etc.
        df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
        df = df.reset_index()
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")
