"""Module for importing Oiken data."""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

DATA_PATH = r"C:\Users\metra\Desktop\3ème\6eme semestre\EnInf 2\data\oiken-data clean.csv"

df = pl.read_csv(
    DATA_PATH,
    separator=";",
    decimal_comma=True,
    try_parse_dates=True,
    null_values=["#N/A", "N/A", "", "NA"],
    schema_overrides={
        "standardised load [-]": pl.Float64,
        "standardised forecast load [-]": pl.Float64,
        "central valais solar production [kWh]": pl.Float64,
        "sion area solar production [kWh]": pl.Float64,
        "sierre area production [kWh]": pl.Float64,
        "remote solar production [kWh]": pl.Float64,
    },
)

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = data_dir / f"oiken_{timestamp}.csv"

df.write_csv(filename)
print(f"Data saved to {filename}")
