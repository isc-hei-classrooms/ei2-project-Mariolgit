"""EDA notebook — Oiken & Prévisions COSMO (sans météo réelle)."""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full", app_title="EDA — Oiken & Prévisions COSMO")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import scipy.stats as stats

    return go, np, pl, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **Features Engineering**
    ## Data : Oiken & Prévisions COSMO/ICON
    Données horaires Oct 2022 – Mar 2026 (sans météo réelle pour éviter le data leakage)
    ---
    """)
    return


@app.cell(hide_code=True)
def _(pl):
    df = pl.read_csv("data/dataset_clean.csv", try_parse_dates=True, infer_schema_length=None)
    df = df.with_columns(
        [
            pl.col("timestamp").dt.hour().alias("hour"),
            pl.col("timestamp").dt.weekday().alias("weekday"),
            pl.col("timestamp").dt.month().alias("month"),
            pl.when(pl.col("timestamp").dt.month().is_in([6, 7, 8]))
            .then(pl.lit("été"))
            .when(pl.col("timestamp").dt.month().is_in([12, 1, 2]))
            .then(pl.lit("hiver"))
            .otherwise(pl.lit("mi-saison"))
            .alias("season"),
        ]
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 — Exploration des données
    """)
    return


@app.cell(hide_code=True)
def _(df, mo):
    import datetime as _dt

    _ts_min = df["timestamp"].min()
    _ts_max = df["timestamp"].max()
    if isinstance(_ts_min, _dt.datetime):
        _d_min = _ts_min.date()
        _d_max = _ts_max.date()
    elif isinstance(_ts_min, _dt.date):
        _d_min = _ts_min
        _d_max = _ts_max
    else:
        _d_min = _dt.date(2022, 10, 1)
        _d_max = _dt.date(2026, 3, 31)
    date_range = mo.ui.date_range(
        start=_d_min,
        stop=_d_max,
        value=(_d_min, _d_max),
        label="Période d'analyse",
    )
    date_range
    return (date_range,)


@app.cell(hide_code=True)
def _(date_range, df, mo, pl):
    _start, _stop = date_range.value
    df_filtered = df.filter(pl.col("timestamp").dt.date().is_between(_start, _stop))
    _output = mo.vstack(
        [
            mo.md(
                f"**Shape** : {df_filtered.shape[0]:,} lignes × {df_filtered.shape[1]} colonnes"
            ),
            mo.md("### Valeurs nulles par colonne"),
            mo.as_html(df_filtered.null_count()),
            mo.md("### Statistiques descriptives"),
            mo.as_html(df_filtered.describe()),
        ]
    )
    _output
    return (df_filtered,)


@app.cell(hide_code=True)
def _(df_filtered, mo):
    mo.ui.table(df_filtered.head(500))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2 — Visualisation temporelle
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _numeric_vars = [
        # --- Oiken ---
        "load",
        "load_forecast",
        "pv_total",
        "pv_central_valais",
        "pv_sion",
        "pv_sierre",
        "pv_remote",
        # --- Prévisions best run ---
        "pred_temperature",
        "pred_radiation",
        "pred_sunshine",
        "pred_precipitation",
        "pred_pressure",
        "pred_humidity",
        "pred_wind_speed",
        # --- Prévisions spread ---
        "pred_temperature_spread",
        "pred_radiation_spread",
        "pred_sunshine_spread",
        "pred_precipitation_spread",
        "pred_pressure_spread",
    ]
    s2_vars = mo.ui.multiselect(
        options=_numeric_vars,
        value=["load"],
        label="Variables à visualiser",
    )
    s2_resolution = mo.ui.dropdown(
        options={"Heure": "1h", "Jour": "1d", "Semaine": "1w", "Mois": "1mo"},
        value="Jour",
        label="Résolution",
    )
    mo.hstack([s2_vars, s2_resolution])
    return s2_resolution, s2_vars


@app.cell(hide_code=True)
def _(df_filtered, go, mo, pl, s2_resolution, s2_vars):
    _every = s2_resolution.value
    _vars = [v for v in s2_vars.value if v in df_filtered.columns]
    _df_sorted = df_filtered.sort("timestamp")
    if _every == "1h":
        _df_res = _df_sorted
    else:
        _df_res = _df_sorted.group_by_dynamic("timestamp", every=_every).agg(
            [pl.col(v).mean() for v in _vars]
        )
    _fig = go.Figure()
    for _i, _v in enumerate(_vars):
        if _v in _df_res.columns:
            _yaxis = "y1" if _i == 0 else "y2"
            _fig.add_trace(
                go.Scatter(
                    x=_df_res["timestamp"].to_list(),
                    y=_df_res[_v].to_list(),
                    name=_v,
                    mode="lines",
                    yaxis=_yaxis,
                )
            )
    _layout = dict(
        title=f"Séries temporelles — résolution {s2_resolution.value}",
        xaxis_title="Date",
        legend=dict(orientation="h"),
        height=500,
        yaxis=dict(title=_vars[0] if _vars else "Valeur"),
    )
    if len(_vars) > 1:
        _layout["yaxis2"] = dict(  # type: ignore[assignment]
            title=" / ".join(_vars[1:]),
            overlaying="y",
            side="right",
            showgrid=False,
        )
    _fig.update_layout(**_layout)
    mo.ui.plotly(_fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 — Scatter plot & Corrélations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _scatter_vars = [
        # --- Cibles ---
        "load",
        "load_forecast",
        "pv_total",
        "pv_central_valais",
        "pv_sion",
        "pv_sierre",
        "pv_remote",
        # --- Prévisions best run ---
        "pred_temperature",
        "pred_radiation",
        "pred_sunshine",
        "pred_precipitation",
        "pred_pressure",
        "pred_humidity",
        "pred_wind_speed",
        "pred_wind_dir",
        # --- Prévisions spread ---
        "pred_temperature_spread",
        "pred_radiation_spread",
        "pred_sunshine_spread",
        "pred_precipitation_spread",
        "pred_pressure_spread",
        # --- Méta ---
        "GLOB_n_runs",
        "GLOB_lead_time",
        "T_2M_n_runs",
        "T_2M_lead_time",
    ]
    s3_x = mo.ui.dropdown(
        options=_scatter_vars,
        value="pred_radiation",
        label="Variable X",
    )
    s3_y = mo.ui.dropdown(
        options=_scatter_vars,
        value="pv_total",
        label="Variable Y",
    )
    mo.hstack([s3_x, s3_y])
    return s3_x, s3_y


@app.cell(hide_code=True)
def _(df_filtered, go, mo, s3_x, s3_y, stats):
    _xv = s3_x.value
    _yv = s3_y.value
    _sub = df_filtered.select([c for c in [_xv, _yv] if c in df_filtered.columns]).drop_nulls()
    if _xv in _sub.columns and _yv in _sub.columns and _sub.shape[0] > 10:
        _x = _sub[_xv].to_numpy().astype(float)
        _y = _sub[_yv].to_numpy().astype(float)
        _r_p, _p_p = stats.pearsonr(_x, _y)
        _r_s, _p_s = stats.spearmanr(_x, _y)
        _fig = go.Figure(
            go.Scatter(
                x=_x.tolist(),
                y=_y.tolist(),
                mode="markers",
                marker=dict(size=4, opacity=0.5, color="teal"),
            )
        )
        _fig.update_layout(
            title=f"{_yv} vs {_xv} — r={_r_p:.3f}",
            xaxis_title=_xv,
            yaxis_title=_yv,
            height=480,
        )
        _output = mo.vstack(
            [
                mo.ui.plotly(_fig),
                mo.md(
                    f"**Pearson r** = `{_r_p:.4f}` (p = {_p_p:.2e}) | "
                    f"**Spearman r** = `{_r_s:.4f}` (p = {_p_s:.2e})"
                ),
            ]
        )
    else:
        _output = mo.md("Données insuffisantes pour le scatter plot.")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4 — Heatmap de corrélation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _all_numeric = [
        # --- Cibles ---
        "load",
        "load_forecast",
        "pv_total",
        # --- Prévisions best run ---
        "pred_temperature",
        "pred_radiation",
        "pred_sunshine",
        "pred_precipitation",
        "pred_pressure",
        "pred_humidity",
        "pred_wind_speed",
        # --- Prévisions spread ---
        "pred_temperature_spread",
        "pred_radiation_spread",
        "pred_sunshine_spread",
        "pred_precipitation_spread",
        # --- Méta ---
        "GLOB_n_runs",
        "GLOB_lead_time",
        # --- Temporel ---
        "hour",
        "month",
    ]
    s4_vars = mo.ui.multiselect(
        options=_all_numeric,
        value=[
            "load",
            "pv_total",
            "pred_temperature",
            "pred_radiation",
            "pred_pressure",
            "pred_humidity",
            "pred_wind_speed",
            "pred_radiation_spread",
            "hour",
        ],
        label="Variables pour heatmap",
    )
    s4_vars
    return (s4_vars,)


@app.cell(hide_code=True)
def _(df_filtered, go, mo, np, s4_vars):
    _vars = [v for v in s4_vars.value if v in df_filtered.columns]
    if len(_vars) >= 2:
        _sub = df_filtered.select(_vars).drop_nulls()
        _mat = _sub.to_numpy().astype(float)
        _n = len(_vars)
        _corr = np.corrcoef(_mat, rowvar=False)
        _text = [[f"{_corr[i][j]:.2f}" for j in range(_n)] for i in range(_n)]
        _fig = go.Figure(
            go.Heatmap(
                z=_corr.tolist(),
                x=_vars,
                y=_vars,
                colorscale="RdBu",
                zmid=0,
                text=_text,
                texttemplate="%{text}",
            )
        )
        _fig.update_layout(title="Matrice de corrélation de Pearson", height=550)
        _output = mo.ui.plotly(_fig)
    else:
        _output = mo.md("Sélectionnez au moins 2 variables.")
    _output
    return


if __name__ == "__main__":
    app.run()
