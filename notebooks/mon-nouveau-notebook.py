"""EDA notebook for Oiken & Météo Sion data."""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full", app_title="EDA — Oiken & Météo Sion")


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
    from plotly.subplots import make_subplots

    return go, make_subplots, np, pl, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **Features Engineering**
    ## Data : Oiken & Météo Sion
     données horaires Oct 2022 – Mar 2026
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
            pl.when(pl.col("timestamp").dt.month().is_in([6, 7, 8]))
            .then(pl.lit("été"))
            .when(pl.col("timestamp").dt.month().is_in([12, 1, 2]))
            .then(pl.lit("hiver"))
            .otherwise(pl.lit("mi-saison"))
            .alias("season"),
            (pl.col("radiation").fill_null(0.0) > 0).alias("is_day"),
        ]
    )
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1 — Exploration data
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
    mo.vstack(
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
        "load",
        "load_forecast",
        "pv_total",
        "pv_central_valais",
        "pv_sion",
        "pv_sierre",
        "pv_remote",
        "temperature",
        "pressure",
        "radiation",
        "precipitation",
        "sunshine",
        # predictions
        "pred_temperature",
        "pred_radiation",
        "pred_precipitation",
        "pred_sunshine",
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
    mo.md(r"""
    ## Scatter plot
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _numeric_vars = [
        "load",
        "load_forecast",
        "pv_total",
        "pv_central_valais",
        "pv_sion",
        "pv_sierre",
        "pv_remote",
        "temperature",
        "pressure",
        "radiation",
        "precipitation",
        "sunshine",
    ]
    s2_scatter_x = mo.ui.dropdown(
        options=_numeric_vars,
        value="temperature",
        label="Variable X (abscisse)",
    )
    s2_scatter_y = mo.ui.dropdown(
        options=_numeric_vars,
        value="load",
        label="Variable Y (ordonnée)",
    )
    mo.hstack([s2_scatter_x, s2_scatter_y])
    return s2_scatter_x, s2_scatter_y


@app.cell(hide_code=True)
def _(df_filtered, go, mo, np, s2_scatter_x, s2_scatter_y):
    _x_var = s2_scatter_x.value
    _y_var = s2_scatter_y.value
    _df = df_filtered.drop_nulls([_x_var, _y_var])
    _corr = np.corrcoef(_df[_x_var].to_numpy(), _df[_y_var].to_numpy())[0, 1]
    _fig_scatter = go.Figure(
        go.Scatter(
            x=_df[_x_var].to_list(),
            y=_df[_y_var].to_list(),
            mode="markers",
            marker=dict(size=3, opacity=0.5),
            name=f"{_x_var} vs {_y_var}",
        )
    )
    _fig_scatter.update_layout(
        title=f"Nuage de points — {_x_var} vs {_y_var}  (r = {_corr:.3f})",
        xaxis_title=_x_var,
        yaxis_title=_y_var,
        height=450,
    )
    mo.ui.plotly(_fig_scatter)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3 — Analyse des corrélations
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _all_numeric = [
        "load",
        "load_forecast",
        "pv_total",
        "temperature",
        "pressure",
        "radiation",
        "precipitation",
        "sunshine",
        "pred_temperature",
        "pred_radiation",
        "pred_precipitation",
        "pred_sunshine",
        "hour",
    ]
    s3_heatmap_vars = mo.ui.multiselect(
        options=_all_numeric,
        value=["load", "temperature", "radiation", "pv_total", "pressure"],
        label="Variables pour heatmap de corrélation",
    )
    s3_x = mo.ui.dropdown(
        options=_all_numeric,
        value="temperature",
        label="Axe X (scatter)",
    )
    s3_y = mo.ui.dropdown(
        options=_all_numeric,
        value="load",
        label="Axe Y (scatter)",
    )
    s3_color = mo.ui.dropdown(
        options=["hour", "season", "is_day"],
        value="hour",
        label="Couleur (scatter)",
    )
    mo.vstack(
        [
            s3_heatmap_vars,
            mo.hstack([s3_x, s3_y, s3_color]),
        ]
    )
    return s3_color, s3_heatmap_vars, s3_x, s3_y


@app.cell(hide_code=True)
def _(df_filtered, go, mo, np, s3_heatmap_vars):
    _vars = [v for v in s3_heatmap_vars.value if v in df_filtered.columns]
    if len(_vars) >= 2:
        _sub = df_filtered.select(_vars).drop_nulls()
        _mat = _sub.to_numpy().astype(float)
        _n = len(_vars)
        _corr = np.corrcoef(_mat, rowvar=False)
        _text = [[f"{_corr[i][j]:.2f}" for j in range(_n)] for i in range(_n)]
        _fig_heat = go.Figure(
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
        _fig_heat.update_layout(title="Matrice de corrélation de Pearson", height=500)
        mo.ui.plotly(_fig_heat)
    else:
        mo.md("Sélectionnez au moins 2 variables.")
    return


@app.cell(hide_code=True)
def _(df_filtered, go, mo, s3_color, s3_x, s3_y, stats):
    _xv = s3_x.value
    _yv = s3_y.value
    _cv = s3_color.value
    _avail_cols = [c for c in [_xv, _yv, _cv] if c in df_filtered.columns]
    _sub_scat = df_filtered.select(_avail_cols).drop_nulls()
    if _xv in _sub_scat.columns and _yv in _sub_scat.columns and _sub_scat.shape[0] > 10:
        _x = _sub_scat[_xv].to_numpy().astype(float)
        _y = _sub_scat[_yv].to_numpy().astype(float)
        _r_p, _p_p = stats.pearsonr(_x, _y)
        _r_s, _p_s = stats.spearmanr(_x, _y)
        _c_raw = _sub_scat[_cv].to_list() if _cv in _sub_scat.columns else None
        if _cv == "season":
            _smap = {"été": 0, "mi-saison": 1, "hiver": 2}
            _c_vals = [_smap.get(str(v), 0) for v in (_c_raw or [])]
        elif _cv == "is_day":
            _c_vals = [1 if v else 0 for v in (_c_raw or [])]
        else:
            _c_vals = _c_raw
        _fig_scat = go.Figure(
            go.Scatter(
                x=_x.tolist(),
                y=_y.tolist(),
                mode="markers",
                marker=dict(
                    color=_c_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=_cv),
                    size=4,
                    opacity=0.5,
                ),
            )
        )
        _fig_scat.update_layout(
            title=f"{_yv} vs {_xv} — r_Pearson={_r_p:.3f} | r_Spearman={_r_s:.3f}",
            xaxis_title=_xv,
            yaxis_title=_yv,
            height=480,
        )
        mo.vstack(
            [
                mo.ui.plotly(_fig_scat),
                mo.md(
                    f"**Pearson r** = `{_r_p:.4f}` (p = {_p_p:.2e}) | "
                    f"**Spearman r** = `{_r_s:.4f}` (p = {_p_s:.2e})"
                ),
            ]
        )
    else:
        mo.md("Données insuffisantes pour le scatter plot.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ##4 — Analyse par période
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    s4_weekend = mo.ui.switch(label="Weekend uniquement", value=False)
    s4_season = mo.ui.dropdown(
        options=["Toutes", "été", "hiver", "mi-saison"],
        value="Toutes",
        label="Saison",
    )
    mo.hstack([s4_weekend, s4_season])
    return s4_season, s4_weekend


@app.cell(hide_code=True)
def _(df_filtered, go, mo, pl, s4_season, s4_weekend):
    _df4 = df_filtered
    if s4_weekend.value:
        _df4 = _df4.filter(pl.col("weekday") >= 5)
    else:
        _df4 = _df4.filter(pl.col("weekday") < 5)
    if s4_season.value != "Toutes":
        _df4 = _df4.filter(pl.col("season") == s4_season.value)
    _profile = _df4.group_by(["hour", "season"]).agg(pl.col("load").mean()).sort("hour")
    _fig4b = go.Figure()
    for _s in sorted(_profile["season"].unique().to_list()):
        _sub4 = _profile.filter(pl.col("season") == _s).sort("hour")
        _fig4b.add_trace(
            go.Scatter(
                x=_sub4["hour"].to_list(),
                y=_sub4["load"].to_list(),
                name=_s,
                mode="lines+markers",
            )
        )
    _fig4b.update_layout(
        title="Profil journalier moyen par saison",
        xaxis_title="Heure",
        yaxis_title="Charge moyenne",
        height=450,
    )
    mo.ui.plotly(_fig4b)
    return


@app.cell(hide_code=True)
def _(df_filtered, go, make_subplots, mo, pl):
    _df_day = df_filtered.filter(pl.col("is_day"))
    _df_night = df_filtered.filter(~pl.col("is_day"))
    _fig4c = make_subplots(rows=1, cols=2, subplot_titles=["Charge (load)", "PV total"])
    for _cn, _ci in [("load", 1), ("pv_total", 2)]:
        _fig4c.add_trace(
            go.Box(
                y=_df_day[_cn].drop_nulls().to_list(),
                name="Jour",
                marker_color="gold",
                legendgroup="Jour",
                showlegend=(_ci == 1),
            ),
            row=1,
            col=_ci,
        )
        _fig4c.add_trace(
            go.Box(
                y=_df_night[_cn].drop_nulls().to_list(),
                name="Nuit",
                marker_color="navy",
                legendgroup="Nuit",
                showlegend=(_ci == 1),
            ),
            row=1,
            col=_ci,
        )
    _fig4c.update_layout(title="Jour vs Nuit — Charge et Production PV", height=450)
    mo.ui.plotly(_fig4c)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5 — Production solaire
    """)
    return


@app.cell(hide_code=True)
def _(df_filtered, go, make_subplots, mo, pl):
    _df6 = (
        df_filtered.sort("timestamp")
        .group_by_dynamic("timestamp", every="1d")
        .agg(
            [
                pl.col("pv_total").sum().alias("pv_total"),
                pl.col("radiation").mean().alias("radiation"),
            ]
        )
    )
    _fig6a = make_subplots(specs=[[{"secondary_y": True}]])
    _fig6a.add_trace(
        go.Bar(
            x=_df6["timestamp"].to_list(),
            y=_df6["pv_total"].to_list(),
            name="PV total",
            marker_color="gold",
            opacity=0.75,
        ),
        secondary_y=False,
    )
    _fig6a.add_trace(
        go.Scatter(
            x=_df6["timestamp"].to_list(),
            y=_df6["radiation"].to_list(),
            name="Rayonnement moy.",
            line=dict(color="crimson", width=1.5),
        ),
        secondary_y=True,
    )
    _fig6a.update_yaxes(title_text="PV total (somme horaire)", secondary_y=False)
    _fig6a.update_yaxes(title_text="Rayonnement moyen (W/m²)", secondary_y=True)
    _fig6a.update_layout(
        title="Production PV journalière vs Rayonnement solaire",
        legend=dict(orientation="h"),
        height=500,
    )
    mo.ui.plotly(_fig6a)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ##7 — Détection d'anomalies
    """)
    return


@app.cell(hide_code=True)
def _(df_filtered, mo, pl):
    _numeric_types = [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.UInt32]
    _skip = {"hour", "weekday"}
    _num_cols = [
        c for c in df_filtered.columns if df_filtered[c].dtype in _numeric_types and c not in _skip
    ]
    s7_var = mo.ui.dropdown(
        options=_num_cols,
        value="load" if "load" in _num_cols else _num_cols[0],
        label="Variable à analyser",
    )
    s7_var
    return (s7_var,)


@app.cell(hide_code=True)
def _(df_filtered, go, make_subplots, mo, s7_var):
    _v7 = s7_var.value
    _vals7 = df_filtered[_v7].drop_nulls().to_list()
    _fig7b = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"Histogramme — {_v7}", f"Boîte à moustaches — {_v7}"],
    )
    _fig7b.add_trace(
        go.Histogram(x=_vals7, name=_v7, marker_color="teal", nbinsx=80),
        row=1,
        col=1,
    )
    _fig7b.add_trace(
        go.Box(y=_vals7, name=_v7, marker_color="teal", boxmean=True),
        row=1,
        col=2,
    )
    _fig7b.update_layout(
        title=f"Distribution de « {_v7} »",
        height=450,
        showlegend=False,
    )
    mo.ui.plotly(_fig7b)
    return


if __name__ == "__main__":
    app.run()
