"""XGBoost — Prédiction de charge horaire Oiken."""

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full", app_title="XGBoost — Prédiction de charge Oiken")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return go, mean_absolute_error, mean_squared_error, np, pd, r2_score, xgb


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # **XGBoost — Prédiction de charge horaire**
    ## Données : `features_v2.csv` (oct 2022 – sept 2025)
    19 features + target `load` (z-score) + baseline `load_forecast` (Oiken)
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo, pd):
    df = pd.read_csv("data/features_v2.csv", parse_dates=[0], index_col=0)
    _n_before = len(df)
    df = df.dropna(subset=["load", "load_J1_same_hour", "load_J1_mean", "load_J7_mean"])
    df["clear_sky_ratio"] = df["clear_sky_ratio"].clip(upper=1.0)
    _n_dropped = _n_before - len(df)
    feature_cols = [c for c in df.columns if c not in ["load", "load_forecast"]]
    mo.md(f"""
### Chargement

| | |
|---|---|
| Lignes chargées | **{_n_before}** |
| Lignes droppées (NaN) | {_n_dropped} |
| Lignes restantes | **{len(df)}** |
| Période | {df.index.min().date()} → {df.index.max().date()} |
| `load_forecast` présent | **Oui** (baseline Oiken) |
""")
    return df, feature_cols


@app.cell(hide_code=True)
def _(df, feature_cols, mo):
    train_end = "2024-10-07"
    train = df[df.index < train_end].copy()
    test = df[df.index >= train_end].copy()
    X_train = train[feature_cols]
    y_train = train["load"]
    X_test = test[feature_cols]
    y_test = test["load"]
    mo.md(f"""
## 1 — Split train / test chronologique

| Split | Lignes | De | À |
|---|---|---|---|
| **Train** | {len(X_train)} | {train.index.min().date()} | {train.index.max().date()} |
| **Test** | {len(X_test)} | {test.index.min().date()} | {test.index.max().date()} |

Ratio train/test : **{len(X_train) / len(X_test):.2f}**
""")
    return X_test, X_train, test, train, y_test, y_train


@app.cell(hide_code=True)
def _(X_test, X_train, mo, xgb, y_test, y_train):
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    mo.md(f"""
## 2 — Entraînement XGBoost

Meilleur itération : **{model.best_iteration}** / 500
""")
    return (model,)


@app.cell(hide_code=True)
def _(
    X_test,
    X_train,
    mean_absolute_error,
    mean_squared_error,
    mo,
    model,
    np,
    r2_score,
    y_test,
    y_train,
):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    _mae_tr = mean_absolute_error(y_train, y_pred_train)
    _rmse_tr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    _r2_tr = r2_score(y_train, y_pred_train)
    _mae_te = mean_absolute_error(y_test, y_pred_test)
    _rmse_te = np.sqrt(mean_squared_error(y_test, y_pred_test))
    _r2_te = r2_score(y_test, y_pred_test)
    _ratio = _rmse_te / _rmse_tr
    _status = "OK" if _ratio < 1.5 else "Overfitting possible"
    mo.md(f"""
## 3 — Métriques XGBoost

| Métrique | Train | Test |
|---|---|---|
| MAE | {_mae_tr:.4f} | {_mae_te:.4f} |
| RMSE | {_rmse_tr:.4f} | {_rmse_te:.4f} |
| R² | {_r2_tr:.4f} | {_r2_te:.4f} |

Ratio RMSE test/train : **{_ratio:.2f}** — {_status}
""")
    return y_pred_test, y_pred_train


@app.cell(hide_code=True)
def _(feature_cols, go, mo, model, pd):
    _importance = model.feature_importances_
    _feat_imp = pd.DataFrame({"feature": feature_cols, "importance": _importance}).sort_values(
        "importance", ascending=True
    )
    _fig = go.Figure(
        go.Bar(
            x=_feat_imp["importance"].tolist(),
            y=_feat_imp["feature"].tolist(),
            orientation="h",
            marker_color=[
                "#2196F3" if imp >= sorted(_importance)[-5] else "#BBDEFB"
                for imp in _feat_imp["importance"]
            ],
        )
    )
    _fig.update_layout(
        title="Feature Importance — XGBoost",
        xaxis_title="Importance",
        height=550,
        margin=dict(l=200),
    )
    _top = _feat_imp.sort_values("importance", ascending=False).head(10)
    _rows = "\n".join(f"| {r['feature']} | {r['importance']:.4f} |" for _, r in _top.iterrows())
    mo.vstack(
        [
            mo.md(f"## 4 — Feature Importance\n\n| Feature | Importance |\n|---|---|\n{_rows}"),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, mean_absolute_error, mean_squared_error, mo, np, r2_score, test, y_pred_test, y_test):
    _oiken = test["load_forecast"].values
    _persist = test["load_J1_same_hour"].values
    _persist7 = test["load_J7_mean"].values
    _real = y_test.values
    _xgb = y_pred_test

    def _m(y_true, y_pred):
        _mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt, yp = y_true[_mask], y_pred[_mask]
        return (mean_absolute_error(yt, yp), np.sqrt(mean_squared_error(yt, yp)), r2_score(yt, yp))

    _models = {
        "XGBoost": _m(_real, _xgb),
        "Oiken (load_forecast)": _m(_real, _oiken),
        "Persistence J-1 (same hour)": _m(_real, _persist),
        "Moyenne J-7": _m(_real, _persist7),
    }

    _rows = []
    for name, (mae, rmse, r2) in _models.items():
        _rows.append(f"| {name} | {mae:.4f} | {rmse:.4f} | {r2:.4f} |")
    _table = "\n".join(_rows)

    # Bar chart comparatif
    _names = list(_models.keys())
    _maes = [v[0] for v in _models.values()]
    _rmses = [v[1] for v in _models.values()]
    _colors = ["#2196F3", "#FF9800", "#9E9E9E", "#9E9E9E"]

    _fig = go.Figure()
    _fig.add_trace(go.Bar(name="MAE", x=_names, y=_maes, marker_color=_colors))
    _fig.add_trace(
        go.Bar(
            name="RMSE",
            x=_names,
            y=_rmses,
            marker_color=[c.replace("F3", "64").replace("00", "40") for c in _colors],
        )
    )
    _fig.update_layout(
        title="Benchmark — MAE & RMSE sur le test set (oct 2024 → sept 2025)",
        barmode="group",
        height=450,
        yaxis_title="Erreur (z-score)",
        legend=dict(orientation="h"),
    )

    mo.vstack(
        [
            mo.md(f"""
## 5 — Benchmark : XGBoost vs Oiken vs Baselines

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
{_table}
"""),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(feature_cols, go, mo, model, test):
    _mask = (test.index >= "2025-01-13") & (test.index < "2025-01-20")
    _week = test[_mask].copy()
    _week_pred = model.predict(_week[feature_cols])
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_week.index.tolist(),
            y=_week["load"].tolist(),
            name="Réel",
            mode="lines",
            line=dict(color="#1565C0", width=2),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_week.index.tolist(),
            y=_week_pred.tolist(),
            name="XGBoost",
            mode="lines",
            line=dict(color="#F57C00", width=2, dash="dash"),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_week.index.tolist(),
            y=_week["load_forecast"].tolist(),
            name="Oiken",
            mode="lines",
            line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        )
    )
    _fig.update_layout(
        title="6 — Prédiction vs Réel — jan 2025 (hiver)",
        yaxis_title="Load (z-score)",
        height=450,
        legend=dict(orientation="h"),
    )
    mo.vstack(
        [
            mo.md("## 6 — Semaine hiver : XGBoost vs Oiken vs Réel"),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(feature_cols, go, mo, model, test):
    _mask = (test.index >= "2025-07-14") & (test.index < "2025-07-21")
    _week = test[_mask].copy()
    _week_pred = model.predict(_week[feature_cols])
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=_week.index.tolist(),
            y=_week["load"].tolist(),
            name="Réel",
            mode="lines",
            line=dict(color="#1565C0", width=2),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_week.index.tolist(),
            y=_week_pred.tolist(),
            name="XGBoost",
            mode="lines",
            line=dict(color="#F57C00", width=2, dash="dash"),
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=_week.index.tolist(),
            y=_week["load_forecast"].tolist(),
            name="Oiken",
            mode="lines",
            line=dict(color="#9E9E9E", width=1.5, dash="dot"),
        )
    )
    _fig.update_layout(
        title="7 — Prédiction vs Réel — jul 2025 (été)",
        yaxis_title="Load (z-score)",
        height=450,
        legend=dict(orientation="h"),
    )
    mo.vstack(
        [
            mo.md("## 7 — Semaine été : XGBoost vs Oiken vs Réel"),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, mo, y_pred_test, y_test):
    _fig = go.Figure()
    _fig.add_trace(
        go.Scatter(
            x=y_test.tolist(),
            y=y_pred_test.tolist(),
            mode="markers",
            marker=dict(size=3, opacity=0.15, color="#1565C0"),
            name="Points",
        )
    )
    _fig.add_trace(
        go.Scatter(
            x=[-3.5, 3.5],
            y=[-3.5, 3.5],
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="y = x",
        )
    )
    _fig.update_layout(
        title="8 — Scatter — Test set",
        xaxis_title="Load réel",
        yaxis_title="Load prédit",
        height=550,
        width=600,
    )
    mo.vstack(
        [
            mo.md("## 8 — Scatter : prédit vs réel (test set)"),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(go, mo, np, y_pred_test, y_test):
    _errors = y_test.values - y_pred_test
    _fig = go.Figure(
        go.Histogram(
            x=_errors.tolist(),
            nbinsx=80,
            marker_color="#42A5F5",
            marker_line_color="black",
            marker_line_width=0.5,
        )
    )
    _fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=2)
    _fig.add_vline(
        x=float(_errors.mean()),
        line_dash="solid",
        line_color="orange",
        line_width=2,
        annotation_text=f"mean={_errors.mean():.4f}",
    )
    _fig.update_layout(
        title=f"9 — Distribution des erreurs — mean={_errors.mean():.4f}, std={_errors.std():.4f}",
        xaxis_title="Erreur (réel - prédit)",
        height=400,
    )
    mo.vstack(
        [
            mo.md("## 9 — Distribution des erreurs (test set)"),
            mo.ui.plotly(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def _(feature_cols, go, mean_absolute_error, mean_squared_error, mo, model, np, test):
    _eval = test.copy()
    _eval["xgb_pred"] = model.predict(test[feature_cols])
    _eval["month"] = _eval.index.month
    _mois = {
        1: "Jan",
        2: "Fév",
        3: "Mar",
        4: "Avr",
        5: "Mai",
        6: "Jun",
        7: "Jul",
        8: "Aoû",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Déc",
    }

    _months_sorted = sorted(_eval["month"].unique())
    _xgb_maes, _oiken_maes, _labels = [], [], []
    _rows = []
    for _m in _months_sorted:
        _s = _eval[_eval["month"] == _m]
        _mae_xgb = mean_absolute_error(_s["load"], _s["xgb_pred"])
        _rmse_xgb = np.sqrt(mean_squared_error(_s["load"], _s["xgb_pred"]))
        _mae_oik = mean_absolute_error(_s["load"], _s["load_forecast"])
        _rmse_oik = np.sqrt(mean_squared_error(_s["load"], _s["load_forecast"]))
        _gain = (1 - _mae_xgb / _mae_oik) * 100
        _label = _mois.get(_m, str(_m))
        _labels.append(_label)
        _xgb_maes.append(_mae_xgb)
        _oiken_maes.append(_mae_oik)
        _win = "**" if _mae_xgb < _mae_oik else ""
        _rows.append(
            f"| {_label} | {_mae_xgb:.4f} | {_rmse_xgb:.4f} | {_mae_oik:.4f} | {_rmse_oik:.4f} | {_gain:+.1f}% | {len(_s)} |"
        )

    _table = "\n".join(_rows)

    _fig = go.Figure()
    _fig.add_trace(go.Bar(name="XGBoost", x=_labels, y=_xgb_maes, marker_color="#2196F3"))
    _fig.add_trace(go.Bar(name="Oiken", x=_labels, y=_oiken_maes, marker_color="#FF9800"))
    _fig.update_layout(
        title="MAE par mois — XGBoost vs Oiken",
        barmode="group",
        height=400,
        yaxis_title="MAE (z-score)",
        legend=dict(orientation="h"),
    )

    mo.vstack(
        [
            mo.md(f"""
## 10 — Performance par mois : XGBoost vs Oiken

| Mois | XGB MAE | XGB RMSE | Oiken MAE | Oiken RMSE | Gain XGB | n |
|---|---|---|---|---|---|---|
{_table}

_Gain positif = XGBoost meilleur qu'Oiken_
"""),
            mo.ui.plotly(_fig),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
