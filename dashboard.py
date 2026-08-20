"""Dash dashboard for electricity demand forecasting artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


ARTIFACT_ROOT = Path("outputs")
DATA_PATH = ARTIFACT_ROOT / "data" / "electricity_demand_dataset.csv"
METRICS_PATH = ARTIFACT_ROOT / "evaluation" / "model_comparison.csv"
INTERVAL_PATH = ARTIFACT_ROOT / "evaluation" / "prediction_intervals.csv"
ANOMALY_PATH = ARTIFACT_ROOT / "anomaly_testing" / "anomaly_injection_metrics.json"

app = Dash(__name__, title="BSES Demand Forecast Dashboard")


def load_csv(path: Path) -> pd.DataFrame:
    """Loads a CSV file or returns an empty dataframe."""
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_json(path: Path) -> dict[str, Any]:
    """Loads a JSON file or returns an empty dictionary."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def empty_figure(title: str) -> go.Figure:
    """Builds an empty Plotly figure with a clear title."""
    figure = go.Figure()
    figure.update_layout(title=title, template="plotly_white", height=360)
    return figure


def historical_figure() -> go.Figure:
    """Builds the historical demand trend chart."""
    data = load_csv(DATA_PATH)
    if data.empty:
        return empty_figure("Historical Demand Trends")
    data["datetime"] = pd.to_datetime(data["datetime"])
    figure = px.line(data, x="datetime", y="demand_kw", title="Historical Demand Trends")
    figure.update_layout(template="plotly_white", height=380, xaxis_rangeslider_visible=True)
    return figure


def forecast_figure() -> go.Figure:
    """Builds the live forecast visualization from prediction interval artifacts."""
    data = load_csv(INTERVAL_PATH)
    if data.empty:
        return empty_figure("Live Forecast")
    data["datetime"] = pd.to_datetime(data["datetime"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=data["datetime"], y=data["actual_kw"], name="Actual", mode="lines"))
    figure.add_trace(go.Scatter(x=data["datetime"], y=data["mean_prediction_kw"], name="Forecast", mode="lines"))
    figure.add_trace(
        go.Scatter(
            x=pd.concat([data["datetime"], data["datetime"].iloc[::-1]]),
            y=pd.concat([data["upper_95_kw"], data["lower_95_kw"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(35,100,170,0.18)",
            line={"color": "rgba(255,255,255,0)"},
            name="95% interval",
        )
    )
    figure.update_layout(title="Live Forecast", template="plotly_white", height=380)
    return figure


def metrics_panel() -> html.Div:
    """Builds the model performance metrics display."""
    metrics = load_csv(METRICS_PATH)
    if metrics.empty:
        return html.Div("Metrics unavailable", className="panel muted")
    best = metrics.sort_values("mae").iloc[0]
    items = [
        ("Best model", str(best["model"])),
        ("MAE", f"{best['mae']:,.2f}"),
        ("RMSE", f"{best['rmse']:,.2f}"),
        ("MAPE", f"{best['mape']:,.2f}%"),
        ("R2", f"{best['r2']:,.3f}"),
    ]
    return html.Div([html.Div([html.Div(label, className="metric-label"), html.Div(value, className="metric-value")], className="metric") for label, value in items], className="metrics-grid")


def anomaly_panel() -> html.Div:
    """Builds the anomaly alert panel."""
    payload = load_json(ANOMALY_PATH)
    if not payload:
        return html.Div("Anomaly test unavailable", className="panel muted")
    degradation = payload.get("mae_degradation", np.nan)
    status = "Elevated sensitivity" if degradation and degradation > 0 else "Stable"
    return html.Div(
        [
            html.Div(status, className="alert-title"),
            html.Div(f"Clean MAE: {payload.get('clean_mae', np.nan):,.2f}"),
            html.Div(f"Injected MAE: {payload.get('injected_mae', np.nan):,.2f}"),
            html.Div(f"MAE degradation: {degradation:,.2f}"),
        ],
        className="panel",
    )


def correlation_figure() -> go.Figure:
    """Builds the weather feature correlation chart."""
    data = load_csv(DATA_PATH)
    columns = ["demand_kw", "temperature_c", "humidity_pct", "wind_speed_kmh", "dew_point_c", "solar_irradiance"]
    if data.empty or any(column not in data.columns for column in columns):
        return empty_figure("Weather Feature Correlations")
    corr = data[columns].corr()
    figure = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Weather Feature Correlations")
    figure.update_layout(template="plotly_white", height=420)
    return figure


def build_layout() -> html.Div:
    """Builds the dashboard layout."""
    return html.Div(
        [
            dcc.Interval(id="refresh", interval=60_000, n_intervals=0),
            html.Div(
                [
                    html.H1("BSES Rajdhani Demand Forecasting"),
                    html.Div(id="metrics"),
                ],
                className="header",
            ),
            html.Div(
                [
                    html.Div(dcc.Graph(id="forecast-graph"), className="panel"),
                    html.Div(dcc.Graph(id="history-graph"), className="panel"),
                ],
                className="grid two",
            ),
            html.Div(
                [
                    html.Div(id="anomaly-alert"),
                    html.Div(dcc.Graph(id="correlation-graph"), className="panel"),
                ],
                className="grid two",
            ),
        ],
        className="page",
    )


app.layout = build_layout()


@app.callback(
    Output("forecast-graph", "figure"),
    Output("history-graph", "figure"),
    Output("metrics", "children"),
    Output("anomaly-alert", "children"),
    Output("correlation-graph", "figure"),
    Input("refresh", "n_intervals"),
)
def refresh_dashboard(_: int) -> tuple[go.Figure, go.Figure, html.Div, html.Div, go.Figure]:
    """Refreshes dashboard panels from saved pipeline artifacts."""
    return forecast_figure(), historical_figure(), metrics_panel(), anomaly_panel(), correlation_figure()


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; background: #f4f6f8; color: #1f2933; }
            .page { padding: 24px; }
            .header { display: flex; justify-content: space-between; align-items: center; gap: 24px; margin-bottom: 20px; }
            h1 { font-size: 28px; margin: 0; font-weight: 650; }
            .grid { display: grid; gap: 18px; margin-bottom: 18px; }
            .grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .panel { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px; min-height: 120px; box-shadow: 0 1px 2px rgba(16,24,40,0.04); }
            .muted { color: #64748b; display: flex; align-items: center; justify-content: center; }
            .metrics-grid { display: grid; grid-template-columns: repeat(5, minmax(110px, 1fr)); gap: 10px; }
            .metric { background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; padding: 10px 12px; }
            .metric-label { color: #64748b; font-size: 12px; }
            .metric-value { color: #102a43; font-size: 17px; font-weight: 650; margin-top: 4px; }
            .alert-title { font-weight: 700; margin-bottom: 8px; color: #9f1239; }
            @media (max-width: 960px) { .header { align-items: stretch; flex-direction: column; } .grid.two { grid-template-columns: 1fr; } .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=False)
