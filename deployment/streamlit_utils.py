"""
deployment/streamlit_utils.py
==============================
Reusable Streamlit UI building blocks: probability bar charts, activity
timeline charts, dark-theme CSS injection, and CSV/report export helpers.
Kept separate from ``app.py`` so the app file stays focused on layout/flow.
"""

from __future__ import annotations

import io
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DARK_THEME_CSS = """
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .metric-card {
        background-color: #1a1d29;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #2a2e3d;
    }
    .prediction-badge {
        font-size: 1.6rem;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        display: inline-block;
    }
    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
"""


def inject_dark_theme() -> None:
    """Inject the project's dark-theme CSS into the current Streamlit page."""
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)


def render_probability_bar_chart(probabilities: Dict[str, float]) -> None:
    """Render a horizontal bar chart of per-class prediction probabilities.

    Args:
        probabilities: Mapping of class name -> probability (0-1).
    """
    df = pd.DataFrame(
        {"Activity": list(probabilities.keys()), "Confidence": list(probabilities.values())}
    ).sort_values("Confidence", ascending=True)

    fig = px.bar(
        df, x="Confidence", y="Activity", orientation="h",
        color="Confidence", color_continuous_scale="Viridis",
        range_x=[0, 1], text_auto=".1%",
    )
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa", showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_activity_timeline(timeline: List[Dict]) -> None:
    """Render a step chart showing predicted activity over time for a
    video processed with a sliding window.

    Args:
        timeline: List of prediction dicts (each with ``activity``,
            ``confidence``, ``start_frame``), as returned by
            ``ActivityPredictor.predict_video_timeline``.
    """
    if not timeline:
        st.info("No timeline data available.")
        return

    df = pd.DataFrame(timeline)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["start_frame"], y=df["activity"], mode="lines+markers",
            line=dict(shape="hv", color="#8b5cf6"),
            marker=dict(size=8, color=df["confidence"], colorscale="Viridis", showscale=True),
            text=[f"{c:.1%}" for c in df["confidence"]],
            hovertemplate="Frame %{x}<br>Activity: %{y}<br>Confidence: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#fafafa", xaxis_title="Frame", yaxis_title="Activity",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_metric_cards(metrics: Dict[str, float]) -> None:
    """Render a row of metric cards (e.g. accuracy, precision, recall, F1).

    Args:
        metrics: Mapping of metric name -> value (0-1 fractions are shown
            as percentages).
    """
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        with col:
            st.metric(label=name, value=f"{value:.1%}" if value <= 1 else f"{value:.2f}")


def predictions_to_csv(history: List[Dict]) -> bytes:
    """Convert a list of prediction records into downloadable CSV bytes.

    Args:
        history: List of dicts, each typically containing ``timestamp``,
            ``activity``, and ``confidence``.

    Returns:
        UTF-8 encoded CSV bytes suitable for ``st.download_button``.
    """
    df = pd.DataFrame(history)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def build_text_report(
    activity: str, confidence: float, probabilities: Dict[str, float], extra: Dict | None = None
) -> str:
    """Build a plain-text summary report for a single prediction, suitable
    for a download button.

    Args:
        activity: Predicted activity label.
        confidence: Confidence of the top prediction.
        probabilities: Full per-class probability mapping.
        extra: Optional extra key/value pairs to append (e.g. inference
            time, video filename).

    Returns:
        Formatted multi-line report string.
    """
    lines = [
        "Human Activity Recognition — Prediction Report",
        "=" * 50,
        f"Predicted Activity : {activity}",
        f"Confidence          : {confidence:.2%}",
        "",
        "Class Probabilities:",
    ]
    for cls, prob in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {cls:<15} {prob:.2%}")

    if extra:
        lines.append("")
        lines.append("Additional Info:")
        for k, v in extra.items():
            lines.append(f"  {k}: {v}")

    return "\n".join(lines)
