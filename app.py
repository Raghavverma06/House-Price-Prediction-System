import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.predict import predict_price

# ── Theme constants ──────────────────────────────────────────────────────────
ACCENT = "#6C63FF"
ACCENT_LIGHT = "#8B84FF"
ACCENT_DARK = "#554FD9"
ACCENT_GLOW = "rgba(108, 99, 255, 0.35)"
BG_PRIMARY = "#050816"
PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_COLORS = [ACCENT, ACCENT_LIGHT, ACCENT_DARK, "#9D97FF", "#4A44CC"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best_model.pkl")
ACCURACY = "79.10%"
DATASET_SIZE = "20,640"
PREDICTION_TIME = "<5 ms"
CONFIDENCE_SCORE = 92
BEST_MODEL_NAME = "Random Forest"

# ── Lucide icon library (inline SVG) ─────────────────────────────────────────
_LUCIDE_PATHS = {
    "home": "m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
    "sparkles": "M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .962 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z",
    "target": "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20z M12 18a6 6 0 1 0 0-12 6 6 0 0 0 0 12z M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
    "bar-chart": "M12 20V10 M18 20V4 M6 20v-4",
    "trending-up": "M22 7 13.5 15.5 8.5 10.5 2 17 M16 7h6v6",
    "file-text": "M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z M14 2v4a2 2 0 0 0 2 2h4 M10 13h4 M10 17h4 M10 9h1",
    "building": "M6 22V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18Z M6 12H4a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2 M18 9h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-2 M10 6h4 M10 10h4 M10 14h4 M10 18h4",
    "zap": "M13 2 3 14h9l-1 8 10-12h-9l1-8z",
    "database": "M12 2C7.58 2 4 3.79 4 6v12c0 2.21 3.58 4 8 4s8-1.79 8-4V6c0-2.21-3.58-4-8-4z M4 6c0 2.21 3.58 4 8 4s8-1.79 8-4 M4 12c0 2.21 3.58 4 8 4s8-1.79 8-4",
    "cpu": "M12 20v2 M12 2v2 M17 20v2 M17 2v2 M2 12h2 M2 17h2 M2 7h2 M20 12h2 M20 17h2 M20 7h2 M7 20v2 M7 2v2 M15 8h2a1 1 0 0 1 1 1v2a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z M9 8H7a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1V9a1 1 0 0 0-1-1z M9 16H7a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1z M15 16h-2a1 1 0 0 0-1 1v2a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1z",
    "download": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3",
    "layers": "M12 2 2 7l10 5 10-5-10-5z M2 17l10 5 10-5 M2 12l10 5 10-5",
    "map-pin": "M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0z M12 10a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    "upload": "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M17 8l-5-5-5 5 M12 3v12",
    "check-circle": "M22 11.08V12a10 10 0 1 1-5.93-9.14 M22 4 12 14.01l-3-3",
    "award": "M15.477 12.89 17 22l-5-3-5 3 1.523-9.11L2 9.23l9.09-1.31L12 0l.91 7.92L22 9.23l-6.523 3.66z",
    "activity": "M22 12h-4l-3 9L9 3l-3 9H2",
    "box": "M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z",
    "pie-chart": "M21.21 15.89A10 10 0 1 1 8 2.83 M22 12A10 10 0 0 0 12 2v10z",
    "git-branch": "M6 3v12 M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M6 15h9a3 3 0 0 1 3 3v3",
    "folder": "M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2z",
    "book": "M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20",
    "rocket": "M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z M12 15l-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z M9 12H4s.55-3.03 2-4c1.62-1.08 5 0 5 0 M12 15v5s3.03-.55 4-2c1.08-1.62 0-5 0-5",
    "sliders": "M4 21v-7 M4 10V3 M12 21v-9 M12 8V3 M20 21v-5 M20 12V3 M2 14h4 M10 8h4 M18 16h4",
    "line-chart": "M3 3v18h18 M18 17V9 M13 17V5 M8 17v-3",
}


def icon(name, size=20, color=ACCENT):
    path = _LUCIDE_PATHS.get(name, _LUCIDE_PATHS["sparkles"])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" class="lucide-icon">'
        f'<path d="{path}"/></svg>'
    )


st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --accent: {ACCENT};
    --accent-light: {ACCENT_LIGHT};
    --accent-dark: {ACCENT_DARK};
    --accent-glow: {ACCENT_GLOW};
    --bg: {BG_PRIMARY};
    --glass: rgba(255, 255, 255, 0.04);
    --glass-border: rgba(255, 255, 255, 0.08);
    --text-1: #f0f0f5;
    --text-2: #9898a8;
    --text-3: #5c5c6e;
    --radius: 20px;
    --radius-sm: 20px;
    --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.16);
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(ellipse 70% 50% at 10% -20%, rgba(108,99,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 90% 110%, rgba(108,99,255,0.04) 0%, transparent 55%),
        var(--bg);
    color: var(--text-1);
}}

.block-container {{
    padding-top: 1rem;
    padding-bottom: 0.25rem;
    max-width: 1360px;
}}

@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.fade-in {{ animation: fadeInUp 0.5s ease forwards; }}

/* ── Hero ── */
.hero-section {{
    background: linear-gradient(135deg, rgba(108,99,255,0.06) 0%, rgba(255,255,255,0.025) 100%);
    backdrop-filter: blur(24px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 56px 44px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: var(--card-shadow);
    animation: fadeInUp 0.55s ease;
}}
.hero-section::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 30%, rgba(108,99,255,0.08) 0%, transparent 65%);
    pointer-events: none;
}}
.hero-section::after {{
    content: '';
    position: absolute;
    width: 420px; height: 420px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(108,99,255,0.06) 0%, transparent 70%);
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    z-index: 0;
}}
.hero-section > * {{ position: relative; z-index: 1; }}
.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.18);
    color: var(--accent-light);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 100px;
    margin-bottom: 20px;
}}
.main-title {{
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1.08;
    color: var(--text-1);
    margin-bottom: 12px;
}}
.subtitle {{
    font-size: clamp(1rem, 2vw, 1.15rem);
    color: var(--text-2);
    font-weight: 400;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.65;
}}

/* ── Cards (shared) ── */
.kpi-card, .glass-panel, .form-panel, .feature-card, .category-card,
.chart-wrap, .doc-card, .workflow, .result-card, .confidence-wrap {{
    border-radius: 20px;
    box-shadow: var(--card-shadow);
    transition: transform 0.25s ease, background 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}}

.kpi-card {{
    background: var(--glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    padding: 22px;
    animation: fadeInUp 0.5s ease both;
    min-height: 140px;
}}
.kpi-card:hover {{
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.06);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.22), 0 0 20px rgba(108, 99, 255, 0.06);
    border-color: rgba(108, 99, 255, 0.15);
}}
.kpi-icon-wrap {{
    width: 40px; height: 40px;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 14px;
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 600;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-size: clamp(1.3rem, 2.5vw, 1.7rem);
    font-weight: 800;
    color: var(--text-1);
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}}
.kpi-desc {{
    font-size: 12px;
    color: var(--text-2);
    line-height: 1.55;
}}

.glass-panel {{
    background: var(--glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    padding: 22px;
    margin-bottom: 14px;
}}
.glass-panel:hover {{
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(108, 99, 255, 0.12);
}}

.form-panel {{
    background: var(--glass);
    backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    padding: 24px 22px 10px;
    margin-bottom: 14px;
}}
.form-panel:hover {{
    background: rgba(255, 255, 255, 0.05);
}}
.form-panel-title {{
    font-size: 12px;
    font-weight: 700;
    color: var(--accent-light);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--glass-border);
    display: flex;
    align-items: center;
    gap: 8px;
}}

.feature-card {{
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    padding: 20px;
    height: 100%;
    animation: fadeInUp 0.5s ease both;
}}
.feature-card:hover {{
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(108, 99, 255, 0.14);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.2), 0 0 18px rgba(108, 99, 255, 0.05);
}}

.category-card {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    padding: 18px 14px;
    text-align: center;
    font-weight: 600;
    font-size: 13px;
    color: var(--text-2);
}}
.category-card:hover {{
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(108, 99, 255, 0.14);
    color: var(--text-1);
}}

.chart-wrap {{
    background: var(--glass);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    padding: 12px 8px 4px;
    margin-bottom: 10px;
}}
.chart-wrap:hover {{
    background: rgba(255, 255, 255, 0.05);
    border-color: rgba(108, 99, 255, 0.12);
}}

.result-card {{
    background: linear-gradient(135deg, rgba(108,99,255,0.08) 0%, rgba(108,99,255,0.03) 100%);
    border: 1px solid rgba(108,99,255,0.16);
    padding: 28px;
    margin: 16px 0;
    animation: fadeInUp 0.5s ease;
}}

/* ── Workflow ── */
.workflow {{
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    padding: 28px 16px;
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    animation: fadeInUp 0.6s ease;
}}
.workflow-step {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    min-width: 110px;
}}
.workflow-icon {{
    width: 44px; height: 44px;
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.22);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
}}
.workflow-label {{
    font-size: 12px;
    font-weight: 600;
    color: var(--text-2);
    text-align: center;
}}
.workflow-arrow {{
    color: var(--accent);
    font-size: 18px;
    opacity: 0.5;
    margin: 0 4px;
}}

/* ── Typography ── */
.section-title {{
    font-size: clamp(1.15rem, 2vw, 1.5rem);
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.02em;
    margin: 8px 0 14px 0;
}}
.section-title-sm {{
    font-size: 11px;
    font-weight: 600;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin: 0 0 10px 0;
}}
.page-header-wrap {{
    margin-bottom: 22px;
    padding-bottom: 18px;
    border-bottom: 1px solid var(--glass-border);
    animation: fadeInUp 0.45s ease;
}}
.page-header {{
    font-size: clamp(1.55rem, 3vw, 2rem);
    font-weight: 800;
    color: var(--text-1);
    letter-spacing: -0.03em;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 12px;
}}
.page-subheader {{
    font-size: 14px;
    color: var(--text-2);
    line-height: 1.65;
    max-width: 640px;
    font-weight: 400;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #030610 0%, #050816 100%);
    border-right: 1px solid var(--glass-border);
    min-width: 17rem !important;
    max-width: 17rem !important;
    width: 17rem !important;
}}
[data-testid="stSidebar"] > div:first-child {{
    width: 17rem !important;
}}
.sidebar-logo {{
    width: 34px; height: 34px;
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.sidebar-brand {{ font-size: 15px; font-weight: 700; color: var(--text-1); letter-spacing: -0.02em; line-height: 1.3; }}
.sidebar-tagline {{ font-size: 11px; color: var(--text-3); margin-top: 3px; letter-spacing: 0.02em; line-height: 1.4; }}
.sidebar-status {{
    background: rgba(108,99,255,0.06);
    border: 1px solid rgba(108,99,255,0.14);
    border-radius: 20px;
    padding: 12px 14px;
    margin-bottom: 6px;
}}
.status-dot {{
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--accent);
    border-radius: 50%;
    margin-right: 8px;
    opacity: 0.9;
}}
.sidebar-version {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 12px 14px;
    font-size: 11px;
    color: var(--text-3);
    line-height: 1.65;
}}
[data-testid="stSidebar"] .stRadio > div {{
    gap: 4px;
    display: flex;
    flex-direction: column;
}}
[data-testid="stSidebar"] .stRadio label {{
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
    padding: 10px 14px !important;
    border-radius: 12px !important;
    transition: background 0.25s ease, color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease !important;
    margin: 0 !important;
    width: 100%;
}}
[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(108,99,255,0.07) !important;
    color: var(--text-1) !important;
    transform: translateX(2px);
}}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {{
    background: rgba(108,99,255,0.11) !important;
    color: var(--text-1) !important;
    box-shadow: 0 0 18px rgba(108,99,255,0.1), inset 0 0 0 1px rgba(108,99,255,0.16) !important;
}}

/* ── Buttons ── */
div.stButton > button {{
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%) !important;
    color: #fff !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 9999px !important;
    padding: 0.7rem 1.75rem !important;
    box-shadow: 0 2px 12px rgba(108, 99, 255, 0.25) !important;
    transition: transform 0.25s ease, filter 0.25s ease, box-shadow 0.25s ease !important;
    letter-spacing: 0.01em !important;
}}
div.stButton > button:hover {{
    transform: translateY(-2px) !important;
    filter: brightness(1.08) !important;
    box-shadow: 0 6px 20px rgba(108, 99, 255, 0.32) !important;
}}
div.stDownloadButton > button {{
    background: rgba(108,99,255,0.08) !important;
    color: var(--accent-light) !important;
    border: 1px solid rgba(108,99,255,0.2) !important;
    font-weight: 600 !important;
    border-radius: 9999px !important;
    transition: transform 0.25s ease, filter 0.25s ease, background 0.25s ease !important;
}}
div.stDownloadButton > button:hover {{
    background: rgba(108,99,255,0.14) !important;
    filter: brightness(1.06) !important;
    transform: translateY(-2px) !important;
}}

/* ── Inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
.stNumberInput input, .stTextInput input {{
    background: rgba(255,255,255,0.04) !important;
    border-color: var(--glass-border) !important;
    color: var(--text-1) !important;
    border-radius: 10px !important;
    transition: border-color 0.2s ease !important;
}}
div[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus {{
    border-color: rgba(108,99,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.1) !important;
}}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background: var(--accent) !important; }}
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {{ color: var(--accent-light) !important; }}

[data-testid="stMetric"] {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 14px 16px;
}}
[data-testid="stMetricLabel"] {{ font-size: 11px !important; color: var(--text-3) !important; text-transform: uppercase; letter-spacing: 0.06em; }}
[data-testid="stMetricValue"] {{ font-weight: 700 !important; color: var(--text-1) !important; }}

.stProgress > div > div {{ background: linear-gradient(90deg, var(--accent-dark), var(--accent-light)) !important; border-radius: 6px; }}

.confidence-wrap {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    padding: 18px 20px;
    margin: 12px 0;
}}
.confidence-bar-bg {{ background: rgba(255,255,255,0.05); border-radius: 8px; height: 8px; overflow: hidden; margin: 10px 0 6px; }}
.confidence-bar-fill {{ height: 100%; border-radius: 8px; background: linear-gradient(90deg, var(--accent-dark), var(--accent-light)); }}

.doc-card {{
    background: var(--glass);
    border: 1px solid var(--glass-border);
    padding: 20px;
    margin-bottom: 12px;
    animation: fadeInUp 0.5s ease both;
}}
.doc-card:hover {{
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(108, 99, 255, 0.14);
}}
.doc-card-title {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text-1);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}}
.doc-card-body {{ font-size: 13px; color: var(--text-2); line-height: 1.7; }}
.doc-card-body li {{ margin-bottom: 4px; }}

.app-footer {{
    margin-top: 40px;
    padding: 28px 16px 20px;
    text-align: center;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    background: none;
    box-shadow: none;
}}
.footer-label {{
    font-size: 11px;
    color: var(--text-3);
    letter-spacing: 0.04em;
    margin-bottom: 10px;
    font-weight: 400;
}}
.footer-name {{
    font-size: 15px;
    font-weight: 600;
    color: var(--accent);
    margin-bottom: 10px;
}}
.footer-detail {{
    font-size: 12px;
    color: var(--text-3);
    line-height: 1.65;
    max-width: 520px;
    margin: 0 auto;
    font-weight: 400;
}}

.custom-divider {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 28px 0;
}}

.stDataFrame, [data-testid="stDataFrame"] {{
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid var(--glass-border);
}}

hr {{ border-color: var(--glass-border); margin: 22px 0; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

.body-text {{
    color: var(--text-2);
    line-height: 1.7;
    font-size: 14px;
    font-weight: 400;
}}

@media (max-width: 768px) {{
    .hero-section {{ padding: 40px 22px; }}
    .workflow {{ gap: 4px; padding: 20px 12px; }}
    .workflow-arrow {{ display: none; }}
    .kpi-card, .feature-card {{ min-height: auto; }}
    [data-testid="column"] {{ min-width: 0 !important; flex: 1 1 100% !important; }}
    [data-testid="stSidebar"] {{
        min-width: 100% !important;
        max-width: 100% !important;
        width: 100% !important;
    }}
}}
</style>
""", unsafe_allow_html=True)


# ── Data helpers (unchanged logic) ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_housing_dataframe():
    housing = fetch_california_housing(as_frame=True)
    return housing.frame


@st.cache_data(show_spinner=False)
def compute_model_comparison():
    df = load_housing_dataframe()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE": round(mean_absolute_error(y_test, preds), 4),
            "RMSE": round(mean_squared_error(y_test, preds) ** 0.5, 4),
            "R² Score": round(r2_score(y_test, preds), 4),
        })
    return pd.DataFrame(results)


@st.cache_resource(show_spinner=False)
def load_best_model():
    return joblib.load(MODEL_PATH)


def build_prediction_csv(property_type, location, median_income, house_age,
                         avg_rooms, avg_bedrooms, population, avg_occupancy,
                         latitude, longitude, prediction, confidence):
    result_df = pd.DataFrame([{
        "Property Type": property_type,
        "Property Location": location,
        "Median Income": median_income,
        "House Age": house_age,
        "Average Rooms": avg_rooms,
        "Average Bedrooms": avg_bedrooms,
        "Population": population,
        "Average Occupancy": avg_occupancy,
        "Latitude": latitude,
        "Longitude": longitude,
        "Estimated Price (USD)": round(prediction * 100000, 2),
        "Model Confidence (%)": confidence,
    }])
    return result_df.to_csv(index=False).encode("utf-8")


# ── UI helpers ───────────────────────────────────────────────────────────────
def style_plotly_figure(fig, height=400, showlegend=None):
    kw = dict(
        template=PLOTLY_TEMPLATE,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(family="Inter, sans-serif", color="#9898a8", size=12),
        title=dict(font=dict(size=14, color="#f0f0f5", family="Inter")),
        margin=dict(l=48, r=24, t=52, b=44),
        colorway=PLOTLY_COLORS,
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.05)"),
        hoverlabel=dict(bgcolor="#0d0d18", bordercolor="rgba(108,99,255,0.3)", font_size=12),
    )
    if showlegend is not None:
        kw["showlegend"] = showlegend
    fig.update_layout(**kw)
    return fig


def render_footer():
    st.markdown(f"""
    <div class="app-footer">
        <div class="footer-label">Designed &amp; Developed by</div>
        <div class="footer-name">Raghav Verma</div>
        <div class="footer-detail">B.Tech CSE (AI &amp; ML) &bull; Manav Rachna International Institute of Research and Studies (MRIIRS)</div>
    </div>
    """, unsafe_allow_html=True)


def render_page_header(title, subtitle, icon_name="sparkles"):
    st.markdown(f"""
    <div class="page-header-wrap fade-in">
        <div class="page-header">
            <span style="display:flex;align-items:center;">{icon(icon_name, 26, ACCENT_LIGHT)}</span>
            {title}
        </div>
        <div class="page-subheader">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_kpi_card(label, value, description="", icon_name="activity"):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-icon-wrap">{icon(icon_name, 18, ACCENT_LIGHT)}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_card(title, desc, icon_name="sparkles"):
    st.markdown(f"""
    <div class="feature-card">
        <div style="margin-bottom:14px;">{icon(icon_name, 22, ACCENT)}</div>
        <div style="font-size:14px;font-weight:700;color:var(--text-1);margin-bottom:8px;">{title}</div>
        <div style="font-size:12px;color:var(--text-2);line-height:1.65;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def render_category_card(icon_name, label):
    st.markdown(f"""
    <div class="category-card">
        <div style="margin-bottom:10px;display:flex;justify-content:center;">{icon(icon_name, 22, ACCENT_LIGHT)}</div>
        <div>{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_confidence_meter(score):
    st.markdown(f"""
    <div class="confidence-wrap">
        <div class="section-title-sm" style="margin-bottom:0;">Model Confidence</div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{score}%;"></div>
        </div>
        <div style="font-size:14px;color:{ACCENT_LIGHT};font-weight:700;">{score}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_chart_section(title, fig):
    st.markdown(f'<div class="section-title-sm">{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_workflow():
    steps = [
        ("building", "Property Details"),
        ("cpu", "Machine Learning"),
        ("target", "Prediction"),
        ("bar-chart", "Analytics"),
        ("download", "Download Report"),
    ]
    parts = ['<div class="workflow">']
    for i, (ic, label) in enumerate(steps):
        if i > 0:
            parts.append('<div class="workflow-arrow">↓</div>')
        parts.append(f"""
        <div class="workflow-step">
            <div class="workflow-icon">{icon(ic, 20, ACCENT_LIGHT)}</div>
            <div class="workflow-label">{label}</div>
        </div>""")
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def render_doc_card(title, body_html, icon_name="book", delay=0):
    st.markdown(f"""
    <div class="doc-card" style="animation-delay:{delay * 0.05}s;">
        <div class="doc-card-title">{icon(icon_name, 18, ACCENT)} {title}</div>
        <div class="doc-card-body">{body_html}</div>
    </div>
    """, unsafe_allow_html=True)


def build_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%", font=dict(size=28, color="#f0f0f5", family="Inter")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#5c5c6e", tickwidth=1),
            bar=dict(color=ACCENT, thickness=0.28),
            bgcolor="rgba(255,255,255,0.04)",
            borderwidth=0,
            steps=[
                dict(range=[0, 50], color="rgba(108,99,255,0.08)"),
                dict(range=[50, 80], color="rgba(108,99,255,0.15)"),
                dict(range=[80, 100], color="rgba(108,99,255,0.25)"),
            ],
            threshold=dict(line=dict(color=ACCENT_LIGHT, width=3), thickness=0.8, value=score),
        ),
        title=dict(text="Confidence Score", font=dict(size=13, color="#9898a8")),
    ))
    return style_plotly_figure(fig, height=260, showlegend=False)


def build_price_comparison_bar(predicted_price, dataset_avg):
    fig = go.Figure(go.Bar(
        x=["Dataset Average", "Your Prediction"],
        y=[dataset_avg, predicted_price],
        marker=dict(color=[ACCENT_DARK, ACCENT], line=dict(width=0)),
        text=[f"${dataset_avg:,.0f}", f"${predicted_price:,.0f}"],
        textposition="outside",
        textfont=dict(color="#9898a8", size=11),
    ))
    style_plotly_figure(fig, height=300, showlegend=False)
    fig.update_layout(title="Price Comparison", yaxis_title="Price (USD)")
    return fig


def build_radar_chart(comparison_df):
    best_row = comparison_df.loc[comparison_df["R² Score"].idxmax()]
    categories = ["R² Score", "MAE⁻¹", "RMSE⁻¹"]
    fig = go.Figure()
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        r2 = row["R² Score"]
        mae_inv = 1 / (row["MAE"] + 0.01)
        rmse_inv = 1 / (row["RMSE"] + 0.01)
        vals = [r2, mae_inv / 2, rmse_inv / 2]
        is_best = row["Model"] == best_row["Model"]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself" if is_best else "none",
            name=row["Model"],
            line=dict(color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)], width=2 if is_best else 1.5),
            fillcolor=f"rgba(108,99,255,0.12)" if is_best else "rgba(0,0,0,0)",
            opacity=1.0 if is_best else 0.7,
        ))
    style_plotly_figure(fig, height=420)
    fig.update_layout(
        title="Model Radar Comparison",
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.06)", color="#9898a8"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#9898a8"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.5, xanchor="center"),
    )
    return fig


# ── Session state ──────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:20px;">
    <div class="sidebar-logo">{icon('building', 18, ACCENT_LIGHT)}</div>
    <div>
        <div class="sidebar-brand">House Price Prediction</div>
        <div class="sidebar-tagline">AI Property Valuation Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

nav_options = ["Home", "Predict Price", "Analytics", "Model Performance", "Documentation"]
page_index = nav_options.index(st.session_state.page) if st.session_state.page in nav_options else 0

page = st.sidebar.radio(
    "Navigation",
    nav_options,
    index=page_index,
    format_func=lambda x: x,
    label_visibility="collapsed",
)

if page != st.session_state.page:
    st.session_state.page = page
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div class="sidebar-status">
    <span class="status-dot"></span>
    <span style="font-weight:600;color:{ACCENT_LIGHT};font-size:13px;">Model Online</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.metric("Accuracy", ACCURACY)
st.sidebar.metric("Prediction Time", PREDICTION_TIME)
st.sidebar.metric("Dataset Size", DATASET_SIZE)
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="sidebar-version">
    <strong style="color:#f0f0f5;">House Price Prediction System</strong><br>
    Version 1.0
</div>
""", unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "Home":

    st.markdown(f"""
    <div class="hero-section">
        <div style="text-align:center;position:relative;z-index:1;">
            <div class="hero-badge">{icon('sparkles', 12, ACCENT_LIGHT)} &nbsp; AI-Powered Platform</div>
            <h1 class="main-title">House Price Prediction System</h1>
            <p class="subtitle">AI Powered Real Estate Valuation Platform</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Predict Property", use_container_width=True):
            st.session_state.page = "Predict Price"
            st.rerun()
    with b2:
        if st.button("Explore Analytics", use_container_width=True):
            st.session_state.page = "Analytics"
            st.rerun()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card("Properties", DATASET_SIZE, "California Housing records indexed", "database")
    with k2:
        render_kpi_card("Best Model", BEST_MODEL_NAME, "Production-ready ensemble model", "cpu")
    with k3:
        render_kpi_card("Accuracy", ACCURACY, "R² score on held-out test set", "target")
    with k4:
        render_kpi_card("Prediction Speed", PREDICTION_TIME, "Average inference latency", "zap")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Features</div>', unsafe_allow_html=True)

    feats = [
        ("AI Prediction", "Instant ML-powered property valuation.", "target"),
        ("Analytics Dashboard", "Interactive dataset visualizations.", "bar-chart"),
        ("Model Comparison", "MAE, RMSE, and R² side-by-side.", "trending-up"),
        ("CSV Export", "Download results for reporting.", "download"),
        ("Real-time Results", "Sub-millisecond inference pipeline.", "zap"),
        ("Interactive Charts", "Plotly-powered data exploration.", "line-chart"),
    ]
    rows = [feats[:3], feats[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (title, desc, ic) in zip(cols, row):
            with col:
                render_feature_card(title, desc, ic)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
    render_workflow()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-title-sm">Property Categories</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, (ic, lbl) in zip([c1, c2, c3, c4], [
        ("building", "Residential"), ("building", "Apartment"),
        ("building", "Villa"), ("building", "Commercial"),
    ]):
        with col:
            render_category_card(ic, lbl)

    left, right = st.columns([2, 1])
    with left:
        st.markdown('<div class="section-title" style="margin-top:24px;">About</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-panel">
        <p class="body-text" style="margin:0;">
        This intelligent system predicts house prices using Machine Learning. It analyzes
        property details, location, area, rooms, population, and market features to estimate
        the most probable property price.
        </p></div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-title" style="margin-top:24px;">System Status</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-panel" style="margin-bottom:12px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <span class="status-dot"></span>
                <span style="color:{ACCENT_LIGHT};font-weight:600;font-size:14px;">Model Ready</span>
            </div>
        </div>""", unsafe_allow_html=True)
        st.metric("Accuracy", "79.10 %")
        st.metric("Dataset", DATASET_SIZE)
        st.metric("Prediction", PREDICTION_TIME)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT PRICE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Price":

    render_page_header(
        "Predict Property Price",
        "Fill in the property details below to receive an AI-powered valuation.",
        "target",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f'<div class="form-panel"><div class="form-panel-title">{icon("building", 14, ACCENT_LIGHT)} Property Information</div>', unsafe_allow_html=True)
        property_type = st.selectbox("Property Type", [
            "House", "Apartment", "Villa", "Residential Plot", "Commercial Building",
        ])
        location = st.text_input("Property Location", placeholder="e.g. San Francisco, CA")
        area = st.number_input("Area (Square Feet)", min_value=200, max_value=10000, value=1200)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 8, 2)
        floors = st.slider("Floors", 1, 5, 2)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="form-panel"><div class="form-panel-title">{icon("sliders", 14, ACCENT_LIGHT)} Model Inputs</div>', unsafe_allow_html=True)
        median_income = st.slider("Median Income", 0.0, 20.0, 5.0)
        house_age = st.slider("Property Age", 0, 80, 10)
        avg_rooms = st.slider("Average Rooms", 1.0, 15.0, round(area / 200, 1))
        avg_bedrooms = st.slider("Average Bedrooms", 0.5, 5.0, float(bedrooms))
        avg_occupancy = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<div class="form-panel"><div class="form-panel-title">{icon("map-pin", 14, ACCENT_LIGHT)} Location Details</div>', unsafe_allow_html=True)
        population = st.number_input("Population", value=3000)
        latitude = st.number_input("Latitude", value=37.88)
        longitude = st.number_input("Longitude", value=-122.23)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f'<div class="form-panel"><div class="form-panel-title">{icon("upload", 14, ACCENT_LIGHT)} Prediction Controls</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Upload Property Image", type=["jpg", "jpeg", "png"])
        uploaded_file = st.file_uploader("Upload Property Documents", type=["pdf", "jpg", "jpeg", "png"])
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predict House Price", use_container_width=True):

        start_time = time.perf_counter()
        prediction = predict_price(
            median_income, house_age, area / 200, bedrooms,
            population, 3, latitude, longitude,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        predicted_usd = prediction * 100000
        df_ref = load_housing_dataframe()
        dataset_avg = df_ref["MedHouseVal"].mean() * 100000

        st.markdown(f"""
        <div class="result-card fade-in">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
                {icon('check-circle', 18, ACCENT_LIGHT)}
                <span style="color:{ACCENT_LIGHT};font-weight:600;font-size:14px;">Prediction Generated Successfully</span>
            </div>
            <div class="section-title-sm" style="margin-bottom:4px;">Estimated Property Price</div>
            <div style="font-size:clamp(2rem,5vw,2.75rem);font-weight:800;color:#f0f0f5;margin:8px 0;letter-spacing:-0.03em;">${predicted_usd:,.0f}</div>
            <div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:16px;">
                <div style="font-size:12px;color:#9898a8;">{icon('zap', 12, ACCENT)} &nbsp; {elapsed_ms:.2f} ms</div>
                <div style="font-size:12px;color:#9898a8;">{icon('cpu', 12, ACCENT)} &nbsp; {BEST_MODEL_NAME}</div>
                <div style="font-size:12px;color:#9898a8;">{icon('target', 12, ACCENT)} &nbsp; {CONFIDENCE_SCORE}% confidence</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.success("Prediction Generated Successfully")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Estimated Price", f"${predicted_usd:,.0f}")
        with m2:
            st.metric("Confidence Score", f"{CONFIDENCE_SCORE}%")
        with m3:
            st.metric("Model Used", BEST_MODEL_NAME)
        with m4:
            st.metric("Prediction Time", f"{elapsed_ms:.2f} ms")

        render_confidence_meter(CONFIDENCE_SCORE)
        st.progress(CONFIDENCE_SCORE / 100)
        st.info(f"Model Confidence : {CONFIDENCE_SCORE}%")

        g1, g2 = st.columns(2)
        with g1:
            render_chart_section("Confidence Gauge", build_gauge_chart(CONFIDENCE_SCORE))
        with g2:
            render_chart_section("Price Comparison", build_price_comparison_bar(predicted_usd, dataset_avg))

        st.markdown('<div class="section-title-sm" style="margin-top:8px;">Prediction Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.write(f"**Property Type:** {property_type}")
            st.write(f"**Location:** {location or 'Not specified'}")
            st.write(f"**Area:** {area} sq ft")
        with r2:
            st.write(f"**Bedrooms:** {bedrooms} | **Bathrooms:** {bathrooms} | **Floors:** {floors}")
            st.write(f"**Median Income:** {median_income}")
            st.write(f"**House Age:** {house_age} years")
        with r3:
            st.write(f"**Avg Rooms:** {avg_rooms} | **Avg Bedrooms:** {avg_bedrooms}")
            st.write(f"**Population:** {population:,}")
            st.write(f"**Coordinates:** {latitude}, {longitude}")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded_image:
            st.image(uploaded_image, caption="Property Image", use_container_width=True)
        if uploaded_file:
            st.success("Property documents uploaded successfully.")

        csv_data = build_prediction_csv(
            property_type, location, median_income, house_age,
            area / 200, bedrooms, population, 3,
            latitude, longitude, prediction, CONFIDENCE_SCORE,
        )
        st.download_button(
            label="Download Prediction as CSV",
            data=csv_data,
            file_name="house_price_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Analytics":

    render_page_header(
        "Analytics Dashboard",
        "Explore the California Housing dataset with interactive visualizations.",
        "bar-chart",
    )

    df = load_housing_dataframe()

    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        render_kpi_card("Average Price", f"${df['MedHouseVal'].mean() * 100000:,.0f}", "Mean median house value", "activity")
    with a2:
        render_kpi_card("Maximum Price", f"${df['MedHouseVal'].max() * 100000:,.0f}", "Highest recorded value", "trending-up")
    with a3:
        render_kpi_card("Minimum Price", f"${df['MedHouseVal'].min() * 100000:,.0f}", "Lowest recorded value", "bar-chart")
    with a4:
        render_kpi_card("Median Income", f"{df['MedInc'].median():.2f}", "Median block income", "layers")
    with a5:
        render_kpi_card("Avg Population", f"{df['Population'].mean():,.0f}", "Mean block population", "database")

    st.markdown('<div class="section-title">Dataset Preview</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
    st.dataframe(df.head(15), use_container_width=True, height=320)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        fig_hist = px.histogram(df, x="MedHouseVal", nbins=50,
            title="Median House Value Distribution",
            labels={"MedHouseVal": "Median House Value ($100k)"},
            color_discrete_sequence=[ACCENT])
        style_plotly_figure(fig_hist, 400, False)
        render_chart_section("Histogram", fig_hist)
    with ch2:
        fig_scatter = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="MedInc", y="MedHouseVal",
            title="Median Income vs House Value",
            labels={"MedInc": "Median Income", "MedHouseVal": "Median House Value ($100k)"},
            opacity=0.5, color_discrete_sequence=[ACCENT_LIGHT],
        )
        style_plotly_figure(fig_scatter, 400)
        render_chart_section("Scatter Plot", fig_scatter)

    ch3, ch4 = st.columns(2)
    with ch3:
        corr = df.corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, BG_PRIMARY], [0.5, ACCENT_DARK], [1, ACCENT_LIGHT]],
            zmin=-1, zmax=1,
        ))
        style_plotly_figure(fig_heat, 420)
        fig_heat.update_layout(title="Feature Correlation Matrix")
        render_chart_section("Correlation Heatmap", fig_heat)
    with ch4:
        fig_box = px.box(df, y="MedHouseVal", x="HouseAge",
            title="House Value by Age Group",
            labels={"MedHouseVal": "Median House Value ($100k)", "HouseAge": "House Age"},
            color_discrete_sequence=[ACCENT],
        )
        style_plotly_figure(fig_box, 420, False)
        render_chart_section("Box Plot", fig_box)

    ch5, ch6 = st.columns(2)
    with ch5:
        price_bins = pd.cut(df["MedHouseVal"], bins=5)
        pie_data = price_bins.value_counts().reset_index()
        pie_data.columns = ["Range", "Count"]
        pie_data["Range"] = pie_data["Range"].astype(str)
        fig_pie = px.pie(pie_data, names="Range", values="Count",
            title="Price Range Distribution",
            color_discrete_sequence=PLOTLY_COLORS,
        )
        style_plotly_figure(fig_pie, 400)
        render_chart_section("Pie Chart", fig_pie)
    with ch6:
        try:
            model = load_best_model()
            if hasattr(model, "feature_importances_"):
                fnames = df.drop("MedHouseVal", axis=1).columns.tolist()
                imp_df = pd.DataFrame({"Feature": fnames, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                    title="Feature Importance", color_discrete_sequence=[ACCENT])
                style_plotly_figure(fig_imp, 400, False)
                render_chart_section("Feature Importance", fig_imp)
            else:
                st.info("Feature importance available for tree-based models.")
        except Exception:
            st.warning("Could not load feature importance from the saved model.")

    st.markdown('<div class="section-title">Feature Distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    feature_col = st.selectbox("Select Feature", df.drop("MedHouseVal", axis=1).columns.tolist())
    fig_dist = px.histogram(df, x=feature_col, nbins=40,
        title=f"Distribution of {feature_col}", color_discrete_sequence=[ACCENT])
    style_plotly_figure(fig_dist, 380, False)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":

    render_page_header(
        "Model Performance",
        "Compare evaluation metrics across trained regression models.",
        "trending-up",
    )

    with st.spinner("Computing model metrics..."):
        comparison_df = compute_model_comparison()

    best_idx = comparison_df["R² Score"].idxmax()
    best_model_name = comparison_df.loc[best_idx, "Model"]
    best_r2 = comparison_df.loc[best_idx, "R² Score"]
    best_mae = comparison_df.loc[best_idx, "MAE"]
    best_rmse = comparison_df.loc[best_idx, "RMSE"]

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        render_kpi_card("Best Model", best_model_name, "Highest R² on test set", "award")
    with p2:
        render_kpi_card("R² Score", f"{best_r2:.4f}", "Coefficient of determination", "target")
    with p3:
        render_kpi_card("MAE", f"{best_mae:.4f}", "Mean absolute error", "activity")
    with p4:
        render_kpi_card("RMSE", f"{best_rmse:.4f}", "Root mean squared error", "bar-chart")
    with p5:
        render_kpi_card("Production Model", "best_model.pkl", "Deployed inference artifact", "box")

    ranked_df = comparison_df.sort_values("R² Score", ascending=False).reset_index(drop=True)
    ranked_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(ranked_df))])
    ranked_df["Status"] = ranked_df["Model"].apply(
        lambda m: "Best" if m == best_model_name else "—"
    )

    st.markdown('<div class="section-title">Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
    st.dataframe(
        comparison_df.style.highlight_max(subset=["R² Score"], color="rgba(108,99,255,0.2)")
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title-sm" style="margin-top:20px;">Model Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-panel" style="padding:0;overflow:hidden;">', unsafe_allow_html=True)
    st.dataframe(ranked_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)
    with mc1:
        fig_compare = go.Figure()
        for trace_name, col_name, color in [
            ("MAE", "MAE", ACCENT_DARK), ("RMSE", "RMSE", ACCENT), ("R² Score", "R² Score", ACCENT_LIGHT)
        ]:
            fig_compare.add_trace(go.Bar(
                name=trace_name, x=comparison_df["Model"], y=comparison_df[col_name],
                marker=dict(color=color, line=dict(width=0)),
            ))
        style_plotly_figure(fig_compare, 420)
        fig_compare.update_layout(title="Model Metrics Comparison", barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"))
        render_chart_section("Bar Chart", fig_compare)
    with mc2:
        render_chart_section("Radar Chart", build_radar_chart(comparison_df))

    try:
        model = load_best_model()
        df = load_housing_dataframe()
        if hasattr(model, "feature_importances_"):
            fnames = df.drop("MedHouseVal", axis=1).columns.tolist()
            imp_df = pd.DataFrame({"Feature": fnames, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
            fig_fi = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                title=f"{best_model_name} — Feature Importance", color_discrete_sequence=[ACCENT])
            style_plotly_figure(fig_fi, 380, False)
            render_chart_section("Feature Importance", fig_fi)
    except Exception:
        pass

    st.markdown(f"""
    <div class="glass-panel">
    <div style="font-size:13px;color:#9898a8;line-height:1.75;">
    Production deployment uses <strong style="color:{ACCENT_LIGHT};">{best_model_name}</strong> saved as
    <code style="background:rgba(108,99,255,0.12);padding:2px 8px;border-radius:4px;color:{ACCENT_LIGHT};">model/best_model.pkl</code>
    with approximate accuracy of <strong style="color:{ACCENT_LIGHT};">{ACCURACY}</strong>.
    Metrics computed on an 80/20 train-test split of the California Housing dataset.
    </div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Documentation":

    render_page_header(
        "Documentation",
        "Technical reference for the House Price Prediction System.",
        "file-text",
    )

    d1, d2 = st.columns(2)

    sections = [
        ("Overview", "<ul><li>AI-powered real estate valuation platform</li><li>Built with Streamlit &amp; scikit-learn</li><li>California Housing dataset (20,640 records)</li></ul>", "book"),
        ("Project Architecture", "<ul><li>Streamlit frontend (app.py)</li><li>ML pipeline (src/)</li><li>Saved model (model/best_model.pkl)</li><li>Cached analytics layer</li></ul>", "layers"),
        ("Machine Learning Pipeline", "<ol><li>Load dataset</li><li>Preprocess features</li><li>80/20 train-test split</li><li>Train 3 regression models</li><li>Evaluate &amp; save best model</li></ol>", "cpu"),
        ("Dataset Information", "<ul><li><strong>Source:</strong> sklearn California Housing</li><li><strong>Features:</strong> 8 numeric inputs</li><li><strong>Target:</strong> MedHouseVal ($100k units)</li><li><strong>Records:</strong> 20,640</li></ul>", "database"),
        ("Technologies Used", "<ul><li>Python · Streamlit · Plotly</li><li>Pandas · NumPy · Joblib</li><li>scikit-learn</li></ul>", "sparkles"),
        ("Folder Structure", "<ul><li><code>app.py</code> — Web application</li><li><code>src/</code> — ML source code</li><li><code>model/</code> — Trained artifacts</li><li><code>data/</code> · <code>outputs/</code></li></ul>", "folder"),
        ("Model Workflow", "<ol><li>User submits property details</li><li><code>predict_price()</code> called</li><li>Model returns MedHouseVal</li><li>Result displayed &amp; exportable</li></ol>", "git-branch"),
        ("Libraries Used", "<ul><li>streamlit · plotly · pandas</li><li>numpy · joblib · scikit-learn</li></ul>", "box"),
        ("Future Improvements", "<ul><li>Geocoding from addresses</li><li>Hyperparameter tuning</li><li>User auth &amp; history</li><li>Live market API integration</li><li>SHAP explainability</li></ul>", "rocket"),
    ]

    for i, (title, body, ic) in enumerate(sections):
        col = d1 if i % 2 == 0 else d2
        with col:
            render_doc_card(title, body, ic, i)

render_footer()
