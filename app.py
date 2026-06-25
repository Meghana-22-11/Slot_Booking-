from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "bookings_dataset.csv"
ASSET_DIR = BASE_DIR / "assets"

GALLERY_IMAGES = [
    ASSET_DIR / "sports-hero.png",
    ASSET_DIR / "venue-gallery-1.png",
    ASSET_DIR / "venue-gallery-2.png",
    ASSET_DIR / "venue-gallery-3.png",
]

CATEGORICAL_COLS = ["Sport", "Day_of_Week", "Court_Type", "Equipment_Rental"]
FEATURE_COLS = [
    "Sport",
    "Day_of_Week",
    "Court_Type",
    "Equipment_Rental",
    "Is_Weekend",
    "Start_Hour",
    "Duration_Hours",
    "Number_of_Players",
]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class PriceEngine:
    model: object
    encoders: dict[str, LabelEncoder]
    feature_order: list[str]


st.set_page_config(
    page_title="SportSlot Pro | Venue Booking",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def image_uri(path: Path) -> str:
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        --black: #000000;
        --shell: #0c0c0c;
        --surface: rgba(255, 255, 255, 0.035);
        --surface-2: rgba(255, 255, 255, 0.065);
        --line: rgba(255, 255, 255, 0.10);
        --line-hot: rgba(204, 255, 0, 0.45);
        --lime: #ccff00;
        --emerald: #10b981;
        --text: #ebebeb;
        --soft: rgba(235, 235, 235, 0.66);
        --muted: rgba(235, 235, 235, 0.38);
        --light: #e5e5e5;
    }

    .stApp {
        background:
            radial-gradient(circle at 82% 12%, rgba(204,255,0,.16), transparent 30rem),
            radial-gradient(circle at 10% 82%, rgba(16,185,129,.13), transparent 32rem),
            #000000;
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
    }

    .block-container {
        position: relative;
        max-width: 1560px;
        margin-top: 1rem;
        margin-bottom: 2rem;
        padding: 1rem 1.25rem 2rem;
        border-radius: 2.5rem;
        overflow: visible;
        border: 1px solid var(--line);
        background:
            linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px),
            var(--shell);
        background-size: 60px 60px;
        box-shadow: 0 36px 120px rgba(0,0,0,.78);
    }

    .block-container::before {
        content: "";
        position: absolute;
        inset: 0;
        opacity: .15;
        pointer-events: none;
        mix-blend-mode: overlay;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 160 160' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.45'/%3E%3C/svg%3E");
    }

    .block-container > * {
        position: relative;
        z-index: 1;
    }

    h1, h2, h3, p, div, span, label {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0;
    }

    .mono,
    .stSelectbox label,
    .stDateInput label,
    .stTextInput label,
    .stTextArea label {
        font-family: 'JetBrains Mono', monospace !important;
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .67rem !important;
    }

    .nav {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1rem;
        padding: 1.35rem 2rem .7rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: .75rem;
        font-weight: 800;
        color: var(--text);
    }

    .brand-mark {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: .9rem;
        background: var(--lime);
        color: #000;
        display: grid;
        place-items: center;
        font-weight: 900;
        box-shadow: 0 0 34px rgba(204,255,0,.36);
    }

    .nav-pill {
        display: flex;
        gap: .35rem;
        padding: .35rem;
        border-radius: 999px;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.045);
        backdrop-filter: blur(16px);
    }

    .nav-pill a {
        color: rgba(235,235,235,.72);
        padding: .65rem 1.05rem;
        border-radius: 999px;
        font-size: .9rem;
        font-weight: 700;
        text-decoration: none;
    }

    .nav-pill a:first-child {
        color: #000;
        background: var(--lime);
    }

    .nav-pill a:hover {
        color: #000;
        background: rgba(204,255,0,.78);
    }

    .nav-actions {
        display: flex;
        justify-content: flex-end;
        gap: .75rem;
        align-items: center;
    }

    .status {
        display: flex;
        align-items: center;
        gap: .5rem;
        color: var(--soft);
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .15em;
        font-size: .66rem;
    }

    .dot {
        width: .45rem;
        height: .45rem;
        border-radius: 50%;
        background: var(--lime);
        box-shadow: 0 0 18px rgba(204,255,0,.9);
    }

    .white-btn,
    .lime-btn {
        border-radius: 999px;
        padding: .85rem 1.2rem;
        font-weight: 800;
        text-align: center;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        justify-content: center;
    }

    .white-btn {
        background: #fff;
        color: #000;
    }

    .lime-btn {
        background: var(--lime);
        color: #000;
        box-shadow: 0 0 34px rgba(204,255,0,.32);
    }

    .hero {
        display: grid;
        grid-template-columns: minmax(0, 1.04fr) minmax(380px, .96fr);
        gap: 1rem;
        min-height: 650px;
        padding: 1.3rem 2rem 2rem;
        align-items: stretch;
    }

    .hero-copy {
        padding: clamp(2rem, 5vw, 4.5rem) 2.25rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .glass {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 2rem;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    .eyebrow {
        color: var(--lime);
        border: 1px solid rgba(204,255,0,.32);
        background: rgba(204,255,0,.06);
        border-radius: 999px;
        padding: .42rem .75rem;
        display: inline-flex;
        width: fit-content;
        margin-bottom: 2rem;
    }

    .hero h1 {
        color: var(--text);
        font-size: clamp(4rem, 8.8vw, 8.6rem);
        letter-spacing: -0.07em;
        line-height: .84;
        margin: 0 0 1.6rem;
    }

    .hero h1 em {
        color: var(--lime);
        font-style: italic;
        font-family: Georgia, serif;
        letter-spacing: -0.08em;
        text-shadow: 0 0 32px rgba(204,255,0,.22);
    }

    .hero-copy p {
        color: var(--soft);
        font-size: 1.08rem;
        line-height: 1.65;
        max-width: 620px;
        margin: 0 0 2rem;
    }

    .hero-actions {
        display: flex;
        gap: .8rem;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 2.4rem;
    }

    .trusted {
        display: flex;
        gap: 1.7rem;
        color: var(--muted);
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .18em;
        font-size: .72rem;
        margin-top: auto;
    }

    .hero-visual {
        padding: 1.3rem;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .device {
        position: relative;
        width: 100%;
        min-height: 520px;
        border-radius: 2rem;
        border: 1px solid var(--line);
        background: #050505;
        padding: 1.6rem;
        overflow: hidden;
    }

    .device img {
        width: 100%;
        height: 390px;
        object-fit: cover;
        border-radius: 1.4rem;
        filter: grayscale(.15) brightness(.72);
    }

    .venue-card {
        position: absolute;
        left: 3rem;
        right: 3rem;
        bottom: 2rem;
        padding: 1.15rem;
        border-radius: 1.4rem;
        background: rgba(20,20,20,.72);
        border: 1px solid rgba(255,255,255,.13);
        backdrop-filter: blur(16px);
    }

    .venue-card h3 {
        margin: .2rem 0 .45rem;
        font-size: 1.25rem;
        color: var(--text);
    }

    .venue-card p {
        color: var(--soft);
        margin: 0;
        line-height: 1.5;
    }

    .float-tag {
        position: absolute;
        right: 2rem;
        bottom: 1.2rem;
        background: var(--lime);
        color: #000;
        border-radius: 999px;
        padding: .7rem 1rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .7rem;
        font-weight: 800;
        box-shadow: 0 0 30px rgba(204,255,0,.4);
    }

    .section {
        padding: 5rem 2rem;
    }

    .section-head {
        max-width: 760px;
        margin-bottom: 2rem;
    }

    .section-head h2 {
        color: var(--text);
        font-size: clamp(2.6rem, 5.6vw, 5rem);
        line-height: .92;
        letter-spacing: -0.06em;
        margin: 0;
    }

    .section-head h2 em {
        color: var(--lime);
        font-family: Georgia, serif;
        font-style: italic;
    }

    .section-head p {
        color: var(--soft);
        max-width: 650px;
        line-height: 1.65;
        margin-top: 1rem;
    }

    .gallery {
        display: grid;
        grid-template-columns: 1.45fr 1fr 1fr;
        grid-auto-rows: 230px;
        gap: 1rem;
    }

    .gallery-card {
        overflow: hidden;
        border-radius: 2rem;
        border: 1px solid var(--line);
        background: var(--surface);
    }

    .gallery-card:first-child {
        grid-row: span 2;
    }

    .gallery-card img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: saturate(.82) contrast(1.05);
        transition: transform .45s cubic-bezier(.4, 0, .2, 1);
    }

    .gallery-card:hover img {
        transform: scale(1.04);
    }

    .venue-grid,
    .amenity-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }

    .sports-wrap {
        position: relative;
    }

    .sports-grid {
        display: flex;
        gap: 1rem;
        overflow-x: auto;
        scroll-snap-type: x mandatory;
        scroll-behavior: smooth;
        padding: .25rem 3.2rem .8rem 0;
        scrollbar-width: thin;
        scrollbar-color: var(--lime) transparent;
    }

    .sports-grid::-webkit-scrollbar {
        height: .45rem;
    }

    .sports-grid::-webkit-scrollbar-track {
        background: rgba(255,255,255,.05);
        border-radius: 999px;
    }

    .sports-grid::-webkit-scrollbar-thumb {
        background: var(--lime);
        border-radius: 999px;
    }

    .scroll-arrow {
        position: absolute;
        right: .35rem;
        top: 50%;
        transform: translateY(-50%);
        width: 2.8rem;
        height: 2.8rem;
        border-radius: 999px;
        display: grid;
        place-items: center;
        background: var(--lime);
        color: #000;
        font-size: 1.5rem;
        font-weight: 900;
        box-shadow: 0 0 30px rgba(204,255,0,.36);
        border: 0;
        cursor: pointer;
    }

    .info-card,
    .sport-card {
        min-height: 190px;
        padding: 1.5rem;
    }

    .sport-card {
        min-width: 300px;
        scroll-snap-align: start;
    }

    .info-card h3,
    .sport-card h3 {
        color: var(--text);
        margin: 0 0 .8rem;
        font-size: 1.45rem;
    }

    .info-card p,
    .sport-card p {
        color: var(--soft);
        line-height: 1.6;
        margin: 0;
    }

    .icon-box {
        width: 3.2rem;
        height: 3.2rem;
        border-radius: 1rem;
        display: grid;
        place-items: center;
        background: rgba(204,255,0,.11);
        border: 1px solid rgba(204,255,0,.28);
        color: var(--lime);
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 1.2rem;
        font-weight: 800;
    }

    .about-panel {
        display: grid;
        grid-template-columns: minmax(0, .9fr) minmax(0, 1.1fr);
        gap: 2rem;
        align-items: start;
    }

    .about-text {
        color: var(--soft);
        line-height: 1.78;
        font-size: 1.02rem;
    }

    .rule-list {
        display: grid;
        gap: .9rem;
    }

    .rule {
        display: grid;
        grid-template-columns: 3rem 1fr;
        gap: 1rem;
        align-items: start;
        padding: 1rem;
        border-radius: 1.4rem;
        border: 1px solid var(--line);
        background: rgba(255,255,255,.025);
    }

    .rule-num {
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        display: grid;
        place-items: center;
        border: 1px solid var(--line);
        color: var(--lime);
        font-family: 'JetBrains Mono', monospace;
    }

    .booking-section {
        display: grid;
        grid-template-columns: minmax(0, 1.25fr) minmax(330px, .75fr);
        gap: 1rem;
        padding: 5rem 2rem;
    }

    .form-title {
        color: var(--text);
        font-size: 2rem;
        letter-spacing: -0.04em;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stForm"] {
        border: 1px solid var(--line);
        border-radius: 2rem;
        background: var(--surface);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        padding: 1.4rem;
    }

    .stSelectbox label,
    .stDateInput label,
    .stTextInput label,
    .stTextArea label {
        color: var(--soft) !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"],
    div[data-testid="stDateInput"] div[data-baseweb="input"],
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stDateInput"] input,
    div[data-testid="stTextInput"] input,
    textarea {
        min-height: 3rem;
        border-radius: 999px !important;
        background: rgba(255,255,255,.055) !important;
        border: 1px solid rgba(255,255,255,.13) !important;
        color: var(--text) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    div[data-baseweb="input"] {
        box-shadow: none !important;
    }

    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(204,255,0,.48) !important;
        box-shadow: 0 0 0 1px rgba(204,255,0,.24), 0 0 28px rgba(204,255,0,.10) !important;
    }

    textarea {
        border-radius: 1.35rem !important;
    }

    .stButton > button,
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        min-height: 3rem;
        border-radius: 999px;
        background: var(--lime);
        color: #000;
        border: 1px solid var(--lime);
        font-weight: 900;
        box-shadow: 0 0 34px rgba(204,255,0,.34);
        transition: transform .2s cubic-bezier(.4,0,.2,1), box-shadow .2s cubic-bezier(.4,0,.2,1);
    }

    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] button:hover {
        background: var(--lime);
        color: #000;
        border-color: var(--lime);
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 0 52px rgba(204,255,0,.48);
    }

    .checkout {
        padding: 1.4rem;
        min-height: 100%;
    }

    .price {
        color: var(--text);
        font-size: clamp(2.8rem, 5vw, 5rem);
        font-weight: 800;
        letter-spacing: -0.06em;
        line-height: 1;
        margin: .9rem 0 1.2rem;
    }

    .mini-item {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: .72rem 0;
        border-bottom: 1px solid rgba(255,255,255,.08);
        color: var(--soft);
    }

    .mini-item strong {
        color: var(--text);
        text-align: right;
    }

    .success-card {
        border-radius: 1.4rem;
        border: 1px solid var(--line-hot);
        background: rgba(204,255,0,.1);
        padding: 1rem;
        margin-top: 1rem;
        color: var(--text);
    }

    .light-section {
        margin: 1rem 0 0;
        padding: 5rem 2rem;
        border-radius: 4rem 4rem 0 0;
        background: var(--light);
        color: #000;
        display: grid;
        grid-template-columns: minmax(0, .9fr) minmax(0, 1.1fr);
        gap: 2rem;
        align-items: center;
    }

    .light-section h2 {
        font-size: clamp(3rem, 6vw, 5.8rem);
        line-height: .92;
        letter-spacing: -0.07em;
        margin: 0 0 2rem;
    }

    .light-section h2 em {
        color: #176a3a;
        font-family: Georgia, serif;
        font-style: italic;
    }

    .method {
        display: grid;
        grid-template-columns: 3.2rem 1fr;
        gap: 1rem;
        margin: 1.2rem 0;
        align-items: start;
    }

    .method-num {
        width: 3rem;
        height: 3rem;
        border-radius: 50%;
        border: 1px solid #c8c8c8;
        display: grid;
        place-items: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .method h3 {
        margin: .2rem 0 .35rem;
        color: #000;
    }

    .method p {
        color: #666;
        margin: 0;
        line-height: 1.55;
    }

    .portrait-card {
        min-height: 520px;
        border-radius: 999px 999px 2rem 2rem;
        background:
            radial-gradient(circle at 50% 32%, rgba(0,0,0,.22), transparent 18rem),
            linear-gradient(145deg, #bdbdbd, #ececec);
        position: relative;
        overflow: hidden;
    }

    .portrait-card::after {
        content: "Venue Flow";
        position: absolute;
        inset: auto 2rem 2rem;
        border-radius: 1.5rem;
        padding: 1.25rem;
        color: #fff;
        background: rgba(0,0,0,.42);
        border: 1px solid rgba(255,255,255,.28);
        backdrop-filter: blur(16px);
        font-size: 1.4rem;
        font-weight: 700;
    }

    .footer {
        padding: 5rem 2rem 2rem;
        background: #000;
        position: relative;
        overflow: hidden;
    }

    .watermark {
        position: absolute;
        left: 2rem;
        top: 2rem;
        font-size: clamp(5rem, 12vw, 11rem);
        font-weight: 900;
        color: rgba(255,255,255,.055);
        letter-spacing: -0.08em;
    }

    .footer-cta {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        gap: 2rem;
        min-height: 260px;
        text-align: right;
    }

    .footer-cta h2 {
        color: var(--text);
        font-size: clamp(2.5rem, 5vw, 4.8rem);
        line-height: .95;
        margin: 0;
        letter-spacing: -0.05em;
    }

    .footer-bottom {
        position: relative;
        z-index: 1;
        border-top: 1px solid rgba(255,255,255,.08);
        padding-top: 1.8rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: var(--muted);
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .12em;
        font-size: .68rem;
    }

    .help-float {
        position: fixed;
        right: 1rem;
        bottom: 1rem;
        z-index: 99;
        border-radius: 999px;
        padding: .9rem 1rem;
        color: #000;
        background: var(--lime);
        box-shadow: 0 0 36px rgba(204,255,0,.42);
        font-weight: 900;
    }

    @media (max-width: 980px) {
        .nav,
        .hero,
        .booking-section,
        .light-section,
        .about-panel {
            grid-template-columns: 1fr;
        }

        .nav {
            justify-items: start;
        }

        .nav-actions {
            justify-content: flex-start;
        }

        .gallery,
        .venue-grid,
        .amenity-grid {
            grid-template-columns: 1fr;
        }

        .gallery-card:first-child {
            grid-row: auto;
        }

        .hero {
            min-height: auto;
            padding: 1rem;
        }

        .section,
        .booking-section,
        .light-section,
        .footer {
            padding: 3rem 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Booking_Date"] = pd.to_datetime(df["Booking_Date"])
    return df


@st.cache_resource
def train_engine(df: pd.DataFrame) -> PriceEngine:
    model_df = df.drop(columns=["Booking_Date"]).copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        encoder = LabelEncoder()
        model_df[col] = encoder.fit_transform(model_df[col])
        encoders[col] = encoder

    X = model_df[FEATURE_COLS]
    y = model_df["Booking_Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    ]
    best_model = models[0]
    best_score = -1.0

    for model in models:
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
        if score > best_score:
            best_model = model
            best_score = score

    return PriceEngine(model=best_model, encoders=encoders, feature_order=FEATURE_COLS)


def initialize_state() -> None:
    defaults = {
        "access_mode": None,
        "user_name": "",
        "quote": None,
        "booking": None,
        "spot_booked": False,
        "paid": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def ordered_options(df: pd.DataFrame, column: str) -> list:
    values = df[column].dropna().unique().tolist()
    if column == "Day_of_Week":
        return [day for day in DAY_ORDER if day in values]
    return sorted(values)


def predict_price(engine: PriceEngine, values: dict) -> float:
    encoded = values.copy()
    for col in CATEGORICAL_COLS:
        encoded[col] = engine.encoders[col].transform([encoded[col]])[0]
    row = pd.DataFrame([encoded])[engine.feature_order]
    return max(float(engine.model.predict(row)[0]), 0)


def format_price(value: float) -> str:
    return f"Rs. {value:,.2f}"


def render_static_top() -> None:
    hero_img = image_uri(GALLERY_IMAGES[0])
    st.markdown(
        f"""
            <div class="nav">
                <div class="brand"><div class="brand-mark">S</div><div>SportSlot Pro</div></div>
                <div class="nav-pill"><a href="#play">PLAY</a><a href="#book">BOOK</a><a href="#teams">TEAMS</a><a href="#venue">VENUES</a></div>
                <div class="nav-actions">
                    <div class="status"><span class="dot"></span><span>Venue live</span></div>
                    <a class="white-btn" href="#book">Login / Signup</a>
                </div>
            </div>
            <section class="hero" id="play">
                <div class="hero-copy glass">
                    <div class="eyebrow mono">Premium sports venue desk</div>
                    <h1>Book your <em>sports slot</em></h1>
                    <p>A polished booking experience for courts, turfs, training lanes, corporate sports days, coaching batches, and team play.</p>
                    <div class="hero-actions">
                        <a class="lime-btn" href="#book">Book Now</a>
                        <a class="white-btn" href="#venue">View Venue</a>
                    </div>
                    <div class="trusted"><span>Open 6 AM - 11 PM</span><span>4.8 venue score</span><span>Bulk / Corporate</span></div>
                </div>
                <div class="hero-visual glass">
                    <div class="device">
                        <img src="{hero_img}" alt="Premium sports venue" />
                        <div class="venue-card">
                            <div class="mono" style="color: var(--lime);">Featured venue</div>
                            <h3>Velocity Sports Arena</h3>
                            <p>Multi-sport booking hub with clean surfaces, flexible slots, team scheduling, and venue-grade amenities.</p>
                        </div>
                        <div class="float-tag">Bookable today</div>
                    </div>
                </div>
            </section>
        """,
        unsafe_allow_html=True,
    )


def render_gallery_and_venue(df: pd.DataFrame) -> None:
    images = [image_uri(path) for path in GALLERY_IMAGES]
    cards = "".join(
        f'<div class="gallery-card"><img src="{src}" alt="Venue image {index}" /></div>'
        for index, src in enumerate(images, start=1)
        if src
    )
    sport_copy = {
        "Badminton": ("BD", "Indoor court slots for singles, doubles, training drills, and quick evening matches."),
        "Basketball": ("BK", "Indoor and outdoor court slots for pickup games, coaching, and team sessions."),
        "Cricket": ("CR", "Netted cricket setup for short-format matches, practice lanes, and group bookings."),
        "Football": ("FT", "Fast-paced turf slots for casual games, corporate play, and competitive teams."),
        "Pickleball": ("PB", "Compact court sessions for doubles play, beginner batches, and social leagues."),
        "Tennis": ("TN", "Clean court access for coaching, friendly games, and focused practice sessions."),
    }
    sport_cards = ""
    for sport in ordered_options(df, "Sport"):
        code, description = sport_copy.get(
            sport,
            (sport[:2].upper(), "Flexible sports slot for casual play, coaching, and team bookings."),
        )
        sport_cards += (
            f'<div class="sport-card glass"><div class="icon-box">{code}</div>'
            f'<h3>{sport}</h3><p>{description}</p></div>'
        )
    st.markdown(
        f"""
            <section class="section" id="venue">
                <div class="section-head">
                    <div class="eyebrow mono">Venue profile</div>
                    <h2>A premium arena <em>built for teams.</em></h2>
                    <p>Browse venue photos, check sports, review amenities, and move straight into slot booking without digging through a messy interface.</p>
                </div>
                <div class="gallery">{cards}</div>
            </section>

            <section class="section">
                <div class="venue-grid">
                    <div class="info-card glass"><div class="icon-box">01</div><h3>Timing</h3><p>6 AM - 11 PM, open all week for morning training, evening games, and weekend tournaments.</p></div>
                    <div class="info-card glass"><div class="icon-box">02</div><h3>Location</h3><p>Central city access with parking, public transport connectivity, and easy arrival for large teams.</p></div>
                    <div class="info-card glass"><div class="icon-box">03</div><h3>Venue Score</h3><p>4.8 average user score across surface quality, lighting, cleanliness, and support experience.</p></div>
                </div>
            </section>

        """,
        unsafe_allow_html=True,
    )

    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --lime: #ccff00;
                    --text: #ebebeb;
                    --soft: rgba(235, 235, 235, 0.66);
                    --line: rgba(255, 255, 255, 0.10);
                    --surface: rgba(255, 255, 255, 0.035);
                }}

                html, body {{
                    margin: 0;
                    padding: 0;
                    background: transparent;
                    color: var(--text);
                    font-family: 'Space Grotesk', sans-serif;
                    overflow: hidden;
                }}

                .section {{
                    padding: 5rem 2rem;
                    box-sizing: border-box;
                }}

                .section-head {{
                    max-width: 760px;
                    margin-bottom: 2rem;
                }}

                .eyebrow {{
                    color: var(--lime);
                    border: 1px solid rgba(204,255,0,.32);
                    background: rgba(204,255,0,.06);
                    border-radius: 999px;
                    padding: .42rem .75rem;
                    display: inline-flex;
                    width: fit-content;
                    margin-bottom: 2rem;
                    font-family: 'JetBrains Mono', monospace;
                    text-transform: uppercase;
                    letter-spacing: .16em;
                    font-size: .67rem;
                }}

                h2 {{
                    color: var(--text);
                    font-size: clamp(2.6rem, 5.6vw, 5rem);
                    line-height: .92;
                    letter-spacing: -0.06em;
                    margin: 0;
                }}

                h2 em {{
                    color: var(--lime);
                    font-family: Georgia, serif;
                    font-style: italic;
                }}

                .sports-wrap {{
                    position: relative;
                }}

                .sports-grid {{
                    display: flex;
                    gap: 1rem;
                    overflow-x: auto;
                    scroll-snap-type: x mandatory;
                    scroll-behavior: smooth;
                    padding: .25rem 3.2rem .8rem 0;
                    scrollbar-width: thin;
                    scrollbar-color: var(--lime) transparent;
                }}

                .sports-grid::-webkit-scrollbar {{
                    height: .45rem;
                }}

                .sports-grid::-webkit-scrollbar-track {{
                    background: rgba(255,255,255,.05);
                    border-radius: 999px;
                }}

                .sports-grid::-webkit-scrollbar-thumb {{
                    background: var(--lime);
                    border-radius: 999px;
                }}

                .sport-card {{
                    min-height: 190px;
                    min-width: 300px;
                    padding: 1.5rem;
                    scroll-snap-align: start;
                    background: var(--surface);
                    border: 1px solid var(--line);
                    border-radius: 2rem;
                    backdrop-filter: blur(16px);
                    -webkit-backdrop-filter: blur(16px);
                    box-sizing: border-box;
                }}

                .sport-card h3 {{
                    color: var(--text);
                    margin: 0 0 .8rem;
                    font-size: 1.45rem;
                }}

                .sport-card p {{
                    color: var(--soft);
                    line-height: 1.6;
                    margin: 0;
                    font-size: 1rem;
                }}

                .icon-box {{
                    width: 3.2rem;
                    height: 3.2rem;
                    border-radius: 1rem;
                    display: grid;
                    place-items: center;
                    background: rgba(204,255,0,.11);
                    border: 1px solid rgba(204,255,0,.28);
                    color: var(--lime);
                    font-family: 'JetBrains Mono', monospace;
                    margin-bottom: 1.2rem;
                    font-weight: 800;
                }}

                .scroll-arrow {{
                    position: absolute;
                    right: .35rem;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 2.8rem;
                    height: 2.8rem;
                    border-radius: 999px;
                    display: grid;
                    place-items: center;
                    background: var(--lime);
                    color: #000;
                    font-size: 1.5rem;
                    font-weight: 900;
                    box-shadow: 0 0 30px rgba(204,255,0,.36);
                    border: 0;
                    cursor: pointer;
                }}

                .scroll-arrow:active {{
                    transform: translateY(-50%) scale(.94);
                }}

                @media (max-width: 980px) {{
                    .section {{
                        padding: 3rem 1rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <section class="section">
                <div class="section-head">
                    <div class="eyebrow">Sports available</div>
                    <h2>Choose your <em>game.</em></h2>
                </div>
                <div class="sports-wrap">
                    <div class="sports-grid" id="sportsRail">{sport_cards}</div>
                    <button class="scroll-arrow" type="button" aria-label="Scroll games" onclick="document.getElementById('sportsRail').scrollBy({{left: 330, behavior: 'smooth'}})">›</button>
                </div>
            </section>
        </body>
        </html>
        """,
        height=520,
        scrolling=False,
    )

    st.markdown(
        """
            <section class="section">
                <div class="section-head">
                    <div class="eyebrow mono">Amenities</div>
                    <h2>Everything needed <em>on-site.</em></h2>
                </div>
                <div class="amenity-grid">
                    <div class="info-card glass"><div class="icon-box">P</div><h3>Parking</h3><p>Dedicated parking access for players, coaches, and event organizers.</p></div>
                    <div class="info-card glass"><div class="icon-box">R</div><h3>Restroom</h3><p>Clean restroom and changing access available for booked teams.</p></div>
                    <div class="info-card glass"><div class="icon-box">W</div><h3>Drinking Water</h3><p>Water access available for players during booked sessions.</p></div>
                </div>
            </section>
        """,
        unsafe_allow_html=True,
    )


def render_about() -> None:
    st.markdown(
        """
            <section class="section">
                <div class="about-panel">
                    <div class="section-head">
                        <div class="eyebrow mono">About venue</div>
                        <h2>Ready for <em>serious play.</em></h2>
                        <p class="about-text">
                            Velocity Sports Arena is a modern multi-sport facility designed for teams, schools, clubs, and corporate groups. Book football, cricket, basketball, badminton, tennis, and pickleball slots from one clean venue desk.
                            The venue supports casual matches, training sessions, coaching batches, and bulk bookings with a smooth checkout flow.
                        </p>
                    </div>
                    <div class="rule-list">
                        <div class="rule"><div class="rule-num">01</div><div><h3>Multi-sport ready</h3><p>Switch between sport types and court formats depending on team size and slot availability.</p></div></div>
                        <div class="rule"><div class="rule-num">02</div><div><h3>Team-friendly booking</h3><p>Designed for individual players, friend groups, clubs, and recurring sports sessions.</p></div></div>
                        <div class="rule"><div class="rule-num">03</div><div><h3>Rules at a glance</h3><p>Use turf-friendly footwear, arrive on time, respect the next slot, and follow venue safety guidelines.</p></div></div>
                    </div>
                </div>
            </section>
        """,
        unsafe_allow_html=True,
    )


def render_access() -> bool:
    if st.session_state.access_mode in {"login", "guest"}:
        return True

    st.markdown('<section class="booking-section" id="book">', unsafe_allow_html=True)
    left, right = st.columns([1.15, .85], gap="medium")
    with left:
        st.markdown('<div class="form-title">Login / Signup</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            name = st.text_input("Name", placeholder="Enter your name")
            email = st.text_input("Email", placeholder="name@email.com")
            login_clicked = st.form_submit_button("Continue")
        if login_clicked:
            st.session_state.access_mode = "login"
            st.session_state.user_name = name.strip() or "Guest"
            st.rerun()

    with right:
        st.markdown(
            """
            <div class="checkout glass">
                <div class="mono" style="color: var(--lime);">Quick access</div>
                <h3 style="font-size: 2rem; margin: .8rem 0 1rem;">Skip login</h3>
                <p style="color: var(--soft); line-height: 1.6;">Continue straight to slot details. You can still add contact and payment information before confirmation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Skip Login"):
            st.session_state.access_mode = "guest"
            st.session_state.user_name = "Guest"
            st.rerun()
    st.markdown("</section>", unsafe_allow_html=True)
    return False


def render_booking(df: pd.DataFrame, engine: PriceEngine) -> None:
    st.markdown('<section class="booking-section" id="book">', unsafe_allow_html=True)
    left, right = st.columns([1.25, .75], gap="medium")

    with left:
        st.markdown('<div class="form-title">Book Your Slot</div>', unsafe_allow_html=True)
        sport_options = ordered_options(df, "Sport")
        hour_options = ordered_options(df, "Start_Hour")
        duration_options = ordered_options(df, "Duration_Hours")
        player_options = ordered_options(df, "Number_of_Players")
        min_booking_date = df["Booking_Date"].min().date()
        max_booking_date = df["Booking_Date"].max().date()

        with st.form("booking_form"):
            c1, c2, c3 = st.columns(3, gap="medium")
            with c1:
                sport = st.selectbox("Sport", sport_options, index=None, placeholder="Select game")
                booking_date = st.date_input(
                    "Booking date",
                    value=None,
                    min_value=min_booking_date,
                    max_value=max_booking_date,
                    format="DD/MM/YYYY",
                )
                start_hour = st.selectbox("Start hour", hour_options, index=None, placeholder="Select time")
            with c2:
                duration = st.selectbox("Duration in hours", duration_options, index=None, placeholder="Select duration")
                players = st.selectbox("Number of players", player_options, index=None, placeholder="Select players")
                court_type = st.selectbox("Court type", ordered_options(df, "Court_Type"), index=None, placeholder="Select court")
            with c3:
                equipment = st.selectbox("Equipment rental", ordered_options(df, "Equipment_Rental"), index=None, placeholder="Select option")
                contact = st.text_input("Contact person", value=st.session_state.user_name)
            submitted = st.form_submit_button("Show Price")

        if submitted:
            missing_fields = [
                name
                for name, value in {
                    "sport": sport,
                    "booking date": booking_date,
                    "start hour": start_hour,
                    "duration": duration,
                    "players": players,
                    "court type": court_type,
                    "equipment rental": equipment,
                }.items()
                if value is None
            ]
            if missing_fields:
                st.warning("Please select " + ", ".join(missing_fields) + " before viewing the price.")
                return
            day = booking_date.strftime("%A")
            is_weekend = 1 if day in {"Saturday", "Sunday"} else 0
            booking = {
                "Sport": sport,
                "Booking_Date": booking_date.isoformat(),
                "Day_of_Week": day,
                "Court_Type": court_type,
                "Equipment_Rental": equipment,
                "Is_Weekend": is_weekend,
                "Start_Hour": int(start_hour),
                "Duration_Hours": int(duration),
                "Number_of_Players": int(players),
                "Contact": contact.strip() or "Guest",
            }
            st.session_state.booking = booking
            st.session_state.quote = predict_price(engine, booking)
            st.session_state.spot_booked = False
            st.session_state.paid = False
            st.rerun()

    with right:
        render_checkout()

    st.markdown("</section>", unsafe_allow_html=True)


def render_checkout() -> None:
    booking = st.session_state.booking
    quote = st.session_state.quote

    if not booking or quote is None:
        st.markdown(
            """
            <div class="checkout glass">
                <div class="mono" style="color: var(--lime);">Checkout</div>
                <div class="price" style="font-size: 2.9rem;">Show price</div>
                <p style="color: var(--soft); line-height: 1.6;">Enter slot details to view the booking price and complete payment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="checkout glass">
            <div class="mono" style="color: var(--lime);">Slot price</div>
            <div class="price">{format_price(quote)}</div>
            <div class="mini-item"><span>Venue</span><strong>Velocity Sports Arena</strong></div>
            <div class="mini-item"><span>Sport</span><strong>{booking["Sport"]}</strong></div>
            <div class="mini-item"><span>Date</span><strong>{booking["Booking_Date"]}</strong></div>
            <div class="mini-item"><span>Slot</span><strong>{booking["Day_of_Week"]}, {booking["Start_Hour"]:02d}:00</strong></div>
            <div class="mini-item"><span>Players</span><strong>{booking["Number_of_Players"]}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.spot_booked:
        if st.button("Book Spot"):
            st.session_state.spot_booked = True
            st.rerun()
        return

    if st.session_state.paid:
        st.markdown(
            """
            <div class="success-card">
                Booking confirmed. Payment has been recorded for this slot.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.form("payment_form"):
        payment_method = st.selectbox("Payment method", ["Corporate card", "UPI", "Net banking", "Invoice"])
        billing_ref = st.text_input("Billing reference", placeholder="PO number or billing note")
        paid = st.form_submit_button("Pay and Confirm")

    if paid:
        st.session_state.paid = True
        st.session_state.payment_method = payment_method
        st.session_state.billing_ref = billing_ref
        st.rerun()


def render_light_and_footer() -> None:
    st.markdown(
        """
            <section class="light-section" id="teams">
                <div>
                    <div class="mono" style="color: #176a3a; border: 1px solid #b8c5bc; border-radius: 999px; width: fit-content; padding: .45rem .8rem;">Venue operations</div>
                    <h2>Simple booking, <em>premium control.</em></h2>
                    <div class="method"><div class="method-num">1</div><div><h3>Discover the venue</h3><p>Users scan photos, sports, amenities, timing, and rules before they commit.</p></div></div>
                    <div class="method"><div class="method-num">2</div><div><h3>Choose slot details</h3><p>The booking desk collects sport, day, duration, court, equipment, and team size cleanly.</p></div></div>
                    <div class="method"><div class="method-num">3</div><div><h3>Reserve and pay</h3><p>Teams can book the spot and complete payment from the same checkout panel.</p></div></div>
                </div>
                <div class="portrait-card"></div>
            </section>
            <footer class="footer">
                <div class="watermark">SLOT</div>
                <div class="footer-cta">
                    <div>
                        <h2>Ready to book your next match?</h2>
                        <div class="lime-btn" style="display: inline-block; margin-top: 1.4rem;">Book Now</div>
                    </div>
                </div>
                <div class="footer-bottom">
                    <div>Privacy Policy &nbsp;&nbsp; Terms of Service &nbsp;&nbsp; Support</div>
                    <div>2026 SportSlot Pro</div>
                </div>
            </footer>
        <div class="help-float">Need help?</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    initialize_state()
    df = load_data()
    engine = train_engine(df)

    render_static_top()
    render_gallery_and_venue(df)
    render_about()
    if render_access():
        render_booking(df, engine)
    render_light_and_footer()


if __name__ == "__main__":
    main()
