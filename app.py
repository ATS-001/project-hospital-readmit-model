"""
======================================================================
 PULSE · 30-Day Hospital Readmission Risk Console
======================================================================
A Streamlit dashboard that predicts 30-day diabetic patient readmission
risk using a pre-trained XGBoost model.

Pipeline this app replicates (must match the training notebook exactly):
  1. Collect raw clinical inputs from the user.
  2. Engineer features the same way the training data was engineered
     (age -> numeric midpoint, diag codes -> ICD chapter buckets,
     drug columns -> ordinal 0-3 stored as strings, total_utilization sum).
  3. One-hot encode remaining categorical columns (drop_first=True,
     same as pd.get_dummies during training) using EXPLICIT category
     sets so a single-row input never silently drops its own category
     (see CATEGORY_LEVELS comment below for why this matters).
  4. Reindex columns to match feature_columns.pkl (fills any missing
     dummy columns with 0, drops/reorders to match training schema).
  5. Scale with the saved StandardScaler.
  6. Predict with the saved XGBoost model.

Deployment files required in the same directory as this script:
  - model.pkl              (trained XGBClassifier, joblib-dumped)
  - scaler.pkl             (fitted StandardScaler, joblib-dumped)
  - feature_columns.pkl    (list of training column names, joblib-dumped)
======================================================================
"""

import re
import base64
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Pulse · Readmission Risk Console",
    page_icon="◈",
    layout="wide",
)

# ======================================================================
# CONSTANTS — must mirror the preprocessing logic used during training
# ======================================================================

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "insulin",
    "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-pioglitazone",
]
DRUG_LEVELS = {"No": 0, "Steady": 1, "Up": 2, "Down": 3}
RAW_BINARY_DRUG_COLS = ["examide", "citoglipton"]

AGE_GROUPS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
RACE_OPTIONS = ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other"]
GENDER_OPTIONS = ["Female", "Male"]

ADMISSION_TYPE_MAP = {
    1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn",
    5: "Not Available", 6: "NULL", 7: "Trauma Center", 8: "Not Mapped",
}
DISCHARGE_DISPOSITION_MAP = {
    1: "Discharged to home",
    2: "Discharged/transferred to another short term hospital",
    3: "Discharged/transferred to SNF",
    4: "Discharged/transferred to ICF",
    5: "Discharged/transferred to another type of inpatient care institution",
    6: "Discharged/transferred to home with home health service",
    7: "Left AMA",
    8: "Discharged/transferred to home under care of Home IV provider",
    9: "Admitted as an inpatient to this hospital",
    10: "Neonate discharged to another hospital for neonatal aftercare",
    11: "Expired",
    12: "Still patient or expected to return for outpatient services",
    13: "Hospice / home",
    14: "Hospice / medical facility",
    15: "Discharged/transferred within institution to Medicare swing bed",
    16: "Discharged/transferred/referred to another institution for outpatient services",
    17: "Discharged/transferred/referred to this institution for outpatient services",
    18: "NULL",
    19: "Expired at home (Medicaid hospice)",
    20: "Expired in a medical facility (Medicaid hospice)",
    21: "Expired, place unknown (Medicaid hospice)",
    22: "Discharged/transferred to another rehab facility",
    23: "Discharged/transferred to a long term care hospital",
    24: "Discharged/transferred to a nursing facility (Medicaid only)",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Discharged/transferred to a federal health care facility",
    28: "Discharged/transferred/referred to a psychiatric hospital",
    29: "Discharged/transferred to a Critical Access Hospital",
    30: "Discharged/transferred to another type of health care institution",
}
ADMISSION_SOURCE_MAP = {
    1: "Physician Referral", 2: "Clinic Referral", 3: "HMO Referral",
    4: "Transfer from a hospital", 5: "Transfer from a Skilled Nursing Facility",
    6: "Transfer from another health care facility", 7: "Emergency Room",
    8: "Court/Law Enforcement", 9: "Not Available",
    10: "Transfer from critical access hospital", 11: "Normal Delivery",
    12: "Premature Delivery", 13: "Sick Baby", 14: "Extramural Birth",
    15: "Not Available", 17: "NULL", 18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency", 20: "Not Mapped",
    21: "Unknown/Invalid", 22: "Transfer from hospital inpatient (separate claim)",
    23: "Born inside this hospital", 24: "Born outside this hospital",
    25: "Transfer from Ambulatory Surgery Center", 26: "Transfer from Hospice",
}

DIAG_CHOICES = [
    "Circulatory", "Respiratory", "Digestive", "Diabetes",
    "Injury", "Musculoskeletal", "Genitourinary", "Neoplasms", "Other",
]

# ----------------------------------------------------------------------
# Explicit category sets for every column that gets one-hot encoded.
#
# WHY THIS MATTERS: pd.get_dummies(drop_first=True) infers categories
# from the data it's given. With a single input row, whatever value
# that row has would be the ONLY category present -- and drop_first=True
# would drop it entirely, silently zeroing a feature that should have
# been 1. Casting to a Categorical with the FULL training-time category
# list (alphabetical, matching pandas' own ordering) guarantees the same
# dummy columns are produced regardless of how many rows are encoded.
# ----------------------------------------------------------------------
CATEGORY_LEVELS = {
    "race": ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"],
    "gender": ["Female", "Male", "Unknown/Invalid"],
    "diag_1": sorted(DIAG_CHOICES),
    "diag_2": sorted(DIAG_CHOICES),
    "diag_3": sorted(DIAG_CHOICES),
    "change": ["Ch", "No"],
    "diabetesMed": ["No", "Yes"],
    "examide": ["No"],
    "citoglipton": ["No"],
}
for _drug in DRUG_COLS:
    CATEGORY_LEVELS[_drug] = ["0", "1", "2", "3"]


# ======================================================================
# CACHED RESOURCE LOADING
# ======================================================================
@st.cache_resource
def load_artifacts():
    """Load the trained model, scaler, and exact training column schema.
    Cached so these load once per session rather than on every rerun."""
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
    artifacts_loaded = True
    load_error = None
except Exception as e:
    artifacts_loaded = False
    load_error = str(e)


# ======================================================================
# FEATURE ENGINEERING — replicate notebook logic exactly
# ======================================================================
def age_group_to_midpoint(age_group: str) -> float:
    """Convert '[40-50)' -> 45.0, matching range_to_mean() in training."""
    nums = np.array(re.findall(r"\d+", age_group), dtype=float)
    return float(nums.mean())


def encode_drug(value: str) -> int:
    return DRUG_LEVELS.get(value, 0)


def build_raw_input_row(fv: dict) -> pd.DataFrame:
    """Assemble a single-row DataFrame of engineered features, pre-encoding,
    in the same shape the training pipeline produced before one-hot encoding."""
    row = {}
    row["age"] = age_group_to_midpoint(fv["age_group"])
    row["admission_type_id"] = fv["admission_type_id"]
    row["discharge_disposition_id"] = fv["discharge_disposition_id"]
    row["admission_source_id"] = fv["admission_source_id"]
    row["time_in_hospital"] = fv["time_in_hospital"]
    row["num_lab_procedures"] = fv["num_lab_procedures"]
    row["num_procedures"] = fv["num_procedures"]
    row["num_medications"] = fv["num_medications"]
    row["number_outpatient"] = fv["number_outpatient"]
    row["number_emergency"] = fv["number_emergency"]
    row["number_inpatient"] = fv["number_inpatient"]
    row["number_diagnoses"] = fv["number_diagnoses"]

    # Engineered feature: total_utilization (sum of 3 visit-type counts)
    row["total_utilization"] = (
        fv["number_outpatient"] + fv["number_emergency"] + fv["number_inpatient"]
    )

    row["diag_1"] = fv["diag_1"]
    row["diag_2"] = fv["diag_2"]
    row["diag_3"] = fv["diag_3"]
    row["race"] = fv["race"]
    row["gender"] = fv["gender"]
    row["change"] = fv["change"]
    row["diabetesMed"] = fv["diabetesMed"]

    # Drug columns: notebook quirk replication. In the training notebook,
    # df[drug_cols] = df[drug_cols].replace({"No":0,...}) maps strings to
    # ints but pandas keeps the column dtype as object/string. That means
    # these columns get swept up by select_dtypes(include=['object']) and
    # one-hot encoded into "<drug>_1", "<drug>_2", "<drug>_3" columns. We
    # replicate that by storing the encoded value as a STRING here.
    for col in DRUG_COLS:
        row[col] = str(encode_drug(fv.get(col, "No")))

    # examide / citoglipton: left as raw categorical text, almost always
    # "No" in the real dataset, one-hot encoded same as any other category.
    for col in RAW_BINARY_DRUG_COLS:
        row[col] = fv.get(col, "No")

    return pd.DataFrame([row])


def preprocess_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categoricals (matching training's pd.get_dummies
    with drop_first=True), then reindex to the exact training schema."""
    categorical_cols = raw_df.select_dtypes(include=["object"]).columns.tolist()

    for col in categorical_cols:
        if col in CATEGORY_LEVELS:
            raw_df[col] = pd.Categorical(raw_df[col], categories=CATEGORY_LEVELS[col])

    encoded_df = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=True)

    # Reindex: adds any missing dummy columns as 0, drops/reorders extras,
    # guarantees column order matches what the scaler/model expect.
    input_df = encoded_df.reindex(columns=feature_columns, fill_value=0)
    return input_df


def predict_readmission(fv: dict):
    """Run the full inference pipeline, return (prediction, probability)."""
    raw_df = build_raw_input_row(fv)
    input_df = preprocess_for_model(raw_df)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[:, 1]
    return int(prediction[0]), float(probability[0])


# ======================================================================
# STYLES — "Pulse" clinical-monitor identity
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500;600&display=swap');

:root {
    --bg:      #07111a;
    --panel:   #0c1b24;
    --panel2:  #0f222d;
    --line:    rgba(45,212,191,0.10);
    --cyan:    #2dd4bf;
    --cyan-dim:#1a8f82;
    --amber:   #f0a868;
    --rose:    #f4536b;
    --ink:     #dbe9ee;
    --dim:     #547585;
    --disp:    'Space Grotesk', sans-serif;
    --mono:    'IBM Plex Mono', monospace;
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: var(--disp); background: var(--bg); color: var(--ink); }
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
hr { display: none !important; }

/* Faint monitor-grid backdrop */
.stApp::before {
    content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(45,212,191,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(45,212,191,0.025) 1px, transparent 1px);
    background-size: 38px 38px;
}

/* ── TOP BAR ── */
.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 48px; height: 56px;
    background: rgba(7,17,26,0.96); backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--line);
    position: sticky; top: 0; z-index: 200;
}
.topbar-mark { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 16px; letter-spacing: -0.3px; }
.topbar-mark .glyph { color: var(--cyan); font-size: 18px; }
.topbar-sub { font-family: var(--mono); font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--dim); }
.topbar-status { display: inline-flex; align-items: center; gap: 7px; font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--cyan); background: rgba(45,212,191,0.06); border: 1px solid rgba(45,212,191,0.22); padding: 6px 13px; border-radius: 100px; }
.topbar-status::before { content: ''; width: 5px; height: 5px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 6px var(--cyan); animation: pulse-dot 2.4s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:0.2;} }

/* ── HERO ── */
.hero { display: grid; grid-template-columns: 1.05fr 0.95fr; min-height: 460px; border-bottom: 1px solid var(--line); }
.hero-copy { padding: 64px 48px; display: flex; flex-direction: column; justify-content: center; border-right: 1px solid var(--line); }
.eyebrow { display: inline-flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--cyan); margin-bottom: 20px; }
.eyebrow::before { content: ''; width: 22px; height: 1px; background: var(--cyan); box-shadow: 0 0 6px rgba(45,212,191,0.6); }
.headline { font-size: clamp(2.1rem, 3.4vw, 3.3rem); font-weight: 700; line-height: 1.08; letter-spacing: -1.4px; color: #fff; margin-bottom: 20px; }
.headline .accent { color: var(--cyan); }
.subcopy { font-family: var(--mono); font-size: 12px; font-weight: 300; line-height: 1.85; color: var(--dim); max-width: 440px; margin-bottom: 30px; }
.scope-note { display: inline-flex; align-items: center; gap: 9px; background: rgba(240,168,104,0.05); border: 1px solid rgba(240,168,104,0.18); color: var(--amber); font-family: var(--mono); font-size: 10px; padding: 9px 16px; border-radius: 6px; width: fit-content; }

.hero-trace { background: linear-gradient(150deg, #08141e, #0b1c28); display: flex; align-items: center; justify-content: center; padding: 24px; }

/* ── SECTION LABELS ── */
.content { max-width: 1180px; margin: 0 auto; padding: 48px 48px 0; }
.sec-head { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.sec-tag { font-family: var(--mono); font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: var(--dim); }
.sec-rule { flex: 1; height: 1px; background: linear-gradient(90deg, var(--line), transparent); }
.sec-title { font-size: 1.2rem; font-weight: 700; letter-spacing: -0.3px; color: var(--ink); margin-bottom: 22px; }

/* ── EXPANDERS as panel cards ── */
.stExpander { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 10px !important; margin-bottom: 12px !important; }
.stExpander summary { font-family: var(--mono) !important; font-size: 11px !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; color: var(--cyan) !important; padding: 14px 20px !important; }

/* ── FORM INPUTS ── */
.stSelectbox label, .stNumberInput label, .stSlider label {
    font-family: var(--mono) !important; font-size: 9px !important; font-weight: 500 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important; color: var(--dim) !important;
}
.stSelectbox > div > div, .stNumberInput > div > div {
    background: var(--bg) !important; border: 1px solid rgba(45,212,191,0.13) !important;
    border-radius: 8px !important; color: var(--ink) !important;
    font-family: var(--mono) !important; font-size: 13px !important; transition: all .15s;
}
.stSelectbox > div > div:focus-within, .stNumberInput > div > div:focus-within {
    border-color: rgba(45,212,191,0.45) !important; box-shadow: 0 0 0 3px rgba(45,212,191,0.07) !important;
}
.stSlider [data-baseweb="slider"] > div > div { background: var(--cyan-dim) !important; }

/* ── BUTTON ── */
div.stFormSubmitButton > button {
    background: transparent !important; color: var(--cyan) !important;
    font-family: var(--disp) !important; font-weight: 700 !important; font-size: 13px !important;
    letter-spacing: 0.8px !important; text-transform: uppercase !important;
    padding: 14px 38px !important; border: 1px solid rgba(45,212,191,0.4) !important;
    border-radius: 8px !important; width: 100% !important; transition: all .18s !important;
    box-shadow: 0 0 22px rgba(45,212,191,0.05) !important;
}
div.stFormSubmitButton > button:hover {
    background: rgba(45,212,191,0.07) !important; color: #fff !important;
    box-shadow: 0 0 36px rgba(45,212,191,0.16) !important;
}

/* ── RESULT PANEL ── */
.result-top { display: flex; align-items: center; justify-content: space-between; padding: 13px 26px; background: var(--panel2); border: 1px solid var(--line); border-bottom: none; border-radius: 12px 12px 0 0; }
.result-top-lbl { font-family: var(--mono); font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--dim); }
.badge { font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; padding: 5px 14px; border-radius: 100px; }
.badge.high { background: rgba(244,83,107,0.1); border: 1px solid rgba(244,83,107,0.3); color: var(--rose); }
.badge.low  { background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.25); color: var(--cyan); }

.result-l { padding: 36px 32px; background: var(--panel); border: 1px solid var(--line); border-top: none; border-right: none; border-radius: 0 0 0 12px; height: 100%; }
.result-r { padding: 36px 32px; background: var(--panel); border: 1px solid var(--line); border-top: none; border-left: 1px solid rgba(45,212,191,0.07); border-radius: 0 0 12px 0; height: 100%; display: flex; flex-direction: column; justify-content: center; }

.result-verdict { font-size: clamp(1.4rem,2.6vw,2.1rem); font-weight: 700; letter-spacing: -0.6px; line-height: 1.15; margin-bottom: 10px; }
.result-verdict.high { color: var(--rose); text-shadow: 0 0 36px rgba(244,83,107,0.25); }
.result-verdict.low  { color: var(--cyan); text-shadow: 0 0 36px rgba(45,212,191,0.25); }
.result-desc { font-family: var(--mono); font-size: 11.5px; font-weight: 300; line-height: 1.85; color: var(--dim); margin-bottom: 26px; }
.divider-line { height: 1px; background: var(--line); margin: 22px 0; }
.prob-label { font-family: var(--mono); font-size: 9px; letter-spacing: 2.5px; text-transform: uppercase; color: var(--dim); margin-bottom: 6px; }
.prob-num { font-size: clamp(2.6rem,4.8vw,4rem); font-weight: 700; letter-spacing: -2px; line-height: 1; }
.prob-num.high { color: var(--rose); }
.prob-num.low  { color: var(--cyan); }
.prob-unit { font-family: var(--mono); font-size: 11px; color: var(--dim); letter-spacing: 1px; margin-bottom: 22px; }

.note-box { background: rgba(45,212,191,0.03); border-left: 2px solid rgba(45,212,191,0.32); padding: 12px 15px; border-radius: 0 6px 6px 0; font-family: var(--mono); font-size: 10.5px; line-height: 1.75; color: var(--dim); }
.warn-box { background: rgba(240,168,104,0.05); border: 1px solid rgba(240,168,104,0.16); border-left: 2px solid var(--amber); padding: 11px 15px; border-radius: 0 6px 6px 0; font-family: var(--mono); font-size: 10.5px; color: var(--amber); margin-top: 10px; line-height: 1.6; }

.chip-row { display: flex; gap: 12px; margin-top: 14px; }
.chip { flex: 1; background: rgba(255,255,255,0.02); border: 1px solid var(--line); border-radius: 8px; padding: 13px 14px; text-align: center; }
.chip-lbl { font-family: var(--mono); font-size: 9px; letter-spacing: 1.5px; text-transform: uppercase; color: var(--dim); margin-bottom: 6px; }
.chip-val { font-size: 1.15rem; font-weight: 700; }
.chip-val.cyan { color: var(--cyan); }
.chip-val.amber { color: var(--amber); }

/* ── FOOTER ── */
.foot { max-width: 1180px; margin: 0 auto; padding: 26px 48px 44px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; border-top: 1px solid var(--line); }
.foot-mark { font-weight: 700; font-size: 13px; letter-spacing: -0.3px; }
.foot-mark .glyph { color: var(--cyan); }
.foot-note { font-family: var(--mono); font-size: 9px; letter-spacing: 1.8px; text-transform: uppercase; color: var(--dim); }

/* ── RESPONSIVE ── */
@media (max-width: 900px) {
    .topbar { padding: 0 20px; }
    .topbar-sub { display: none; }
    .hero { grid-template-columns: 1fr; }
    .hero-copy { padding: 44px 22px; border-right: none; border-bottom: 1px solid var(--line); }
    .hero-trace { display: none; }
    .content { padding: 36px 20px 0; }
    .result-l { border-right: 1px solid var(--line) !important; border-radius: 0 0 12px 12px !important; }
    .result-r { border-left: none !important; border-top: 1px solid var(--line) !important; border-radius: 0 0 12px 12px !important; }
    .foot { padding: 22px 20px 36px; flex-direction: column; text-align: center; }
}
@media (max-width: 640px) {
    [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
    [data-testid="stHorizontalBlock"] > div { width: 100% !important; min-width: 100% !important; flex: none !important; }
    .headline { font-size: 1.9rem; }
    .chip-row { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# TOP BAR
# ======================================================================
st.markdown("""
<div class="topbar">
  <div class="topbar-mark"><span class="glyph">◈</span> Pulse</div>
  <div style="display:flex;align-items:center;gap:18px;">
    <span class="topbar-sub">30-Day Readmission Console</span>
    <span class="topbar-status">Model Online</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ======================================================================
# HERO — animated discharge-trace SVG as the signature element
# ======================================================================
hero_copy, hero_trace = st.columns([1.05, 0.95], gap="small")

with hero_copy:
    st.markdown("""
    <div class="hero-copy">
      <div class="eyebrow">Clinical Discharge Planning</div>
      <h1 class="headline">Readmission Risk,<br><span class="accent">Before the Door Closes</span></h1>
      <p class="subcopy">
        Enter the encounter details a discharge planner already has on hand —
        admission type, utilization history, diagnoses, medications — and get
        a 30-day readmission probability from a trained XGBoost model before
        the patient leaves the floor.
      </p>
      <div class="scope-note">&#9888;&ensp; Decision support only — does not replace clinical judgment</div>
    </div>
    """, unsafe_allow_html=True)

with hero_trace:
    # Signature visual: a hospital-monitor trace that forks at "discharge"
    # into a calm line (low risk) or a spiking, irregular line (high risk).
    # This stands in for the page's single most characteristic image
    # without leaning on any borrowed iconography from other domains.
    trace_svg = """<svg viewBox="0 0 420 420" xmlns="http://www.w3.org/2000/svg" width="420" height="420">
  <defs>
    <linearGradient id="fadeL" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2dd4bf" stop-opacity="0"/>
      <stop offset="100%" stop-color="#2dd4bf" stop-opacity="0.9"/>
    </linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="2.2" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>
  <rect width="420" height="420" fill="#081420"/>
  <line x1="0" y1="105" x2="420" y2="105" stroke="rgba(45,212,191,0.05)" stroke-width="0.6"/>
  <line x1="0" y1="210" x2="420" y2="210" stroke="rgba(45,212,191,0.08)" stroke-width="0.6"/>
  <line x1="0" y1="315" x2="420" y2="315" stroke="rgba(45,212,191,0.05)" stroke-width="0.6"/>
  <line x1="140" y1="0" x2="140" y2="420" stroke="rgba(45,212,191,0.05)" stroke-width="0.6"/>

  <!-- admission marker -->
  <line x1="24" y1="40" x2="24" y2="380" stroke="rgba(45,212,191,0.18)" stroke-width="1" stroke-dasharray="2 5"/>
  <text x="24" y="28" font-family="IBM Plex Mono,monospace" font-size="9" fill="rgba(45,212,191,0.45)" letter-spacing="1" text-anchor="middle">ADMIT</text>

  <!-- discharge fork marker -->
  <line x1="160" y1="40" x2="160" y2="380" stroke="rgba(240,168,104,0.22)" stroke-width="1" stroke-dasharray="2 5"/>
  <text x="160" y="28" font-family="IBM Plex Mono,monospace" font-size="9" fill="rgba(240,168,104,0.55)" letter-spacing="1" text-anchor="middle">DISCHARGE</text>

  <!-- shared pre-discharge waveform (steady heartbeat) -->
  <path d="M 24 210 L 60 210 L 68 210 L 74 160 L 80 260 L 86 210 L 110 210 L 118 210 L 124 160 L 130 260 L 136 210 L 160 210"
        fill="none" stroke="#2dd4bf" stroke-width="1.8" filter="url(#glow)"/>

  <!-- low-risk branch: calm steady rhythm -->
  <path d="M 160 210 L 196 210 L 204 210 L 210 168 L 216 252 L 222 210 L 250 210 L 258 210 L 264 168 L 270 252 L 276 210 L 310 210 L 318 210 L 324 168 L 330 252 L 336 210 L 396 210"
        fill="none" stroke="#2dd4bf" stroke-width="1.6" opacity="0.85" filter="url(#glow)">
    <animate attributeName="opacity" values="0.85;1;0.85" dur="3.2s" repeatCount="indefinite"/>
  </path>
  <text x="398" y="200" font-family="IBM Plex Mono,monospace" font-size="9" fill="rgba(45,212,191,0.6)" letter-spacing="1" text-anchor="end">LOW RISK</text>

  <!-- high-risk branch: irregular spiking rhythm -->
  <path d="M 160 210 L 190 315 L 198 315 L 206 280 L 214 350 L 222 300 L 240 315 L 252 315 L 262 270 L 270 360 L 280 300 L 300 315 L 320 315 L 330 268 L 340 358 L 350 300 L 396 315"
        fill="none" stroke="#f4536b" stroke-width="1.6" opacity="0.85" filter="url(#glow)">
    <animate attributeName="opacity" values="0.85;0.4;0.85" dur="1.1s" repeatCount="indefinite"/>
  </path>
  <text x="398" y="305" font-family="IBM Plex Mono,monospace" font-size="9" fill="rgba(244,83,107,0.65)" letter-spacing="1" text-anchor="end">HIGH RISK</text>

  <!-- scanning sweep -->
  <line x1="0" y1="0" x2="0" y2="420" stroke="rgba(45,212,191,0.5)" stroke-width="1.4">
    <animate attributeName="x1" values="24;396;24" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="x2" values="24;396;24" dur="6s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="0.5;0.08;0.5" dur="6s" repeatCount="indefinite"/>
  </line>

  <text x="24" y="402" font-family="IBM Plex Mono,monospace" font-size="9" fill="rgba(45,212,191,0.3)" letter-spacing="1">ENCOUNTER TRACE</text>
  <text x="396" y="402" font-family="IBM Plex Mono,monospace" font-size="9" fill="rgba(45,212,191,0.25)" letter-spacing="1" text-anchor="end">T+30D WINDOW</text>
</svg>"""
    svg_b64 = base64.b64encode(trace_svg.encode()).decode()
    st.markdown(
        f'<div class="hero-trace">'
        f'<img src="data:image/svg+xml;base64,{svg_b64}" width="420" height="420" style="max-width:100%;height:auto;"/>'
        f'</div>',
        unsafe_allow_html=True,
    )

if not artifacts_loaded:
    st.error(
        "⚠️ Could not load model artifacts. Make sure `model.pkl`, "
        "`scaler.pkl`, and `feature_columns.pkl` are in the same directory "
        f"as this app.\n\nDetails: {load_error}"
    )
    st.stop()


# ======================================================================
# INPUT FORM
# ======================================================================
st.markdown('<div class="content">', unsafe_allow_html=True)

st.markdown("""
<div class="sec-head"><span class="sec-tag">Encounter Intake</span><span class="sec-rule"></span></div>
<div class="sec-title">Patient & Admission Details</div>
""", unsafe_allow_html=True)

with st.form("readmission_form"):

    with st.expander("👤  Demographics & Age", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            race = st.selectbox("Race", RACE_OPTIONS, index=0)
        with col2:
            gender = st.selectbox("Gender", GENDER_OPTIONS, index=0)
        with col3:
            age_group = st.selectbox("Age Group", AGE_GROUPS, index=6)

    with st.expander("🚑  Admission & Discharge", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            admission_type_label = st.selectbox("Admission Type", list(ADMISSION_TYPE_MAP.values()), index=0)
            admission_type_id = [k for k, v in ADMISSION_TYPE_MAP.items() if v == admission_type_label][0]

            admission_source_label = st.selectbox("Admission Source", list(ADMISSION_SOURCE_MAP.values()), index=6)
            admission_source_id = [k for k, v in ADMISSION_SOURCE_MAP.items() if v == admission_source_label][0]
        with col2:
            discharge_disposition_label = st.selectbox("Discharge Disposition", list(DISCHARGE_DISPOSITION_MAP.values()), index=0)
            discharge_disposition_id = [k for k, v in DISCHARGE_DISPOSITION_MAP.items() if v == discharge_disposition_label][0]

            time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)

    with st.expander("📊  Utilization Metrics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            num_lab_procedures = st.number_input("Lab Procedures Count", 0, 150, 40)
            number_outpatient = st.number_input("Outpatient Visits (past year)", 0, 50, 0)
        with col2:
            num_procedures = st.number_input("Procedures Count", 0, 10, 1)
            number_emergency = st.number_input("Emergency Visits (past year)", 0, 50, 0)
        with col3:
            num_medications = st.number_input("Number of Medications", 0, 80, 15)
            number_inpatient = st.number_input("Inpatient Visits (past year)", 0, 30, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 7)

    with st.expander("🩺  Diagnosis Categories", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            diag_1 = st.selectbox("Primary Diagnosis (diag_1)", DIAG_CHOICES, index=3)
        with col2:
            diag_2 = st.selectbox("Secondary Diagnosis (diag_2)", DIAG_CHOICES, index=0)
        with col3:
            diag_3 = st.selectbox("Additional Diagnosis (diag_3)", DIAG_CHOICES, index=0)

    with st.expander("💊  Medications", expanded=False):
        drug_values = {}
        drug_options = ["No", "Steady", "Up", "Down"]

        primary_drugs = [
            "metformin", "insulin", "glipizide", "glyburide",
            "pioglitazone", "rosiglitazone", "glimepiride", "repaglinide",
        ]
        cols = st.columns(4)
        for i, drug in enumerate(primary_drugs):
            with cols[i % 4]:
                drug_values[drug] = st.selectbox(drug.capitalize(), drug_options, index=0, key=f"drug_{drug}")

        with st.expander("Additional / less common medications", expanded=False):
            remaining_drugs = [d for d in DRUG_COLS if d not in primary_drugs]
            cols2 = st.columns(4)
            for i, drug in enumerate(remaining_drugs):
                with cols2[i % 4]:
                    drug_values[drug] = st.selectbox(drug.replace("-", " ").title(), drug_options, index=0, key=f"drug_{drug}")

        col1, col2 = st.columns(2)
        with col1:
            change = st.selectbox("Medication Change During Encounter", ["No", "Ch"], index=0,
                                   help="'Ch' = dosage was changed, 'No' = no change")
        with col2:
            diabetesMed = st.selectbox("Diabetes Medication Prescribed", ["No", "Yes"], index=1)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Run Risk Assessment →")


# ======================================================================
# PREDICTION & RESULTS
# ======================================================================
if submitted:
    form_values = {
        "race": race, "gender": gender, "age_group": age_group,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "diag_1": diag_1, "diag_2": diag_2, "diag_3": diag_3,
        "change": change, "diabetesMed": diabetesMed,
        **drug_values,
    }

    try:
        prediction, probability = predict_readmission(form_values)
        risk_pct = probability * 100
        safe_pct = 100 - risk_pct
        confidence = "HIGH" if abs(risk_pct - 50) > 20 else "MODERATE"

        st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="sec-head"><span class="sec-tag">Output</span><span class="sec-rule"></span></div>
        <div class="sec-title">Risk Assessment</div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            cls, verdict = "high", "High Readmission Risk"
            desc = ("This encounter pattern is consistent with patients who return "
                    "within 30 days. Consider enhanced discharge planning, a closer "
                    "follow-up window, and medication reconciliation before release.")
            badge_label = "Elevated"
        else:
            cls, verdict = "low", "Low Readmission Risk"
            desc = ("This encounter pattern is consistent with patients who are not "
                    "readmitted within 30 days. Standard discharge and follow-up "
                    "protocols are likely sufficient.")
            badge_label = "Stable"

        st.markdown(f"""
        <div class="result-top">
          <span class="result-top-lbl">Diagnostic Output</span>
          <span class="badge {cls}">{badge_label}</span>
        </div>
        """, unsafe_allow_html=True)

        lcol, rcol = st.columns([1, 1], gap="medium")

        with lcol:
            warn_html = ""
            if 45 <= risk_pct <= 55:
                warn_html = ('<div class="warn-box">&#9888; Borderline result — within the '
                              'indeterminate zone. Recommend clinical review before relying on this score.</div>')
            st.markdown(f"""
            <div class="result-l">
              <div class="result-verdict {cls}">{verdict}</div>
              <p class="result-desc">{desc}</p>
              <div class="divider-line"></div>
              <div class="prob-label">30-Day Readmission Probability</div>
              <div class="prob-num {cls}">{risk_pct:.1f}</div>
              <div class="prob-unit">percent</div>
              {warn_html}
              <div class="note-box">
                This score is probabilistic and generated by a model trained on
                historical encounter data. It should support, not replace,
                clinical judgment and full chart review.
              </div>
            </div>
            """, unsafe_allow_html=True)

        with rcol:
            gauge_color = "#f4536b" if cls == "high" else "#2dd4bf"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%", "valueformat": ".1f",
                        "font": {"size": 40, "color": "#dbe9ee", "family": "Space Grotesk"}},
                title={"text": "RISK INDEX",
                       "font": {"size": 10, "color": "#547585", "family": "IBM Plex Mono"}},
                gauge={
                    "axis": {"range": [0, 100], "nticks": 6,
                             "tickcolor": "#0c1b24",
                             "tickfont": {"color": "#2a4250", "size": 9, "family": "IBM Plex Mono"}},
                    "bar": {"color": gauge_color, "thickness": 0.2},
                    "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                    "steps": [
                        {"range": [0, 35], "color": "rgba(45,212,191,0.06)"},
                        {"range": [35, 65], "color": "rgba(240,168,104,0.06)"},
                        {"range": [65, 100], "color": "rgba(244,83,107,0.06)"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 2},
                                  "thickness": 0.72, "value": risk_pct},
                },
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=260, margin=dict(l=22, r=22, t=54, b=8),
                font={"color": "#dbe9ee", "family": "Space Grotesk"},
            )
            st.markdown('<div class="result-r">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div class="chip-row">
              <div class="chip"><div class="chip-lbl">Stable Prob.</div><div class="chip-val cyan">{safe_pct:.1f}%</div></div>
              <div class="chip"><div class="chip-lbl">Confidence</div><div class="chip-val amber">{confidence}</div></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Linear probability bar as a secondary, more literal readout
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            st.progress(min(max(probability, 0.0), 1.0))

    except Exception as e:
        st.error(f"An error occurred while generating the prediction: {e}")

else:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.info("Fill in the encounter details above and select **Run Risk Assessment** to generate a score.")

st.markdown("</div>", unsafe_allow_html=True)


# ======================================================================
# FOOTER
# ======================================================================
st.markdown("""
<div style="height:46px"></div>
<div class="foot">
  <span class="foot-mark"><span class="glyph">◈</span> Pulse</span>
  <span class="foot-note">XGBoost Model &middot; Research Prototype &middot; Not for Clinical Deployment</span>
  <span class="foot-note">Aaron Thalakkottor Sooraj &middot; Designed & Developed by ATS-PDZ &middot; © Since 2023</span>
</div>
""", unsafe_allow_html=True)
