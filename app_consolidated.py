"""
app.py — Synchronous Telehealth Dashboard
==========================================
Reads aggregated parquet files produced by build_aggregates.py.
All paths are relative to AGG_DIR at the top of the file.

To replicate for Remote Monitoring or eConsults:
  • Change VARIANT_LABEL
  • Change AGG_DIR to the matching build_aggregates output folder
  • Nothing else needs to change.

Data files expected in AGG_DIR:
  util_by_period.parquet        monthly util/1K + member months
  util_by_th_type.parquet       monthly util by delivery type
  util_by_demo.parquet          % breakdown by demographic dimension
  util_by_demo_period.parquet   monthly trend by demographic dimension
  top_dx.parquet                top diagnoses (CCS)
  top_mh_dx.parquet             top MH diagnoses (ICD-10 F-chapter)
  county_summary.parquet        TH patients/claims by county × year
  enroll_denom.parquet          member-months + enrolled persons by (year, payer)

Optional (for county map):
  data/us_counties.json         US counties GeoJSON (local, CDN-free)
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION  (change these two lines to switch variants)
# ══════════════════════════════════════════════════════════════════════════════

# Map display name → build_aggregates.py output folder for each variant.
# Paths can be absolute or relative to this file's directory.
VARIANTS: dict[str, Path] = {
    "Synchronous Telehealth"    : Path(__file__).parent / "apcd_cache" / "aggregates" / "telehealth",
    "Remote Patient Monitoring" : Path(__file__).parent / "apcd_cache" / "aggregates" / "remote_monitoring",
    # "eConsults"                 : Path(__file__).parent / "apcd_cache" / "aggregates" / "econsults",
}

# NOTE: NOTEBOOK_DIRS has been removed. county_th_metrics.parquet and the
# payer-stratified breakdowns are now produced by build_aggregates.py into the
# same per-variant folder as every other aggregate, so VARIANTS above is the
# only data-path config. The notebook no longer writes anything the dashboard
# reads — its Part C Cell 24 and Part D can be deleted.

# US counties GeoJSON — download once, keep locally (no CDN at runtime).
GEOJSON_PATH = Path(__file__).parent / "us_counties.json"

# Rolling-average window (months) shown on utilization trend charts.
ROLLING_WINDOW = 3

# Counts below this threshold are suppressed. Suppression is applied UPSTREAM in
# build_aggregates.py, so the parquet files themselves carry nulls, never small
# counts — the hosted files are safe on their own. This threshold and the flag
# column below are only used to render those nulls as "<11" rather than "—",
# plus a defensive re-check for stale parquets built before that change.
DISPLAY_SUPPRESS_THRESHOLD = 11
SUPPRESS_FLAG_COL          = "suppressed"   # must match build_aggregates.py

# How a suppressed cell is rendered. The whole point of carrying the flag from
# build_aggregates is to distinguish three states that all arrive as null:
#   suppressed  -> "<11"  : data EXISTS but is withheld (a real, small count)
#   no data     -> "—"    : nothing to report for that cell
#   rate on a suppressed row -> SUPPRESS_RATE_MARK, because "<11" is a claim
#                             about a count and would be nonsense on a percent
# Charts get the same treatment: a suppressed category is drawn as a zero-length
# hatched bar labelled "<11" so it still occupies its slot, rather than vanishing
# and implying the group had no telehealth at all.
SUPPRESS_MARK       = f"<{DISPLAY_SUPPRESS_THRESHOLD}"
SUPPRESS_RATE_MARK  = "‡"
NO_DATA_MARK        = "—"
SUPPRESS_COLOR      = "#D9D9D9"    # neutral grey for suppressed chart elements
SUPPRESS_LEGEND     = (f"**{SUPPRESS_MARK}** = suppressed: fewer than "
                       f"{DISPLAY_SUPPRESS_THRESHOLD} people, withheld for privacy · "
                       f"**{SUPPRESS_RATE_MARK}** = rate withheld on a suppressed row · "
                       f"**{NO_DATA_MARK}** = no data")


def _supp_flag(df: pd.DataFrame) -> pd.Series:
    """Boolean suppression flag for a frame, False everywhere if absent."""
    if SUPPRESS_FLAG_COL in df.columns:
        return df[SUPPRESS_FLAG_COL].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def _bar_display(values: pd.Series, flag: pd.Series | None = None,
                 base_colors: list | None = None, fmt: str = "{:.1f}%"):
    """
    Build (x, text, colors, patterns) for a bar chart that shows suppressed
    categories instead of dropping them.

    A suppressed bar gets length 0, a grey hatched fill and the "<11" label, so
    the category keeps its row on the axis and reads as withheld. Without this
    the bar simply isn't drawn and the category looks like a true zero.

    Also guards the NaN-formatting bug: applying "{:.1f}%" to a null renders the
    literal text "nan%" next to an invisible bar.
    """
    if flag is None:
        flag = pd.Series(False, index=values.index)
    flag = flag.reindex(values.index).fillna(False).astype(bool)
    base_colors = base_colors or [PALETTE[0]] * len(values)

    x, text, colors, patterns = [], [], [], []
    for v, f, c in zip(values, flag, base_colors):
        if bool(f) or pd.isna(v):
            x.append(0.0)
            text.append(SUPPRESS_MARK if bool(f) else "")
            colors.append(SUPPRESS_COLOR)
            patterns.append("/" if bool(f) else "")
        else:
            x.append(float(v))
            text.append(fmt.format(v))
            colors.append(c)
            patterns.append("")
    return x, text, colors, patterns
GRAIN_ALL                  = "ALL"          # sentinel for a collapsed dimension

# Utilization metric column. The denominator is DISTINCT CLAIMANTS in the month
# — everyone with any facility claim — matching the notebook's Cell 4, so the
# dashboard and the notebook now produce the same series.
#
# This is a share-of-care-seekers measure, not a population rate, and it is NOT
# comparable to Colorado's dashboard, which reports visits per 1,000 covered
# members. Any side-by-side with Colorado must say which denominator each uses.
UTIL_COL         = "util_per_1k_claimants"
UTIL_COL_LEGACY  = "util_per_1k_member_months"
UTIL_LABEL       = "Utilization per 1,000 Claimants"
UTIL_LABEL_SHORT = "Util / 1K"


def util_col(df: pd.DataFrame) -> str | None:
    """Return whichever utilization column this aggregate carries."""
    for c in (UTIL_COL, UTIL_COL_LEGACY):
        if c in df.columns:
            return c
    return None

# Colour scale cap percentile for the county map (prevents urban outliers
# from washing out rural variation).
MAP_CAP_PCT = 95

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    "#1B4F8A", "#00808B", "#E07B39", "#3A7D44",
    "#6B4C9A", "#C0392B", "#7F8C8D", "#5B9BD5",
]

PAYER_COLORS: dict[str, str] = {
    "Commercial"              : "#2196F3",
    "Medicaid"                : "#4CAF50",
    "Medicare Fee-for-Service": "#FF9800",
    "Medicare Advantage"      : "#E91E63",
    "ALL"                     : "#37474F",
}

TH_TYPE_COLORS: dict[str, str] = {
    "Video (Telemedicine)"   : "#1565C0",
    "Audio-Only (Telehealth)": "#2E7D32",
    "Facility Telehealth"    : "#E65100",
    "Unclassified"           : "#757575",
}

COUNTY_MAP_METRICS: dict[str, dict] = {
    "th_patients": {
        "label": "TH Patients (count)",
        "colorscale": "Blues",
        "fmt": lambda v: f"{v:,.0f}",
    },
    "th_claims": {
        "label": "TH Claims (count)",
        "colorscale": "YlOrBr",
        "fmt": lambda v: f"{v:,.0f}",
    },
    "pct_of_state_patients": {
        "label": "% of Statewide TH Patients",
        "colorscale": "Oranges",
        "fmt": lambda v: f"{v:.2f}%",
    },
    "th_per_1000": {
        "label": "TH Patients per 1,000 Total Facility Claimants",
        "colorscale": "Greens",
        "fmt": lambda v: f"{v:.1f}",
    },
}

# Plain-language definitions shown as an info blurb in the County Map tab.
# Keys must match COUNTY_MAP_METRICS.
MAP_METRIC_DEFINITIONS: dict[str, str] = {
    "th_patients": (
        "Total number of unique patients in the county who had at least one "
        "telehealth claim during the selected period. A patient with multiple "
        "telehealth visits is counted once."
    ),
    "th_claims": (
        "Total number of telehealth claims filed by patients in the county "
        "during the selected period. A single patient may contribute multiple "
        "claims, so this count can exceed the number of unique TH patients."
    ),
    "pct_of_state_patients": (
        "Each county's share of all telehealth patients statewide: "
        "(county TH patients ÷ total TH patients across all counties) × 100. "
        "Values sum to 100% across all counties shown."
    ),
    "th_per_1000": (
        "Telehealth penetration rate, that is, unique telehealth patients per 1,000 "
        "unique patients who filed ANY facility claim (telehealth or "
        "otherwise) in that county. Measures TH adoption relative to overall "
        "healthcare utilization in the county, independent of county size."
    ),
}

FIPS_TO_STATE: dict[str, str] = {
    "01":"Alabama","02":"Alaska","04":"Arizona","05":"Arkansas",
    "06":"California","08":"Colorado","09":"Connecticut","10":"Delaware",
    "11":"District of Columbia","12":"Florida","13":"Georgia","15":"Hawaii",
    "16":"Idaho","17":"Illinois","18":"Indiana","19":"Iowa","20":"Kansas",
    "21":"Kentucky","22":"Louisiana","23":"Maine","24":"Maryland",
    "25":"Massachusetts","26":"Michigan","27":"Minnesota","28":"Mississippi",
    "29":"Missouri","30":"Montana","31":"Nebraska","32":"Nevada",
    "33":"New Hampshire","34":"New Jersey","35":"New Mexico","36":"New York",
    "37":"North Carolina","38":"North Dakota","39":"Ohio","40":"Oklahoma",
    "41":"Oregon","42":"Pennsylvania","44":"Rhode Island",
    "45":"South Carolina","46":"South Dakota","47":"Tennessee","48":"Texas",
    "49":"Utah","50":"Vermont","51":"Virginia","53":"Washington",
    "54":"West Virginia","55":"Wisconsin","56":"Wyoming","72":"Puerto Rico",
}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Telehealth Services Analysis — APCD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background-color: #1B4F8A;
        color: white;
        text-align: center;
        padding: 14px 20px;
        font-size: 24px;
        font-weight: bold;
        letter-spacing: 1.5px;
        border-radius: 5px;
        margin-bottom: 12px;
    }
    .sub-header {
        color: #1B4F8A;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 2px;
    }
    .sub-caption {
        color: #000;
        font-size: 16px;
        font-style: italic;
        margin-top: 0;
        margin-bottom: 8px;
    }
    .kpi-box {
        background: #F4F6FA;
        border-left: 4px solid #1B4F8A;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 6px;
    }
    .sidebar-attribution {
        border-top: 1px solid rgba(49, 51, 63, 0.2);
        margin-top: 24px;
        padding-top: 12px;
        font-size: 12px;
        line-height: 1.45;
        color: #4A5568;
    }
    .sidebar-attribution p {
        margin: 0 0 8px 0;
    }
    .sidebar-attribution p:last-child {
        margin-bottom: 0;
    }
    .sidebar-attribution strong {
        color: #1B4F8A;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2B — NARRATIVE CONTENT (about / code definitions / limitations)
# ══════════════════════════════════════════════════════════════════════════════
# Prose and reference tables live here as module-level constants so the wording
# is edited in exactly one place. Both render_* functions below are called from
# the main column (not the sidebar): the code tables need horizontal room, and
# the sidebar stays a pure control surface.

ABOUT_MD = """
Telemedicine has emerged as one of the most consequential shifts in healthcare
delivery of the past decade, transforming from a niche accommodation into a
mainstream point of access for millions of Virginians.

This report presents an analysis of telemedicine utilization across Virginia
using facility claims data from the **Virginia All-Payer Claims Database
(APCD)**, administered by **Virginia Health Information**, covering the period
**2018 through 2023**. The APCD captures claims submitted across commercial,
Medicaid, Medicare Fee-for-Service, and Medicare Advantage payers, providing a
broad view of covered healthcare encounters in the Commonwealth.
"""

# Category → codes. Kept as (category, codes) pairs rather than one code per row
# so the table stays short enough to read at a glance; the source text groups
# them this way too.
VARIANT_CODE_DEFS: dict[str, list[tuple[str, str]]] = {
    "Synchronous Telehealth": [
        ("CPT-4 procedure codes",
         "Q3014, G0071, G2025, 99441–99443, 98966–98968, G0406, G0407, G0408, "
         "G0459, G0425, G0426, G0427, G0508, G0509"),
        ("Telemedicine billing modifiers", "95, GT"),
        ("Default revenue codes", "0780, 0789"),
        ("Audio-only modifiers", "93 (post-COVID expansion)"),
    ],
    "Remote Patient Monitoring": [
        ("CPT-4 procedure codes", "99453, 99454, 99457, 99458, 99091"),
    ],
}

APCD_LIMITATIONS_MD = """
**1. Many encounters are not reflected in the APCD**, including: visits with no
insurance claim (uninsured, public health clinics); excluded insurance (worker's
compensation, Medigap, long-term disability); ERISA self-funded plans
(non-opt-in employer plans); visits at federal facilities (VA, IHS, military);
non-traditional settings (schools, jails, community programs); pharmacy-only or
lab-only services; and denied or non-reimbursed care.

**2. County demographic data does not always distinguish independent cities
from the surrounding county** (e.g. Fairfax City vs. Fairfax County), so some
regions may appear to have no counts in certain years on the maps. This pattern
is inconsistent — some independent cities are correctly coded (e.g.
Charlottesville City, Staunton City).
"""

# Short form of limitation #2, surfaced directly under the choropleth where the
# gap is actually visible rather than only in the collapsed section at the foot
# of the page.
MAP_CITY_CAVEAT = (
    "Note: APCD county coding does not consistently separate independent cities "
    "from their surrounding county, so some regions may appear empty in a given "
    "year. See *Data Limitations* in the About panel at the top of the page."
)


def render_about(variant_label: str) -> None:
    """Single collapsed panel: overview, limitations, and the data dictionary.

    All three live in one expander rather than splitting limitations to the page
    footer, so a reader evaluating the code definitions sees the caveats in the
    same place. Variant-aware — the dictionary table follows the sidebar
    selection.
    """
    with st.expander("ℹ️  About this dashboard — overview, limitations & data dictionary"):
        st.markdown("#### Dashboard Overview")
        st.markdown(ABOUT_MD)

        st.markdown("#### Data Limitations")
        st.markdown(APCD_LIMITATIONS_MD)

        st.markdown("#### Data Dictionary")
        st.markdown(
            f"Claims are identified as **{variant_label}** encounters using the "
            f"following codes:"
        )
        code_rows = VARIANT_CODE_DEFS.get(variant_label)
        if code_rows:
            st.dataframe(
                pd.DataFrame(code_rows, columns=["Code group", "Codes"]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("Code definitions for this variant have not been published yet.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _read(agg_dir: Path, name: str) -> pd.DataFrame:
    """Read a parquet from agg_dir, return empty DataFrame on missing file."""
    p = agg_dir / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner="Loading data…")
def load_all(agg_dir_str: str) -> dict[str, pd.DataFrame]:
    agg_dir = Path(agg_dir_str)
    frames = {
        "period"      : _read(agg_dir, "util_by_period.parquet"),
        # Annual grain, counted directly from claims rather than summed from
        # util_by_period's (possibly suppressed) monthly rows.
        "year"        : _read(agg_dir, "util_by_year.parquet"),
        "th_type"     : _read(agg_dir, "util_by_th_type.parquet"),
        "demo"        : _read(agg_dir, "util_by_demo.parquet"),
        "demo_period" : _read(agg_dir, "util_by_demo_period.parquet"),
        "top_dx"      : _read(agg_dir, "top_dx.parquet"),
        "top_mh_dx"   : _read(agg_dir, "top_mh_dx.parquet"),
        "county"       : _read(agg_dir, "county_summary.parquet"),
        # Payer-stratified county counts (build_aggregates.py). Carries a
        # Payer_Type column incl. an 'ALL' rollup block. Absent → the County Map
        # falls back to the all-payer county_summary.
        "county_payer" : _read(agg_dir, "county_summary_payer.parquet"),
        "denom"        : _read(agg_dir, "enroll_denom.parquet"),
    }

    # Normalise source_year to a STRING key. It can no longer be an int: the
    # aggregates now carry an explicit "ALL" grain row alongside the real years,
    # and coercing that to Int64 would silently null it out — taking the
    # all-years view of every page with it.
    for name, df in frames.items():
        if "source_year" in df.columns:
            _sy = df["source_year"].astype(str)
            _yr = _sy.str.extract(r"(\d{4})")[0]
            frames[name]["source_year"] = _yr.where(_sy != GRAIN_ALL, GRAIN_ALL)

    # county_fips → zero-padded 5-char string (both county grains)
    for _k in ("county", "county_payer"):
        if "county_fips" in frames[_k].columns:
            frames[_k]["county_fips"] = (
                frames[_k]["county_fips"].astype(str).str.zfill(5)
            )
            frames[_k]["state_fips"] = frames[_k]["county_fips"].str[:2]

    return frames


@st.cache_resource(show_spinner="Loading GeoJSON…")
def load_geojson() -> tuple[dict | None, dict[str, str]]:
    if not GEOJSON_PATH.exists():
        return None, {}
    with open(GEOJSON_PATH) as f:
        raw = json.load(f)
    fips_to_name = {
        str(feat["id"]).zfill(5): (
            feat["properties"].get("NAME", "") + ", " +
            FIPS_TO_STATE.get(feat["properties"].get("STATE", ""), "")
        )
        for feat in raw["features"]
    }
    return raw, fips_to_name


@st.cache_data(show_spinner=False)
def load_county_th_metrics(agg_dir_str: str) -> pd.DataFrame:
    """
    Load county_th_metrics.parquet — now written by build_aggregates.py into
    AGG_DIR alongside every other aggregate, not by the notebook.

    Contains: source_year, Payer_Type, county_fips, state_fips,
              total_claimants, th_claimants, pct_distribution, th_per_1000,
              suppressed.

    Two schema changes from the notebook version:
      • `year` (Int64) → `source_year` (string, carrying the GRAIN_ALL sentinel),
        matching every other aggregate. The map therefore gains a real all-years
        grain: a county suppressed in each individual year can still appear on
        the all-years map, where its total clears the threshold. The old code
        faked this by summing the yearly rows, which could only ever sum values
        that had already been nulled.
      • Payer_Type is now a dimension in this same file, with an ALL block that
        is a true distinct count.
    """
    p = Path(agg_dir_str) / "county_th_metrics.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["county_fips"] = df["county_fips"].astype(str).str.zfill(5)
    df["state_fips"]  = df["county_fips"].str[:2]
    if "source_year" in df.columns:
        _sy = df["source_year"].astype(str)
        _yr = _sy.str.extract(r"(\d{4})")[0]
        df["source_year"] = _yr.where(_sy != GRAIN_ALL, GRAIN_ALL)
    return df


def split_payer_views(DATA: dict, CTH: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Return the payer-stratified views, and collapse DATA to its all-payer rows.

    The *_payer parquets no longer exist. build_aggregates now writes Payer_Type
    as a marginal inside each base file, with an ALL block that is a true
    distinct count rather than a sum of the payer rows — so the payer view and
    the all-payer view are the same file read two ways, and cannot drift apart
    the way two separately-generated directories could.

    This function MUTATES DATA: after it runs, DATA["demo"] and friends contain
    only Payer_Type == ALL. That is deliberate. Every page that reads DATA[...]
    was written when those files had no payer dimension, so leaving the payer
    rows in place would silently stack them on top of the ALL row and roughly
    double every count — rendering without error. Collapsing once, here, means
    none of those call sites need to change.
    """
    keys = ("demo", "demo_period", "top_dx", "top_mh_dx")
    views: dict[str, pd.DataFrame] = {}
    for k in keys:
        df = DATA.get(k, pd.DataFrame())
        views[k] = df
        if not df.empty and "Payer_Type" in df.columns:
            DATA[k] = df[df["Payer_Type"] == GRAIN_ALL].copy()
    views["county"] = CTH
    return views


# ── Payer selection ──────────────────────────────────────────────────────────
# These three used to SUM the rows of the selected payers back into the shape of
# the all-payer file, because the notebook's *_payer parquets held only the
# per-payer detail. build_aggregates now writes payer as a marginal, so each
# payer's rows are already a complete, independently-counted, independently-
# suppressed grain — with its percentages computed against that payer's own
# total. Selecting is therefore both simpler and strictly more correct:
#
#   • summing revived nothing, since suppressed cells are null before they get
#     here, and the min_count=1 guard propagated that null to the total
#   • summing double-counted anyone with coverage under two payers
#   • on the new schema, summing would also add up the dimension_value == ALL
#     marginal alongside the detail rows it summarises, roughly doubling every
#     figure and halving every percentage
#
# The sidebar is single-select, so payers always has exactly one element; the
# isin() is kept so a future multi-select degrades to a sum-free union rather
# than silently mis-aggregating.

def _demo_by_payer(demo_payer: pd.DataFrame, payers: list[str]) -> pd.DataFrame:
    """Select the payer grain from util_by_demo."""
    if demo_payer.empty or "Payer_Type" not in demo_payer.columns:
        return pd.DataFrame()
    return demo_payer[demo_payer["Payer_Type"].isin(payers)].copy()


def _demo_period_by_payer(dp_payer: pd.DataFrame, payers: list[str]) -> pd.DataFrame:
    """Select the payer grain from util_by_demo_period."""
    if dp_payer.empty or "Payer_Type" not in dp_payer.columns:
        return pd.DataFrame()
    return dp_payer[dp_payer["Payer_Type"].isin(payers)].copy()


def _dx_by_payer(dx_payer: pd.DataFrame, payers: list[str]) -> pd.DataFrame:
    """Select the payer grain from top_dx / top_mh_dx."""
    if dx_payer.empty or "Payer_Type" not in dx_payer.columns:
        return pd.DataFrame()
    return dx_payer[dx_payer["Payer_Type"].isin(payers)].copy()


# DATA and GEOJSON are loaded after the variant is selected in the sidebar below.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def ym_to_dt(series: pd.Series) -> pd.Series:
    """Convert YYYYMM integer column → datetime."""
    s = series.astype(str)
    return pd.to_datetime(s.str[:4] + "-" + s.str[4:] + "-01", errors="coerce")


# ── Period-grain handling for util_by_demo_period ────────────────────────────
# That aggregate carries three grains — month, quarter and year — each counted
# from raw claims and suppressed on its own value. The dashboard picks the
# FINEST grain that still has enough unsuppressed cells to draw a real trend,
# which is what keeps the low-volume variants (RPM, eConsults) and the
# pre-COVID years from rendering as an empty chart.

PERIOD_GRAIN_FREQ  = {"month": "MS", "quarter": "QS", "year": "YS"}
PERIOD_GRAIN_LABEL = {"month": "Monthly", "quarter": "Quarterly", "year": "Annual"}
PERIOD_GRAIN_ORDER = ["month", "quarter", "year"]
# Coarsen when more than this share of a grain's cells are suppressed.
PERIOD_GRAIN_MAX_SUPPRESSED = 0.40


def period_to_dt(series: pd.Series) -> pd.Series:
    """
    Parse the string period key into a datetime.
    Handles '2018-03' (month), '2018-Q1' (quarter) and '2018' (year).
    """
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    is_q = s.str.contains("Q", na=False)
    if is_q.any():
        q = s[is_q].str.split("-Q", n=1, expand=True)
        month = (pd.to_numeric(q[1], errors="coerce") - 1) * 3 + 1
        out.loc[is_q] = pd.to_datetime(
            q[0] + "-" + month.astype("Int64").astype(str).str.zfill(2) + "-01",
            errors="coerce")

    is_m = ~is_q & s.str.contains("-", na=False)
    if is_m.any():
        out.loc[is_m] = pd.to_datetime(s[is_m] + "-01", errors="coerce")

    is_y = ~is_q & ~is_m
    if is_y.any():
        out.loc[is_y] = pd.to_datetime(s[is_y] + "-01-01", errors="coerce")

    return out


def pick_period_grain(df: pd.DataFrame, preferred: str = "month") -> str:
    """
    Choose the finest grain whose suppressed share is tolerable, walking
    month → quarter → year. Falls back to the coarsest available grain if every
    option is heavily suppressed, since a sparse annual trend still beats a
    blank panel.
    """
    if df.empty or "period_grain" not in df.columns:
        return preferred
    available = [g for g in PERIOD_GRAIN_ORDER
                 if g in set(df["period_grain"].astype(str))]
    if not available:
        return preferred
    start = available.index(preferred) if preferred in available else 0
    for g in available[start:]:
        sub = df[df["period_grain"].astype(str) == g]
        if sub.empty:
            continue
        if SUPPRESS_FLAG_COL in sub.columns:
            share = float(sub[SUPPRESS_FLAG_COL].fillna(False).astype(bool).mean())
        else:
            share = float(sub["th_patients"].isna().mean())
        if share <= PERIOD_GRAIN_MAX_SUPPRESSED:
            return g
    return available[-1]


def prep_period_frame(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    """
    Slice to one grain and attach period_dt. Also supports aggregates written
    before the multi-grain change, which carry an integer `year_month` instead.
    """
    if df.empty:
        return df
    out = df.copy()
    if "period_grain" in out.columns:
        out = out[out["period_grain"].astype(str) == grain].copy()
        out["period_dt"] = period_to_dt(out["period"])
    elif "year_month" in out.columns:          # legacy aggregate
        out["period_dt"] = ym_to_dt(out["year_month"])
    else:
        return pd.DataFrame()
    return out.dropna(subset=["period_dt"])


def _color_seq(values: list[str], color_map: dict | None = None) -> list[str]:
    if color_map:
        return [color_map.get(v, PALETTE[i % len(PALETTE)])
                for i, v in enumerate(values)]
    return [PALETTE[i % len(PALETTE)] for i in range(len(values))]


def _filter_year(df: pd.DataFrame, year: int | str) -> pd.DataFrame:
    """
    SELECT the requested year grain — never aggregate into it.

    build_aggregates.py writes an explicit source_year == "ALL" row for every
    combination, counted from the raw claims and suppressed on its own value.
    Summing the yearly rows instead would drop every suppressed year and
    understate the total; for a low-volume variant where no single year clears
    the threshold, it would return nothing at all.
    """
    if df.empty or "source_year" not in df.columns:
        return df
    key = GRAIN_ALL if year == "ALL" else str(year)
    sub = df[df["source_year"].astype(str) == key]
    if sub.empty and year == "ALL":
        # Aggregate built before marginals existed: fall back to the yearly rows
        # so the page still renders, and let the caller's null-safe sums apply.
        return df[df["source_year"].astype(str) != GRAIN_ALL]
    return sub


def _filter_dimvalue_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the pre-computed all-demographic-values grain, deduplicated.

    build_aggregates builds its marginals inside a per-dimension loop, so the
    dimension_value == "ALL" block is written once for EVERY dimension (Age
    Band, Race/Ethnicity, Sex, ...) — identical numbers, only the `dimension`
    label differing. Collapsing a dimension's values gives the same grand total
    whichever dimension you collapsed, so the copies are redundant.

    Returning all of them draws one duplicate chart trace per dimension and
    multiplies any subsequent groupby-sum by the number of dimensions. Keeping
    exactly one copy is both correct and necessary.
    """
    if df.empty or "dimension_value" not in df.columns:
        return df
    sub = df[df["dimension_value"].astype(str) == GRAIN_ALL]
    if sub.empty:
        # Aggregate built before the marginals change — no ALL grain to select.
        return df[df["dimension_value"].astype(str) != GRAIN_ALL]
    if "dimension" in sub.columns:
        dims = sorted(sub["dimension"].dropna().astype(str).unique())
        if dims:
            sub = sub[sub["dimension"].astype(str) == dims[0]]
    return sub


def _drop_grain_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Remove ALL-grain rows so detail views don't double-count them."""
    out = df
    for c in cols:
        if c in out.columns:
            out = out[out[c].astype(str) != GRAIN_ALL]
    return out


def _available_years(df: pd.DataFrame) -> list[str]:
    """Real years only — the ALL grain is offered separately by the selector."""
    if df.empty or "source_year" not in df.columns:
        return []
    ys = df["source_year"].astype(str)
    ys = ys[ys != GRAIN_ALL]
    ys = pd.to_numeric(ys, errors="coerce").dropna().astype(int)
    return sorted(ys.unique().tolist())


def _covid_line(fig: go.Figure, row=None, col=None) -> None:
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_vline(
        x=pd.Timestamp("2020-03-01").timestamp() * 1000,
        line_width=1.5, line_dash="dash", line_color="red",
        annotation_text="COVID-19 PHE",
        annotation_font_color="red",
        annotation_font_size=10,
        annotation_position="top right",
        **kw,
    )


def _fmt_count(v) -> str:
    """
    Styler format function for integer count columns.

    Suppression now happens upstream, in build_aggregates.py — a suppressed cell
    arrives here as null. This renders null as '<11' only when the caller has
    already established the row was suppressed; use _mask_counts() for that.
    Standalone, it keeps a defensive threshold check so that an older parquet
    built before parquet-level suppression still can't leak a small count.
    """
    if pd.isna(v):
        return "—"
    try:
        n = float(v)
        if n < DISPLAY_SUPPRESS_THRESHOLD:
            return f"<{DISPLAY_SUPPRESS_THRESHOLD}"
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _suppress_series(series: pd.Series, flag: pd.Series | None = None) -> pd.Series:
    """
    Format a count column for display as object dtype.

      • flag is True            → '<11'  (suppressed upstream; value is null)
      • value is null, no flag  → '—'    (no data / not applicable)
      • value < threshold       → '<11'  (defensive: only reachable with a stale
                                          parquet built before suppression moved
                                          into build_aggregates.py)
      • otherwise               → formatted integer

    Returned as strings, not numbers, because Streamlit's grid reveals the raw
    underlying value when a cell is clicked, expanded or copied — Styler
    formatting alone is not sufficient there. Sort order on the column becomes
    lexicographic, which is the accepted trade-off.
    """
    lab = f"<{DISPLAY_SUPPRESS_THRESHOLD}"

    def _mask(v, is_supp):
        if bool(is_supp):
            return lab
        if pd.isna(v):
            return "—"
        try:
            n = float(v)
        except (TypeError, ValueError):
            return "—"
        return lab if n < DISPLAY_SUPPRESS_THRESHOLD else f"{int(n):,}"

    if flag is None:
        flag = pd.Series(False, index=series.index)
    else:
        flag = flag.reindex(series.index).fillna(False).astype(bool)
    return pd.Series([_mask(v, f) for v, f in zip(series, flag)],
                     index=series.index, dtype=object)


def _mask_rates(df: pd.DataFrame, cols: list[str],
                flag_src: pd.Series | None = None) -> pd.DataFrame:
    """
    Format rate/percentage columns so a suppressed row is visibly withheld
    rather than blank.

    build_aggregates nulls the rate alongside the count (otherwise the count is
    recoverable by arithmetic), so without this the cell renders as "—" and is
    indistinguishable from a cell that genuinely has no data. "<11" would be
    wrong on a percentage, so suppressed rates get their own mark.

    Returns object dtype, so apply Styler.format() to the numeric columns BEFORE
    calling this, not after.
    """
    out = df.copy()
    if flag_src is None:
        flag_src = _supp_flag(out)
    flag_src = flag_src.reindex(out.index).fillna(False).astype(bool)
    for c in cols:
        if c not in out.columns:
            continue
        vals = []
        for v, f in zip(out[c], flag_src):
            if bool(f):
                vals.append(SUPPRESS_RATE_MARK)
            elif pd.isna(v):
                vals.append(NO_DATA_MARK)
            else:
                vals.append(f"{float(v):.1f}%")
        out[c] = pd.Series(vals, index=out.index, dtype=object)
    return out


def _mask_counts(df: pd.DataFrame, cols: list[str],
                 flag_src: pd.Series | None = None) -> pd.DataFrame:
    """
    Apply _suppress_series to several count columns of a display frame at once,
    using the upstream `suppressed` flag when the aggregate carries one.
    Missing columns are ignored so this is safe to call on any display frame.
    """
    out = df.copy()
    if flag_src is None and SUPPRESS_FLAG_COL in out.columns:
        flag_src = out[SUPPRESS_FLAG_COL]
    for c in cols:
        if c in out.columns:
            out[c] = _suppress_series(out[c], flag_src)
    if SUPPRESS_FLAG_COL in out.columns:
        out = out.drop(columns=[SUPPRESS_FLAG_COL])
    return out


def _bar_count_label(series: pd.Series, flag: pd.Series | None = None) -> list[str]:
    """
    Bar chart text labels for count columns. Suppressed bars are labelled '<11';
    a suppressed value is null so the bar itself has no height to read off.
    """
    lab = f"<{DISPLAY_SUPPRESS_THRESHOLD}"
    if flag is None:
        flag = pd.Series(False, index=series.index)
    else:
        flag = flag.reindex(series.index).fillna(False).astype(bool)
    out = []
    for v, f in zip(series, flag):
        if bool(f):
            out.append(lab)
        elif pd.isna(v):
            out.append("")
        elif float(v) < DISPLAY_SUPPRESS_THRESHOLD:
            out.append(lab)
        else:
            out.append(f"{int(v):,.0f}")
    return out


def _common_layout(fig: go.Figure, title: str = "", height: int = 400) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        plot_bgcolor="white", paper_bgcolor="white",
        height=height,
        margin=dict(t=50 if title else 20, b=40, l=50, r=20),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1,
                    xanchor="right", yanchor="bottom"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)",
                     zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)",
                     zeroline=False)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

# ── Navigation first: page radio doesn't depend on any loaded data, so it can
# render before the variant is even resolved. Variant selector comes next
# since everything below it (Filters) depends on the data it loads. ─────────
with st.sidebar:
    st.markdown("**Dashboard View**")
    sel_variant = st.selectbox(
        "Telehealth Type",
        list(VARIANTS.keys()),
        index=0,
        help="Switch between Synchronous TH, Remote Patient Monitoring, and eConsults.",
    )
    
    st.divider()
    
    page = st.radio(
        "Dashboard View",
        ["📈 Utilization Trends", "📊 Demographics", "🏥 Diagnoses", "🗺️ County Map"],
        index=0,
        label_visibility="collapsed",
        key="page",
    )

    # st.divider()
    # sel_variant = st.selectbox(
    #     "Telehealth Type",
    #     list(VARIANTS.keys()),
    #     index=0,
    #     help="Switch between Synchronous TH, Remote Patient Monitoring, and eConsults.",
    # )

AGG_DIR       = VARIANTS[sel_variant]
VARIANT_LABEL = sel_variant
DATA          = load_all(str(AGG_DIR))
GEOJSON, FIPS_TO_NAME = load_geojson()
CTH = load_county_th_metrics(str(AGG_DIR))
# Every file the dashboard reads now comes from AGG_DIR — build_aggregates.py is
# the single source. split_payer_views collapses DATA to its all-payer rows and
# hands back the full frames for the payer-filtered pages.
PAYER = split_payer_views(DATA, CTH)

# Build sidebar option lists from the loaded data
period_df = DATA["period"]
demo_df   = DATA["demo"]

all_years  = _available_years(period_df)
all_payers = (
    sorted(period_df["Payer_Type"].dropna().unique().tolist())
    if "Payer_Type" in period_df.columns else []
)
all_payers_no_all = [p for p in all_payers if p != "ALL"]

all_dimensions = (
    sorted(demo_df["dimension"].dropna().unique().tolist())
    if "dimension" in demo_df.columns
    else ["Age Band", "Sex", "Race/Ethnicity", "Rural/Urban"]
)

all_th_types = (
    sorted(DATA["th_type"]["th_type"].dropna().unique().tolist())
    if "th_type" in DATA["th_type"].columns else []
)

# ── Remaining sidebar filters ─────────────────────────────────────────────────
with st.sidebar:
    # st.caption(f"`{AGG_DIR.name}`")
    st.divider()
    st.markdown("**Filters**")

    sel_year = st.selectbox(
        "Year",
        ["ALL"] + [str(y) for y in all_years],
        index=0,
        help="Applies to Utilization Trends, Demographics, and Diagnoses tabs.",
        key="sel_year",
    )

    # Payer Type only affects Utilization Trends — hidden everywhere else
    # rather than shown-but-inert, so it's never mistaken for an active filter
    # on tabs where it has no effect.
    if page == "📈 Utilization Trends":
        _payer_options = ["ALL"] + all_payers_no_all
        _payer_default = st.session_state.get("_persist_sel_payer", "ALL")
        _payer_index = _payer_options.index(_payer_default) if _payer_default in _payer_options else 0
        sel_payer = st.selectbox(
            "Payer Type",
            _payer_options,
            index=_payer_index,
            key="sel_payer",
        )
        st.session_state["_persist_sel_payer"] = sel_payer
    else:
        # Locked to ALL on every other tab — the filter genuinely has no
        # effect there, so nothing downstream should branch on a stale value.
        sel_payer = "ALL"

    # Single-select Payer Type — Demographics, Diagnoses, County Map. Backed by
    # the payer-stratified parquets from Part D of the notebook; ALL (or absent
    # parquets) uses the all-payer views unchanged.
    _payer_pages = {"📊 Demographics", "🏥 Diagnoses", "🗺️ County Map"}
    if page in _payer_pages and all_payers_no_all:
        _payer_opts = ["ALL"] + all_payers_no_all
        _payer_def  = st.session_state.get("_persist_sel_payer_single", "ALL")
        _payer_idx  = _payer_opts.index(_payer_def) if _payer_def in _payer_opts else 0
        sel_payer_single = st.selectbox(
            "Payer Type",
            _payer_opts,
            index=_payer_idx,
            key="sel_payer_single",
            help="Filter to a single payer, or ALL for every payer.",
        )
        st.session_state["_persist_sel_payer_single"] = sel_payer_single
    else:
        sel_payer_single = st.session_state.get("_persist_sel_payer_single", "ALL")

    if page == "📊 Demographics":
        _dim_default = st.session_state.get("_persist_sel_dimension", all_dimensions[0] if all_dimensions else None)
        _dim_index = all_dimensions.index(_dim_default) if _dim_default in all_dimensions else 0
        sel_dimension = st.selectbox(
            "Demographic Dimension",
            all_dimensions,
            index=_dim_index,
            help="Choose which demographic breakdown to show.",
            key="sel_dimension",
        )
        st.session_state["_persist_sel_dimension"] = sel_dimension
    else:
        # Widgets that aren't instantiated on a given run have their keyed
        # session_state entry cleared by Streamlit, so the last value must be
        # tracked in a separate, stable key that nothing else touches.
        sel_dimension = st.session_state.get("_persist_sel_dimension", all_dimensions[0] if all_dimensions else None)

    if page == "🗺️ County Map":
        # Sub-level under Filters (no divider) rather than its own top-level
        # section — visually nested as a continuation of the same group.
        st.caption("County Map")
        _metric_keys = list(COUNTY_MAP_METRICS.keys())
        _metric_default = st.session_state.get("_persist_sel_map_metric", "pct_of_state_patients")
        _metric_index = _metric_keys.index(_metric_default) if _metric_default in _metric_keys else 2
        sel_map_metric = st.selectbox(
            "Map Metric",
            _metric_keys,
            format_func=lambda k: COUNTY_MAP_METRICS[k]["label"],
            index=_metric_index,
            key="sel_map_metric",
        )
        st.session_state["_persist_sel_map_metric"] = sel_map_metric

        sel_state_fips = st.text_input(
            "State FIPS (optional)",
            value=st.session_state.get("_persist_sel_state_fips", "51"),
            max_chars=2,
            help="2-digit FIPS to zoom to a single state. 51 = Virginia. Leave blank for full US.",
            key="sel_state_fips",
        )
        st.session_state["_persist_sel_state_fips"] = sel_state_fips
    else:
        sel_map_metric = st.session_state.get("_persist_sel_map_metric", "pct_of_state_patients")
        sel_state_fips = st.session_state.get("_persist_sel_state_fips", "51")

    # ── Attribution (pinned to the bottom of the sidebar) ─────────────────────
    # Rendered last so it always sits below the filter stack regardless of which
    # tab-specific widgets are instantiated on a given run.
    st.markdown(
        """
        <div class="sidebar-attribution">
            <p>Virginia All-Payer Claims Database is administered by
            <strong><a href="https://www.vhi.org/data/all-payer-claims-database-data/">Virginia Health Information</a></strong>.</p>
            <p>This interactive dashboard is built and managed by <a href="https://datascience.virginia.edu/people/donald-brown">Dr. Brown's</a> lab,
            <strong>School of Data Science, University of Virginia</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Active payer (None → all payers / no filtering; else a single-payer subset).
payer_filter = None if sel_payer_single == "ALL" else [sel_payer_single]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HEADER + PAGE NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    f'<div class="main-header">🏥 {VARIANT_LABEL.upper()} — VIRGINIA APCD</div>',
    unsafe_allow_html=True,
)

# Fix 3 — navigation lives in the sidebar (see "page" radio above) so that
# tab-specific filters can appear/disappear in sync with the active view.
st.markdown(f"### {page}")

# Collapsed by default — costs one row of vertical space but keeps the
# methodology one click away on every tab. Variant-aware, so the code table
# always matches whatever is selected in the sidebar.
render_about(VARIANT_LABEL)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — TAB 1: UTILIZATION TRENDS
# ══════════════════════════════════════════════════════════════════════════════

if page == "📈 Utilization Trends":

    df_p = DATA["period"].copy()
    if df_p.empty:
        st.warning("util_by_period.parquet not found in AGG_DIR.")
        st.stop()

    # Convert YYYYMM → datetime for x-axis
    df_p["period_dt"] = ym_to_dt(df_p["year_month"])

    # Apply year filter to KPI numerics (not to the trend chart itself)
    df_p_yr = _filter_year(df_p, sel_year)

    # ── KPI row ──────────────────────────────────────────────────────────────
    kpi_all = df_p_yr[df_p_yr["Payer_Type"] == "ALL"] if "Payer_Type" in df_p_yr.columns else df_p_yr

    # Claims are additive across months, so summing the monthly rows is fine.
    # DISTINCT PEOPLE ARE NOT. Summing monthly th_patients counts a patient once
    # per month they appear — the "3 claim-months counted 3x" behaviour the old
    # help text described. util_by_year.parquet holds true distinct counts per
    # (year, payer) and for the cross-year ALL grain, computed in one DuckDB
    # pass via GROUPING SETS, so read those instead of re-deriving them here.
    # This is why the county map's Total TH Patients was right and this was not.
    _uy_kpi = DATA["year"]
    if not _uy_kpi.empty and "Payer_Type" in _uy_kpi.columns:
        _key = GRAIN_ALL if sel_year == "ALL" else str(sel_year)
        _uy_kpi = _uy_kpi[(_uy_kpi["Payer_Type"] == "ALL") &
                          (_uy_kpi["source_year"].astype(str) == _key)]
    else:
        _uy_kpi = pd.DataFrame()

    if not _uy_kpi.empty:
        _r = _uy_kpi.iloc[0]
        total_claims   = int(_r["th_claims"])   if pd.notna(_r.get("th_claims"))   else 0
        total_patients = int(_r["th_patients"]) if pd.notna(_r.get("th_patients")) else 0
        total_denom    = _r["n_claimants"]      if pd.notna(_r.get("n_claimants")) else 0
        _patients_exact = True
    else:
        # util_by_year absent (older aggregate) — fall back to the monthly sums
        # and say plainly that the patient count is inflated.
        total_claims   = int(kpi_all["th_claims"].sum())   if "th_claims"   in kpi_all.columns else 0
        total_patients = int(kpi_all["th_patients"].sum()) if "th_patients" in kpi_all.columns else 0
        total_denom    = kpi_all["n_claimants"].sum()      if "n_claimants" in kpi_all.columns else 0
        _patients_exact = False

    _ucol          = util_col(kpi_all)
    peak_util      = kpi_all[_ucol].max() if _ucol else 0

    yr_label = sel_year if sel_year != "ALL" else "2018–2023"
    st.markdown("""
        <style>
        [data-testid="stMetricLabel"] p {
            font-size: 24px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

        
    k1.metric('Telehealth (TH) Claims',             f"{total_claims:,}",              help=f"{yr_label}")
    k2.metric('Unique TH Patients',    f"{total_patients:,}",
              help=("Distinct people over the whole period — a patient with "
                    "claims in several months is counted once."
                    if _patients_exact else
                    "APPROXIMATE: util_by_year.parquet not found, so this is a "
                    "sum of monthly distinct counts and over-counts anyone with "
                    "claims in more than one month. Re-run build_aggregates.py."))
    k3.metric('Claimants (denom)',     f"{total_denom:,.0f}",
              help="Distinct people with any facility claim in the period")
    k4.metric('Peak Util / 1K',        f"{peak_util:.2f}",
              help="Peak monthly telehealth visits per 1,000 claimants that month")

    st.divider()

    # ── Monthly util/1K member months ────────────────────────────────────────
    st.markdown(f'<p style="font-size: 20px;" class="sub-header">Monthly Telehealth {UTIL_LABEL}</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-size: 16px;" class="sub-caption">Overall · dashed line = 3-month rolling average · vertical = COVID-19 PHE (Mar 2020)</p>',
                unsafe_allow_html=True)

    trend_fig = go.Figure()

    # Payer filter: if ALL selected show "ALL" trace; otherwise show the selected payer
    show_payers = ["ALL"] if sel_payer == "ALL" else [sel_payer, "ALL"]
    _any_supp_util = False
    for payer in show_payers:
        sub = df_p[df_p["Payer_Type"] == payer].sort_values("period_dt")
        if sub.empty:
            continue
        col = PAYER_COLORS.get(payer, PALETTE[0])
        lw  = 2.5 if payer == "ALL" else 1.8
        _ucol = util_col(sub) or UTIL_COL
        trend_fig.add_trace(go.Scatter(
            x=sub["period_dt"], y=sub[_ucol],
            name=payer, mode="lines",
            line=dict(color=col, width=lw),
            connectgaps=False,
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>" + payer +
                "<br>Util / 1K: %{y:.2f}<br>"
                "TH Claims: %{customdata[0]}<br>"
                "Claimants: %{customdata[1]:,.0f}<extra></extra>"
            ),
            customdata=pd.DataFrame({
                "c": sub["th_claims"].apply(_fmt_count),
                "m": sub["n_claimants"].fillna(0).astype(int)
                     if "n_claimants" in sub.columns
                     else pd.Series(0, index=sub.index, dtype=int),
            }).values,
        ))
        # connectgaps=False makes a suppressed month break the line, which is
        # right — but on its own it is indistinguishable from a month with no
        # data at all. Mark the withheld months explicitly, matching the
        # demographic trend's convention, so a long gap can be read as
        # "consistently below the threshold" rather than "nothing reported".
        _sup = sub[_supp_flag(sub) & sub[_ucol].isna()]
        if not _sup.empty:
            _any_supp_util = True
            trend_fig.add_trace(go.Scatter(
                x=_sup["period_dt"], y=[0] * len(_sup),
                mode="markers", name=f"{payer} (suppressed)",
                marker=dict(color=col, size=7, symbol="circle-open",
                            line=dict(width=1.5)),
                showlegend=False,
                hovertemplate=("<b>%{x|%b %Y}</b><br>" + SUPPRESS_MARK +
                               " TH claims — withheld<extra>" + payer + "</extra>"),
            ))
        if "rolling_util_per_1k" in sub.columns:
            trend_fig.add_trace(go.Scatter(
                x=sub["period_dt"], y=sub["rolling_util_per_1k"],
                name=f"{payer} ({ROLLING_WINDOW}m avg)", mode="lines",
                line=dict(color=col, width=1.5, dash="dot"),
                connectgaps=False, hoverinfo="skip", showlegend=False,
            ))

    _covid_line(trend_fig)
    _common_layout(trend_fig, height=380)
    trend_fig.update_yaxes(title_text="Visits / 1,000 Claimants", rangemode="tozero")
    trend_fig.update_xaxes(tickformat="%b %Y")
    st.plotly_chart(trend_fig, width='stretch')
    if _any_supp_util:
        st.caption(
            f"Open circles mark months where the count was suppressed "
            f"({SUPPRESS_MARK}); the line breaks there rather than dropping to "
            f"zero. A gap with no circles means no claims were reported.")

    # ── TH type stacked-area + annual bar ────────────────────────────────────
    col_type, col_annual = st.columns([3, 2])

    with col_type:
        st.markdown('<p style="font-size: 20px;" class="sub-header">Telehealth Delivery Type — Composition over Time</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="font-size: 16px;" class="sub-caption">Stacked area — Video / Audio-Only / Facility / Unclassified</p>',
                    unsafe_allow_html=True)

        df_tht = DATA["th_type"].copy()
        if not df_tht.empty and "th_type" in df_tht.columns:
            df_tht["period_dt"] = ym_to_dt(df_tht["year_month"])
            type_fig = go.Figure()
            for tht in sorted(df_tht["th_type"].dropna().unique()):
                sub = df_tht[df_tht["th_type"] == tht].sort_values("period_dt")
                col = TH_TYPE_COLORS.get(tht, PALETTE[0])
                type_fig.add_trace(go.Scatter(
                    x=sub["period_dt"], y=sub[util_col(sub) or UTIL_COL],
                    name=tht, fill="tonexty", mode="none",
                    fillcolor=col, stackgroup="one", connectgaps=False,
                    hovertemplate="<b>%{x|%b %Y}</b><br>%{y:.3f}/1K<extra>" + tht + "</extra>",
                ))
            _covid_line(type_fig)
            _common_layout(type_fig, height=320)
            type_fig.update_yaxes(title_text="Visits / 1,000 Claimants (stacked)")
            type_fig.update_xaxes(tickformat="%b %Y")
            type_fig.update_layout(showlegend=True,
                legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"))
            st.plotly_chart(type_fig, width='stretch')
        else:
            st.info("util_by_th_type.parquet not available.")

    with col_annual:
        st.markdown('<p style="font-size: 20px;" class="sub-header">Annual TH Claims — All Payers</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="font-size: 16px;" class="sub-caption">Total claims per year</p>',
                    unsafe_allow_html=True)

        if not df_p.empty and "Payer_Type" in df_p.columns:
            # Read the annual grain directly. Summing the monthly rows would
            # drop every suppressed month and understate each year.
            _uy = DATA["year"]
            if not _uy.empty and "Payer_Type" in _uy.columns:
                annual = (
                    _uy[(_uy["Payer_Type"] == "ALL")
                        & (_uy["source_year"].astype(str) != GRAIN_ALL)]
                    .rename(columns={util_col(_uy) or UTIL_COL: "util_avg"})
                    .sort_values("source_year")
                )
            else:
                annual = (
                    df_p[df_p["Payer_Type"] == "ALL"]
                    .groupby("source_year", as_index=False)
                    .agg(th_claims=("th_claims", lambda x: x.sum(min_count=1)),
                         util_avg=(util_col(df_p) or UTIL_COL, "mean"))
                    .sort_values("source_year")
                )
            annual_fig = go.Figure(go.Bar(
                x=annual["source_year"].astype(str),
                y=annual["th_claims"],
                marker_color=PALETTE[0], opacity=0.85,
                text=_bar_count_label(annual["th_claims"]),
                textposition="outside",
                customdata=annual["th_claims"].apply(_fmt_count).values,
                hovertemplate="<b>%{x}</b><br>TH Claims: %{customdata}<extra></extra>",
            ))
            annual_fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=320, margin=dict(t=20, b=40, l=50, r=20),
                yaxis=dict(title="TH Claims", rangemode="tozero",
                           tickformat=",", showgrid=True, gridcolor="#EEE"),
                xaxis=dict(title="Year", type="category"),
                bargap=0.35,
            )
            st.plotly_chart(annual_fig, width='stretch')

    # ── Underlying data table ─────────────────────────────────────────────────
    with st.expander("📋 View underlying data — monthly utilization"):
        show = (
            df_p[df_p["Payer_Type"] == ("ALL" if sel_payer == "ALL" else sel_payer)]
            .sort_values(["source_year", "year_month"])
            [[c for c in ["period_dt", "source_year", "Payer_Type",
                           "th_claims", "th_patients", "n_claimants",
                           UTIL_COL, UTIL_COL_LEGACY, "rolling_util_per_1k",
                           SUPPRESS_FLAG_COL]
               if c in df_p.columns]]
            .rename(columns={
                "period_dt"                  : "Month",
                "source_year"                : "Year",
                "Payer_Type"                 : "Payer",
                "th_claims"                  : "TH Claims",
                "th_patients"                : "TH Patients",
                "n_claimants"                : "Claimants",
                UTIL_COL                     : UTIL_LABEL_SHORT,
                UTIL_COL_LEGACY              : UTIL_LABEL_SHORT,
                "rolling_util_per_1k"        : f"{ROLLING_WINDOW}m Avg",
            })
        )
        show["Month"] = pd.to_datetime(show["Month"]).dt.strftime("%Y-%m")
        show = _mask_counts(show, ["TH Claims", "TH Patients"])
        st.dataframe(
            show.style.format({
                "Claimants"     : "{:,.0f}",
                UTIL_LABEL_SHORT: "{:.2f}",
                f"{ROLLING_WINDOW}m Avg" : "{:.3f}",
            }, na_rep="—"),
            width='stretch', height=300,
        )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TAB 2: DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Demographics":

    df_d  = DATA["demo"].copy()
    df_dp = DATA["demo_period"].copy()

    # Payer subset selected → rebuild from the payer-stratified parquets.
    if payer_filter is not None and not PAYER["demo"].empty:
        _dp_sub = _demo_by_payer(PAYER["demo"], payer_filter)
        if not _dp_sub.empty:
            df_d = _dp_sub
        _dpp_sub = _demo_period_by_payer(PAYER["demo_period"], payer_filter)
        if not _dpp_sub.empty:
            df_dp = _dpp_sub
        st.caption(f"**Payer filter: {', '.join(payer_filter)}**")
    elif payer_filter is not None and PAYER["demo"].empty:
        st.info("Payer breakdown not available for this variant — run Part D of the "
                "notebook to generate `util_by_demo_payer.parquet`. Showing all payers.")

    if df_d.empty:
        st.warning("util_by_demo.parquet not found.")
        st.stop()

    # ── Dimension × year slice ────────────────────────────────────────────────
    dim_slice = df_d[df_d["dimension"] == sel_dimension].copy() if "dimension" in df_d.columns else df_d
    dim_slice = _filter_year(dim_slice, sel_year)

    # Always recompute percentages from the filtered slice so they reflect the
    # selected year (or all-years aggregate) rather than the pre-computed
    # parquet values, which aren't recalculated when the year widget changes.
    if not dim_slice.empty:
        # _filter_year already selected one grain, so there is a single row per
        # dimension_value. No aggregation, and none wanted: pct_patients was
        # computed upstream against the true denominator rather than against a
        # sum of possibly-suppressed sibling cells.
        totals = _drop_grain_rows(dim_slice, ["dimension_value"]).copy()
        if "pct_patients" not in totals.columns:
            _yt = totals["th_patients"].sum(min_count=1)
            totals["pct_patients"] = (totals["th_patients"] / _yt * 100
                                      if pd.notna(_yt) and _yt else np.nan)
    else:
        totals = pd.DataFrame()
    yr_label_d = "All Years" if sel_year == "ALL" else sel_year

    if totals.empty:
        st.info(f"No demographic data for dimension '{sel_dimension}' — year {sel_year}.")
        st.stop()

    # Colour map for dimension values
    dim_vals   = totals["dimension_value"].dropna().astype(str).unique().tolist()
    dim_colors = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(sorted(dim_vals))}

    # ── Row 1: % bar + trend ─────────────────────────────────────────────────
    col_pct, col_trend_d = st.columns([1, 2])

    with col_pct:
        st.markdown(f'<p style="font-size: 20px;" class="sub-header">% of TH Patients — {sel_dimension}</p>',
                    unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 16px;" class="sub-caption">{yr_label_d}</p>', unsafe_allow_html=True)

        # Sort suppressed rows to the bottom (NaN last) so they don't disturb
        # the ranking of the values that are actually published.
        pct_sorted = totals.sort_values("pct_patients", ascending=True,
                                        na_position="first")
        cats = pct_sorted["dimension_value"].astype(str).tolist()
        _flag = _supp_flag(pct_sorted)
        _x, _txt, _cols, _pats = _bar_display(
            pct_sorted["pct_patients"], _flag,
            [dim_colors.get(v, PALETTE[0]) for v in cats])
        pct_fig = go.Figure(go.Bar(
            y=cats,
            x=_x,
            orientation="h",
            marker=dict(color=_cols,
                        pattern=dict(shape=_pats, fgcolor="#9E9E9E", size=4)),
            text=_txt,
            textposition="outside",
            showlegend=False,
            customdata=[SUPPRESS_MARK if f else "" for f in _flag],
            hovertemplate=("<b>%{y}</b><br>"
                           "%{customdata}%{x:.1f}% of TH patients<extra></extra>"),
        ))
        pct_fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(280, len(dim_vals) * 52),
            margin=dict(t=10, b=10, l=10, r=60),
            # max() of an all-null column is NaN, which produces a broken axis.
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=[0, (float(pct_sorted["pct_patients"].max()) * 1.35)
                                 if pct_sorted["pct_patients"].notna().any() else 1]),
            yaxis=dict(showgrid=False, type="category",
                       categoryorder="array", categoryarray=cats),
        )
        st.plotly_chart(pct_fig, width='stretch')
        if bool(_flag.any()):
            st.caption(SUPPRESS_LEGEND)

    with col_trend_d:
        st.markdown(f'<p style="font-size: 20px;" class="sub-header">TH Patient Trend — {sel_dimension}</p>',
                    unsafe_allow_html=True)
        _dim_rows_pre = (df_dp[df_dp["dimension"] == sel_dimension]
                         if not df_dp.empty and "dimension" in df_dp.columns
                         else pd.DataFrame())
        _grain_pre = pick_period_grain(_dim_rows_pre, preferred="month")
        st.markdown(
            f'<p class="sub-caption">'
            f'{PERIOD_GRAIN_LABEL.get(_grain_pre, "Monthly")} unique TH patients '
            f'per demographic group</p>',
            unsafe_allow_html=True)
        if _grain_pre != "month":
            st.caption(
                f"Showing the {PERIOD_GRAIN_LABEL.get(_grain_pre, '').lower()} "
                f"grain: too many monthly cells fall below "
                f"{DISPLAY_SUPPRESS_THRESHOLD} to plot a monthly trend for this "
                f"variant.")

        if not df_dp.empty and "dimension" in df_dp.columns:
            _dim_rows = df_dp[df_dp["dimension"] == sel_dimension].copy()
            # Pick the finest period grain that isn't mostly suppressed.
            _grain = pick_period_grain(_dim_rows, preferred="month")
            dp_slice = prep_period_frame(_dim_rows, _grain)

            if not dp_slice.empty:
                full_spine = pd.DataFrame({
                    "period_dt": pd.date_range(
                        dp_slice["period_dt"].min(),
                        dp_slice["period_dt"].max(),
                        freq=PERIOD_GRAIN_FREQ.get(_grain, "MS"),
                    )
                })
                trend_d_fig = go.Figure()
                _any_supp_trend = False
                _keep = [c for c in ["period_dt", "th_patients", SUPPRESS_FLAG_COL]
                         if c in dp_slice.columns]
                for val in sorted(dp_slice["dimension_value"].dropna().unique()):
                    sub = (dp_slice[dp_slice["dimension_value"] == val][_keep]
                           .drop_duplicates("period_dt"))
                    sub = full_spine.merge(sub, on="period_dt", how="left").sort_values("period_dt")
                    col = dim_colors.get(str(val), PALETTE[0])
                    trend_d_fig.add_trace(go.Scatter(
                        x=sub["period_dt"], y=sub["th_patients"],
                        name=str(val), mode="lines",
                        line=dict(color=col, width=2),
                        connectgaps=False,
                        hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,.0f} TH patients<extra>" + str(val) + "</extra>",
                    ))
                    # A suppressed period is a null, so the line just breaks —
                    # visually identical to "no telehealth", which is the
                    # opposite of the truth. Mark those periods explicitly at
                    # y=0 with an open symbol so the gap reads as withheld.
                    _sflag = _supp_flag(sub)
                    if bool(_sflag.any()):
                        _any_supp_trend = True
                        _sx = sub.loc[_sflag, "period_dt"]
                        trend_d_fig.add_trace(go.Scatter(
                            x=_sx, y=[0] * len(_sx),
                            mode="markers", name=f"{val} (suppressed)",
                            marker=dict(color=col, size=7, symbol="circle-open",
                                        line=dict(width=1.5)),
                            showlegend=False,
                            hovertemplate=("<b>%{x|%b %Y}</b><br>"
                                           + SUPPRESS_MARK + " TH patients — "
                                           "withheld<extra>" + str(val) + "</extra>"),
                        ))
                _covid_line(trend_d_fig)
                _common_layout(trend_d_fig, height=max(300, len(dim_vals) * 52))
                _per_word = {"month": "Month", "quarter": "Quarter",
                             "year": "Year"}.get(_grain, "Month")
                trend_d_fig.update_yaxes(
                    title_text=f"Unique TH Patients / {_per_word}", rangemode="tozero")
                trend_d_fig.update_xaxes(
                    tickformat="%Y" if _grain == "year" else "%b %Y")
                trend_d_fig.update_layout(
                    legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"))
                st.plotly_chart(trend_d_fig, width='stretch')
                if _any_supp_trend:
                    st.caption(
                        f"Open circles mark periods where the count was "
                        f"suppressed ({SUPPRESS_MARK}); the line breaks there "
                        f"rather than dropping to zero.")
            else:
                st.info("No trend data available for this dimension.")
        else:
            st.info("util_by_demo_period.parquet not available.")

    # ── Row 2: all-years comparison grouped bar ───────────────────────────────
    st.divider()
    st.markdown(f'<p style="font-size: 20px;" class="sub-header">All-Years Comparison — {sel_dimension}</p>',
                unsafe_allow_html=True)
    st.markdown('<p style="font-size: 16px;" class="sub-caption">% of TH patients by year — years on y-axis, categories as grouped bars</p>',
                unsafe_allow_html=True)

    df_d_all = df_d[df_d["dimension"] == sel_dimension].copy() if "dimension" in df_d.columns else df_d
    # This chart plots one bar per real year, so the ALL-grain rows are dropped
    # from both axes — they are a separate view, not another category.
    df_d_all = _drop_grain_rows(df_d_all, ["source_year", "dimension_value"])
    if not df_d_all.empty:
        yrs_all   = _available_years(df_d_all)
        cats_all  = sorted(df_d_all["dimension_value"].dropna().astype(str).unique())
        n_cats    = len(cats_all)
        bar_h     = max(0.1, 0.8 / n_cats) if n_cats > 0 else 0.4

        all_yr_fig = go.Figure()
        for i, cat in enumerate(cats_all):
            col = PALETTE[i % len(PALETTE)]
            vals, texts, colors, patterns, hovers = [], [], [], [], []
            for yr in yrs_all:
                # source_year is a STRING column now — it has to hold the "ALL"
                # sentinel — while _available_years() returns ints. Comparing the
                # two directly never matches, so every bar silently became 0.0
                # and the chart rendered with correct axes but no data.
                row = df_d_all[(df_d_all["source_year"].astype(str) == str(yr)) &
                               (df_d_all["dimension_value"].astype(str) == cat)]
                if row.empty:
                    # No row at all for this year/category: genuinely absent.
                    vals.append(0.0); texts.append("")
                    colors.append(col); patterns.append("")
                    hovers.append("No data")
                    continue
                _v    = row["pct_patients"].values[0]
                _supp = bool(_supp_flag(row).values[0])
                if _supp or pd.isna(_v):
                    # A suppressed cell used to be coerced to 0.0 with an empty
                    # label, so it drew nothing and read as "this age band had no
                    # telehealth that year" — the opposite of the truth, which is
                    # that it had between 1 and 10 patients. Give it the same
                    # grey hatched zero-length treatment as every other chart so
                    # the category keeps its slot and is visibly withheld.
                    vals.append(0.0)
                    texts.append(SUPPRESS_MARK if _supp else "")
                    # colors.append(SUPPRESS_COLOR)
                    patterns.append("/" if _supp else "")
                    hovers.append(f"Withheld (fewer than "
                                  f"{DISPLAY_SUPPRESS_THRESHOLD} patients)"
                                  if _supp else "No data")
                else:
                    vals.append(float(_v)); texts.append(f"{float(_v):.1f}%")
                    # colors.append(col)
                    patterns.append("")
                    hovers.append(f"{float(_v):.1f}% of TH patients")
            all_yr_fig.add_trace(go.Bar(
                y=[str(y) for y in yrs_all],
                x=vals,
                name=str(cat),
                orientation="h",
                marker=dict(color=col, opacity=0.85,
                            pattern=dict(shape=patterns, solidity=0.35,
                                         fgcolor="white", size=12)),
                text=texts,
                textposition="outside",
                textfont=dict(size=9),
                customdata=hovers,
                hovertemplate="<b>%{y}</b><br>%{customdata}"
                              "<extra>" + str(cat) + "</extra>",
                showlegend=True,
            ))
            # # Legend-only proxy trace: draws nothing, but its single solid
            # # `col` guarantees a correct, stable swatch — the real trace's
            # # marker.color array can start with SUPPRESS_COLOR (grey) when
            # # the first year plotted for this category is suppressed, and
            # # Plotly always takes the legend swatch from colors[0].
            # all_yr_fig.add_trace(go.Bar(
            #     y=[None], x=[None],
            #     name=str(cat),
            #     orientation="h",
            #     marker=dict(color=col, opacity=0.85),
            #     showlegend=True,
            #     hoverinfo="skip",
            # ))
        all_yr_fig.update_layout(
            barmode="group",
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(300, len(yrs_all) * n_cats * 22 + 80),
            margin=dict(t=20, b=40, l=60, r=60),
            xaxis=dict(title="% of TH Patients", showgrid=True, gridcolor="#EEE",
                       ticksuffix="%", zeroline=False),
            yaxis=dict(title="Year", type="category",
                       categoryorder="array",
                       categoryarray=[str(y) for y in reversed(yrs_all)]),
            legend=dict(title=sel_dimension, orientation="v",
                        yanchor="top", y=1, xanchor="left", x=1.01),
            bargroupgap=0.05,
        )
        st.plotly_chart(all_yr_fig, width='stretch')

    # ── Underlying data ────────────────────────────────────────────────────────
    with st.expander("📋 View underlying data — demographics breakdown"):
        show_d = (
            df_d_all[[c for c in ["source_year", "dimension_value", "th_patients",
                                  "th_claims", "pct_patients", "pct_claims",
                                  SUPPRESS_FLAG_COL]
                      if c in df_d_all.columns]]
            .sort_values(["source_year", "pct_patients"], ascending=[True, False])
            .rename(columns={
                "source_year"      : "Year",
                "dimension_value"  : sel_dimension,
                "th_patients"      : "TH Patients",
                "th_claims"        : "TH Claims",
                "pct_patients"     : "% Patients",
                "pct_claims"       : "% Claims",
            })
        )
        _flag_d = _supp_flag(show_d)
        show_d = _mask_rates(show_d,  ["% Patients", "% Claims"], _flag_d)
        show_d = _mask_counts(show_d, ["TH Patients", "TH Claims"], _flag_d)
        st.dataframe(show_d, width='stretch', height=320)
        if bool(_flag_d.any()):
            st.caption(SUPPRESS_LEGEND)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — TAB 3: DIAGNOSES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏥 Diagnoses":

    df_dx    = DATA["top_dx"].copy()
    df_mh_dx = DATA["top_mh_dx"].copy()

    # Payer subset selected → rebuild top_dx / top_mh_dx from payer parquets.
    if payer_filter is not None and (not PAYER["top_dx"].empty or not PAYER["top_mh_dx"].empty):
        _dx_sub = _dx_by_payer(PAYER["top_dx"], payer_filter)
        if not _dx_sub.empty:
            df_dx = _dx_sub
        _mh_sub = _dx_by_payer(PAYER["top_mh_dx"], payer_filter)
        if not _mh_sub.empty:
            df_mh_dx = _mh_sub
        st.caption(f"Payer filter: {', '.join(payer_filter)}")
    elif payer_filter is not None:
        st.info("Payer breakdown not available for this variant — run Part D of the "
                "notebook to generate `top_dx_payer.parquet`. Showing all payers.")

    dx_yr_label = sel_year if sel_year != "ALL" else "All Years"

    # ── CCS diagnoses ─────────────────────────────────────────────────────────
    st.markdown('<p style="font-size: 20px;" class="sub-header">Top Diagnoses (CCS Level 1) Treated via Telehealth</p>',
                unsafe_allow_html=True)
    st.markdown(f'<p style="font-size: 16px;" class="sub-caption">Year: {dx_yr_label}</p>',
                unsafe_allow_html=True)

    if df_dx.empty:
        st.info("top_dx.parquet not available.")
    else:
        # Roll up all dimension values to get overall top-N
        dx_filter = df_dx[df_dx["diagnosis_level"] == "CCS Level 1"].copy()
        dx_filter = _filter_year(dx_filter, sel_year)
        # Select the pre-built all-demographic-values grain rather than summing
        # across demographics: a diagnosis suppressed in every individual
        # demographic cell still has a publishable overall total.
        dx_agg = (
            _filter_dimvalue_all(dx_filter)
            .groupby("diagnosis", as_index=False)
            .agg(th_patients=("th_patients", lambda x: x.sum(min_count=1)),
                 th_claims  =("th_claims",   lambda x: x.sum(min_count=1)),
                 **{SUPPRESS_FLAG_COL: ("th_patients", lambda x: x.isna().any())})
            .sort_values("th_patients", ascending=False)
            .head(15)
        )
        total_dx_pts = dx_agg["th_patients"].sum()
        dx_agg["pct"] = dx_agg["th_patients"] / total_dx_pts * 100 if total_dx_pts else 0
        # Suppressed rows have a null pct. In a horizontal Plotly bar the FIRST
        # element of categoryarray is drawn at the BOTTOM, so the suppressed
        # rows have to come first in the frame to appear at the bottom — sorting
        # the flag ascending puts them last in the array, i.e. at the top, above
        # the largest real bar. Flag descending (True first) then pct ascending
        # gives: withheld at the bottom, then smallest to largest going up.
        dx_agg = dx_agg.sort_values(
            [SUPPRESS_FLAG_COL, "pct"], ascending=[False, True],
            na_position="first")

        _x, _txt, _cols, _pats = _bar_display(
            dx_agg["pct"], _supp_flag(dx_agg),
            base_colors=[PALETTE[1]] * len(dx_agg))

        dx_fig = go.Figure(go.Bar(
            y=dx_agg["diagnosis"].astype(str),
            x=_x,
            orientation="h",
            marker=dict(color=_cols, opacity=0.85,
                        pattern=dict(shape=_pats, solidity=0.35,
                                     fgcolor="white", size=6)),
            text=_txt,
            textposition="outside",
            customdata=_supp_flag(dx_agg).map(
                {True: "Withheld (fewer than 11 patients)", False: ""}),
            hovertemplate="<b>%{y}</b><br>%{text} of TH patients"
                          "%{customdata}<extra></extra>",
        ))
        _xmax = pd.to_numeric(dx_agg["pct"], errors="coerce").max()
        dx_fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            height=max(350, len(dx_agg) * 38 + 60),
            margin=dict(t=10, b=10, l=10, r=60),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       range=[0, (_xmax * 1.35) if pd.notna(_xmax) and _xmax > 0 else 1]),
            yaxis=dict(showgrid=False, type="category",
                       categoryorder="array",
                       categoryarray=dx_agg["diagnosis"].astype(str).tolist()),
        )
        st.plotly_chart(dx_fig, width='stretch')

    st.divider()

    # ── Mental-health diagnoses ───────────────────────────────────────────────
    col_mh_trend, col_mh_bar = st.columns([3, 2])

    with col_mh_trend:
        st.markdown('<p style="font-size: 20px;" class="sub-header">Mental Health Diagnosis Trends</p>',
                    unsafe_allow_html=True)
        st.markdown('<p style="font-size: 16px;" class="sub-caption">ICD-10 F-chapter · % of enrolled members</p>',
                    unsafe_allow_html=True)

        if df_mh_dx.empty:
            st.info("top_mh_dx.parquet not available.")
        else:
            df_denom = DATA["denom"].copy()
            # Keep the "% of enrolled" denominator consistent with the payer
            # filter when enroll_denom carries a payer column.
            if payer_filter is not None:
                _pcol = next((c for c in ("Payer_Type", "payer") if c in df_denom.columns), None)
                if _pcol:
                    df_denom = df_denom[df_denom[_pcol].isin(payer_filter)]
            annual_enrolled = (
                df_denom.groupby("source_year", as_index=False)
                ["total_enrolled_persons"].sum()
                if not df_denom.empty and "total_enrolled_persons" in df_denom.columns
                else pd.DataFrame(columns=["source_year", "total_enrolled_persons"])
            )

            _mh_all = _filter_dimvalue_all(
                df_mh_dx[df_mh_dx["diagnosis_level"] == "ICD-10 F-Chapter"])
            _mh_rank = _mh_all[_mh_all["source_year"].astype(str) == GRAIN_ALL]
            if _mh_rank.empty:
                _mh_rank = _mh_all
            # Collapse to one row per diagnosis BEFORE ranking, and dedupe the
            # label list. head(6) over un-collapsed rows can return the same
            # diagnosis several times, and the trace loop below adds one trace
            # per label — which is what produced four identical legend entries.
            _mh_rank = (_mh_rank.groupby("diagnosis", as_index=False)
                        ["th_patients"].max())
            top_mh_labels = list(dict.fromkeys(
                _mh_rank.sort_values("th_patients", ascending=False)
                .head(6)["diagnosis"].tolist()))
            _mh_sel = _filter_dimvalue_all(
                df_mh_dx[
                    (df_mh_dx["diagnosis_level"] == "ICD-10 F-Chapter") &
                    (df_mh_dx["diagnosis"].isin(top_mh_labels))
                ])
            mh_trend_data = (
                _drop_grain_rows(_mh_sel, ["source_year"])
                .groupby(["source_year", "diagnosis"], as_index=False)
                # The flag has to survive this aggregation, or the chart below
                # cannot tell a withheld year from a year with no claims: both
                # arrive as a null th_patients.
                .agg(th_patients=("th_patients", lambda x: x.sum(min_count=1)),
                     **{SUPPRESS_FLAG_COL: ("th_patients", lambda x: x.isna().any())})
                .merge(annual_enrolled, on="source_year", how="left")
            )
            mh_trend_data["rate_pct"] = (
                mh_trend_data["th_patients"] /
                mh_trend_data["total_enrolled_persons"].replace(0, np.nan) * 100
            )
            mh_fig = go.Figure()
            # With type="category" plotly orders the axis by first appearance
            # across traces, so a diagnosis whose first year is 2023 puts 2023
            # left of 2021. Pin the order explicitly.
            _mh_years = sorted(
                mh_trend_data["source_year"].dropna().astype(str).unique(),
                key=lambda s: (len(s), s))
            _any_point = False
            for i, diag in enumerate(top_mh_labels):
                sub = (mh_trend_data[mh_trend_data["diagnosis"] == diag]
                       .assign(_yr=lambda d: d["source_year"].astype(str))
                       .sort_values("_yr"))
                col = PALETTE[i % len(PALETTE)]
                if sub["rate_pct"].notna().any():
                    _any_point = True
                mh_fig.add_trace(go.Scatter(
                    x=sub["_yr"], y=sub["rate_pct"],
                    name=diag, mode="lines+markers",
                    line=dict(color=col, width=2.2), marker=dict(size=6, color=col),
                    hovertemplate="<b>%{x}</b><br>%{y:.3f}% of enrolled<extra>" + diag + "</extra>",
                ))
                # A suppressed year is a null, so the line simply breaks and the
                # reader cannot tell a withheld point from a year the diagnosis
                # was absent. Mark the withheld years with open circles on the
                # zero line, matching the utilization trend's convention.
                _s = sub[_supp_flag(sub) & sub["rate_pct"].isna()]
                if not _s.empty:
                    mh_fig.add_trace(go.Scatter(
                        x=_s["_yr"], y=[0] * len(_s),
                        name=f"{diag} (suppressed)", mode="markers",
                        marker=dict(size=7, color="white", line=dict(color=col, width=1.5)),
                        showlegend=False,
                        hovertemplate="<b>%{x}</b><br>Withheld (fewer than "
                                      f"{DISPLAY_SUPPRESS_THRESHOLD} patients)"
                                      "<extra>" + diag + "</extra>",
                    ))
            _common_layout(mh_fig, height=360)
            mh_fig.update_yaxes(title_text="% of Enrolled Members")
            mh_fig.update_xaxes(title_text="Year", type="category",
                                categoryorder="array", categoryarray=_mh_years)
            mh_fig.update_layout(
                legend=dict(orientation="v", yanchor="top", y=1,
                            xanchor="left", x=1.01, font=dict(size=9)))
            if not _any_point:
                # Every point suppressed. An empty axis with a legend reads as a
                # rendering failure; say what actually happened instead.
                st.info(
                    f"Mental-health counts are below the disclosure threshold "
                    f"({SUPPRESS_MARK} patients) in every year for this "
                    f"selection, so no trend can be shown. Try the All Years "
                    f"view or a broader filter.")
            else:
                st.plotly_chart(mh_fig, width='stretch')

    with col_mh_bar:
        st.markdown('<p style="font-size: 20px;" class="sub-header">Top MH Diagnoses</p>',
                    unsafe_allow_html=True)
        st.markdown(f'<p style="font-size: 16px;" class="sub-caption">Year: {dx_yr_label}</p>',
                    unsafe_allow_html=True)

        if not df_mh_dx.empty:
            mh_yr = _filter_year(
                df_mh_dx[df_mh_dx["diagnosis_level"] == "ICD-10 F-Chapter"],
                sel_year,
            )
            mh_bar_agg = (
                mh_yr.groupby("diagnosis", as_index=False)
                # Plain .sum() treats a suppressed null as 0, so the bar renders
                # as a genuine zero instead of as withheld. min_count=1 keeps the
                # group null, and the flag records that it was suppressed rather
                # than absent.
                .agg(th_patients=("th_patients", lambda x: x.sum(min_count=1)),
                     **{SUPPRESS_FLAG_COL: ("th_patients", lambda x: x.isna().any())})
                .sort_values("th_patients", ascending=False).head(8)
            )
            total_mh = mh_bar_agg["th_patients"].sum()
            mh_bar_agg["pct"] = mh_bar_agg["th_patients"] / total_mh * 100 if total_mh else 0
            # Flag descending so withheld bars sit at the BOTTOM: categoryarray's
            # first element is the bottom row on a horizontal bar chart.
            mh_bar_agg = mh_bar_agg.sort_values(
                [SUPPRESS_FLAG_COL, "pct"], ascending=[False, True],
                na_position="first")

            _mx, _mtxt, _mcols, _mpats = _bar_display(
                mh_bar_agg["pct"], _supp_flag(mh_bar_agg),
                base_colors=[PALETTE[4]] * len(mh_bar_agg))

            mh_bar_fig = go.Figure(go.Bar(
                y=mh_bar_agg["diagnosis"].astype(str),
                x=_mx,
                orientation="h",
                marker=dict(color=_mcols, opacity=0.85,
                            pattern=dict(shape=_mpats, solidity=0.35,
                                         fgcolor="white", size=6)),
                text=_mtxt,
                textposition="outside",
                customdata=_supp_flag(mh_bar_agg).map(
                    {True: "Withheld (fewer than 11 patients)", False: ""}),
                hovertemplate="<b>%{y}</b><br>%{text}%{customdata}<extra></extra>",
            ))
            _mhmax = pd.to_numeric(mh_bar_agg["pct"], errors="coerce").max()
            mh_bar_fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                height=360,
                margin=dict(t=10, b=10, l=10, r=60),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                           range=[0, (_mhmax * 1.35) if pd.notna(_mhmax) and _mhmax > 0 else 1]),
                yaxis=dict(showgrid=False, type="category",
                           categoryorder="array",
                           categoryarray=mh_bar_agg["diagnosis"].astype(str).tolist()),
            )
            st.plotly_chart(mh_bar_fig, width='stretch')

    # ── Underlying data ────────────────────────────────────────────────────────
    with st.expander("📋 View underlying data — top diagnoses"):
        if not df_dx.empty:
            show_dx = (
                _filter_year(
                    df_dx[df_dx["diagnosis_level"] == "CCS Level 1"],
                    sel_year
                )
                .pipe(_filter_dimvalue_all)
                .groupby(["diagnosis"], as_index=False)
                .agg(th_patients=("th_patients", lambda x: x.sum(min_count=1)),
                     th_claims  =("th_claims",   lambda x: x.sum(min_count=1)),
                     **{SUPPRESS_FLAG_COL: ("th_patients", lambda x: x.isna().any())})
                .sort_values("th_patients", ascending=False).head(30)
            )
            total = show_dx["th_patients"].sum()
            show_dx["pct"] = show_dx["th_patients"] / total * 100 if total else 0
            show_dx = show_dx.rename(columns={
                "diagnosis"   : "Diagnosis (CCS Level 1)",
                "th_patients" : "TH Patients",
                "th_claims"   : "TH Claims",
                "pct"         : "% of Total",
            })
            show_dx = _mask_counts(show_dx, ["TH Patients", "TH Claims"])
            st.dataframe(
                show_dx.style.format({
                    "% of Total"  : "{:.1f}%",
                }, na_rep="—"),
                width='stretch', height=320,
            )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — TAB 4: COUNTY MAP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ County Map":

    df_county = DATA["county"].copy()

    # ── Payer subset → swap in the payer-stratified county file ───────────────
    # Every downstream path on this page (the three count metrics, the hover
    # merge, and the KPI row) reads from df_county, so substituting it here is
    # what makes the payer filter apply to all of them rather than only to
    # th_per_1000. Single-select today; the groupby keeps it correct if the
    # widget ever becomes multi-select.
    _county_payer_missing = False
    if payer_filter is not None:
        _cp = DATA["county_payer"]
        if not _cp.empty and "Payer_Type" in _cp.columns:
            _sub = _cp[_cp["Payer_Type"].isin(payer_filter)]
            df_county = (
                _sub.groupby(["source_year", "county_fips"], as_index=False)
                    .agg(th_patients  =("th_patients",   lambda x: x.sum(min_count=1)),
                         th_claims    =("th_claims",     lambda x: x.sum(min_count=1)),
                         total_allowed=("total_allowed", lambda x: x.sum(min_count=1)),
                         **{SUPPRESS_FLAG_COL: ("th_patients", lambda x: x.isna().any())})
            )
            # Recompute the state share within the selected payer subset.
            _tot = df_county.groupby("source_year")["th_patients"].transform("sum")
            df_county["pct_of_state_patients"] = np.where(
                _tot > 0, df_county["th_patients"] / _tot * 100, np.nan
            ).round(3)
            df_county["state_fips"] = df_county["county_fips"].str[:2]
        else:
            _county_payer_missing = True

    metric_key = sel_map_metric
    metric_cfg = COUNTY_MAP_METRICS[metric_key]
    state_clean = sel_state_fips.strip().zfill(2) if sel_state_fips.strip() else None
    state_name  = FIPS_TO_STATE.get(state_clean, state_clean) if state_clean else "United States"
    map_yr_label = sel_year if sel_year != "ALL" else "All Years"

    st.markdown(f'<p style="font-size: 20px;" class="sub-header">{VARIANT_LABEL} Patients by County — {state_name} ({map_yr_label})</p>',
                unsafe_allow_html=True)
    _src_label = ("county_summary_payer.parquet"
                  if (payer_filter is not None and not _county_payer_missing)
                  else "county_summary.parquet")
    st.markdown(f'<p style="font-size: 16px;" class="sub-caption">Metric: {metric_cfg["label"]} · Source: {_src_label}</p>',
                unsafe_allow_html=True)

    # Fix 6 — definition blurb, updates automatically whenever sel_map_metric changes
    st.info(f"ℹ️ **{metric_cfg['label']}**: {MAP_METRIC_DEFINITIONS.get(metric_key, '')}")

    if df_county.empty:
        st.warning("county_summary.parquet not found in AGG_DIR.")
    elif GEOJSON is None:
        st.warning(f"GeoJSON not found at `{GEOJSON_PATH}`. Download it once:")
        st.code(
            "import urllib.request\n"
            "urllib.request.urlretrieve(\n"
            "    'https://raw.githubusercontent.com/plotly/datasets/master/"
            "geojson-counties-fips.json',\n"
            f"    '{GEOJSON_PATH}')",
            language="python",
        )
    elif metric_key == "th_per_1000" and CTH.empty:
        # Without CTH, th_per_1000 doesn't exist in any data source, which
        # previously rendered a fully grey map with no explanation.
        st.warning(
            f"⚠️ **TH per 1,000 data not available for {VARIANT_LABEL}.** "
            f"This metric requires `county_th_metrics.parquet`, expected at "
            f"`{AGG_DIR / 'county_th_metrics.parquet'}`. Re-run "
            f"`build_aggregates.py --variant {sel_variant}` without "
            f"`--skip-county`, or choose a different map metric from the sidebar."
        )
    else:
        # ── Prepare data slice ────────────────────────────────────────────────
        # th_per_1000 uses county_th_metrics.parquet as its own consistent source
        # (both numerator and denominator were assigned to counties the same way).
        # All other metrics use county_summary.parquet from build_aggregates.py.
        # Payer subset → source county metrics from county_th_metrics_payer,
        # summed across the selected payers. Falls back to the all-payer CTH.
        # Payer is a dimension inside county_th_metrics now, with a true-distinct
        # ALL block, so the grain is selected rather than summed. The old code
        # summed the payer rows, which double-counted anyone with coverage under
        # two payers and could only ever add up values that suppression had
        # already nulled.
        cth_src = CTH
        if not CTH.empty and "Payer_Type" in CTH.columns:
            _payer_key = GRAIN_ALL if payer_filter is None else payer_filter[0]
            cth_src = CTH[CTH["Payer_Type"] == _payer_key]
            if cth_src.empty and _payer_key != GRAIN_ALL:
                cth_src = CTH[CTH["Payer_Type"] == GRAIN_ALL]
                st.info(f"No county rows for payer '{_payer_key}' — showing all payers.")
            elif payer_filter is not None and metric_key == "th_per_1000":
                st.caption(f"Payer filter: {', '.join(payer_filter)}")
        if payer_filter is not None and metric_key != "th_per_1000":
            # Count metrics now honour the filter too, via the df_county swap above.
            st.caption(f"Payer filter: {', '.join(payer_filter)}")

        if _county_payer_missing:
            st.info("Payer breakdown of county counts not available — re-run "
                    "`build_aggregates.py` to generate `county_summary_payer.parquet`. "
                    "Count metrics are showing all payers.")

        if metric_key == "th_per_1000" and not cth_src.empty:
            # All-years is now a published grain counted directly from claims,
            # not a sum of the yearly rows. That matters for low-volume variants:
            # a county suppressed in every individual year still has a real,
            # unsuppressed all-years total, which summing could never recover
            # because the yearly values are already null.
            _yr_key = GRAIN_ALL if sel_year == "ALL" else str(sel_year)
            df_map = cth_src[cth_src["source_year"].astype(str) == _yr_key].copy()

            # Prefer the published rate: it was nulled in step with the counts at
            # write time, so recomputing risks reviving a value suppression
            # removed. Only compute where the column is absent.
            if "th_per_1000" not in df_map.columns:
                df_map["th_per_1000"] = (
                    df_map["th_claimants"]
                    / df_map["total_claimants"].replace(0, pd.NA)
                    * 1000
                ).round(1)

            # Fix — counties with zero recorded telehealth activity were showing
            # as a colored "0.0" here instead of grey. The other three metrics
            # come from county_summary.parquet, which only has a row per county
            # when it had >=1 TH claim, so a TH-inactive county is simply absent
            # → NaN → grey. th_per_1000's denominator (ALL facility claims) is
            # populated almost everywhere, so the same TH-inactive county still
            # gets a real, non-null "0.0" and rendered colored — inconsistent
            # with every other map. Null it out so "no telehealth activity"
            # greys out the same way across all four metrics; real suppression
            # (already NaN from the notebook) is left untouched.
            df_map.loc[df_map["th_claimants"].fillna(0) <= 0, "th_per_1000"] = pd.NA

            # Pull th_patients / th_claims from county_summary for hover text.
            # SUPPRESS_FLAG_COL is deliberately EXCLUDED from the merge: both
            # frames now carry it, and merging two same-named columns yields
            # `suppressed_x`/`suppressed_y`, leaving no plain `suppressed` for
            # _supp_flag() to find. It would then return all-False, every
            # withheld county would fall through to the "No data" layer, and the
            # darker-grey suppressed layer would silently vanish — the exact
            # symptom this map had when county_th_metrics carried no flag at all.
            # county_th_metrics' own flag is the right one here anyway: it was
            # set against th_claimants, which is what this metric is built from.
            _cs_src = _filter_year(df_county, sel_year)
            _cs_cols = [c for c in ["county_fips", "th_patients", "th_claims"]
                        if c in _cs_src.columns]
            _cs = _cs_src[_cs_cols]
            if _cs["county_fips"].duplicated().any():
                # Fallback path only (pre-marginals aggregate): collapse safely.
                _cs = (_cs.groupby("county_fips", as_index=False)
                       .agg(th_patients=("th_patients", lambda x: x.sum(min_count=1)),
                            th_claims  =("th_claims",   lambda x: x.sum(min_count=1))))
            df_map = df_map.merge(_cs, on="county_fips", how="left")
            # Do NOT fillna(0) here — a suppressed count is unknown, not zero,
            # and coercing it to 0 both misreports the hover and silently drops
            # it from the KPI sums below.
            if "th_patients" not in df_map.columns:
                df_map["th_patients"] = pd.NA
            if "th_claims" not in df_map.columns:
                df_map["th_claims"] = pd.NA

            if SUPPRESS_FLAG_COL not in df_map.columns:
                # Pre-consolidation county_th_metrics.parquet (written by the
                # notebook) has no flag column. Without it every withheld county
                # renders identically to a county with no data, which is what
                # made this map disagree with the % of statewide map. Recover the
                # flag from the counts: a county that HAS a denominator but whose
                # numerator came back null was withheld, not absent.
                df_map[SUPPRESS_FLAG_COL] = (
                    df_map["th_claimants"].isna()
                    & df_map["total_claimants"].notna()
                    & (df_map["total_claimants"] > 0)
                )

            if state_clean:
                df_map = df_map[df_map["state_fips"] == state_clean]

        else:
            # Select the requested grain — including the pre-built all-years
            # grain — instead of summing yearly rows that may be suppressed.
            df_map = _filter_year(df_county, sel_year).copy()
            if df_map["county_fips"].duplicated().any():
                # Fallback for an aggregate built before marginals existed.
                df_map = (
                    df_map.groupby("county_fips", as_index=False)
                    .agg(th_patients=("th_patients", lambda x: x.sum(min_count=1)),
                         th_claims  =("th_claims",   lambda x: x.sum(min_count=1)))
                )
            if "pct_of_state_patients" not in df_map.columns:
                total_pts = df_map["th_patients"].sum(min_count=1)
                df_map["pct_of_state_patients"] = (
                    (df_map["th_patients"] / total_pts * 100).round(3)
                    if pd.notna(total_pts) and total_pts else np.nan)
            df_map["state_fips"] = df_map["county_fips"].str[:2]

            if state_clean:
                df_map = df_map[df_map["state_fips"] == state_clean]

        # Keep all county_summary rows — do NOT drop suppressed nulls.
        # Suppressed counties and counties absent from the data are both
        # greyed out; only counties with a real metric value get colour.

        # ── KPI row — data counties only ──────────────────────────────────────
        df_data_kpi = (
            df_map[df_map[metric_key].notna()]
            if metric_key in df_map.columns else df_map
        )
        if not df_data_kpi.empty:
            sort_col = "th_patients" if "th_patients" in df_data_kpi.columns else metric_key
            top_county = df_data_kpi.sort_values(sort_col, ascending=False).iloc[0]
            st.markdown("""
                    <style>
                    [data-testid="stMetricLabel"] p {
                        font-size: 24px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            # Suppressed counties contribute null, which pandas sums as 0. These
            # totals are therefore lower bounds over the unsuppressed counties;
            # say so rather than presenting them as complete statewide figures.
            # Count suppression over df_map, NOT df_data_kpi. df_data_kpi is
            # already filtered to rows where the metric is non-null, and a
            # suppressed row's metric is null by construction — so this counter
            # was always zero and the caption below never appeared.
            _n_supp = int(_supp_flag(df_map).sum())
            _pts = df_data_kpi["th_patients"].sum(min_count=1)
            _cls = df_data_kpi["th_claims"].sum(min_count=1)
            m1.metric("Counties with TH Activity", f"{df_data_kpi['county_fips'].nunique():,}")
            m2.metric("Total TH Patients", "—" if pd.isna(_pts) else f"{int(_pts):,}")
            m3.metric("Total TH Claims",   "—" if pd.isna(_cls) else f"{int(_cls):,}")
            m4.metric(
                "Top County",
                FIPS_TO_NAME.get(str(top_county["county_fips"]).zfill(5),
                                  top_county["county_fips"]),
            )
            if _n_supp:
                st.caption(
                    f"⚠ {_n_supp:,} count{'s' if _n_supp != 1 else ''} suppressed "
                    f"({SUPPRESS_MARK} patients) and shown in darker grey on the "
                    f"map — those counties DO have telehealth activity. The "
                    f"totals above exclude them, so treat them as lower bounds.")

        # ── Filter GeoJSON ────────────────────────────────────────────────────
        features = GEOJSON["features"]
        if state_clean:
            features = [ft for ft in features
                        if str(ft.get("id", "")).zfill(5).startswith(state_clean)]

        filtered_geo = {"type": "FeatureCollection", "features": features}

        # Build a complete county list from the GeoJSON so EVERY county in the
        # state is visible. Left-joining onto df_map means:
        #   • counties with data          → metric value (coloured)
        #   • counties suppressed in data → null metric  (grey)
        #   • counties not in data at all → null metric  (grey)
        geo_fips = [str(ft.get("id", "")).zfill(5) for ft in features]
        df_full  = (
            pd.DataFrame({"county_fips": geo_fips})
            .merge(df_map, on="county_fips", how="left")
        )
        df_full["county_name"] = (
            df_full["county_fips"].map(FIPS_TO_NAME).fillna(df_full["county_fips"])
        )

        has_data   = df_full[metric_key].notna() if metric_key in df_full.columns \
                     else pd.Series(False, index=df_full.index)
        # Split the uncoloured counties in two. Rendering "suppressed" and
        # "no data" in the same grey tells the reader a county had no telehealth
        # when in fact it had between 1 and 10 patients — the opposite of what
        # the data says. They get separate layers and separate hover text.
        is_supp    = _supp_flag(df_full) & (~has_data)
        df_colored = df_full[has_data].copy()
        df_supp    = df_full[is_supp].copy()
        df_grey    = df_full[~has_data & ~is_supp].copy()

        # ── Hover text ────────────────────────────────────────────────────────
        def _make_hover(row, grey: bool, suppressed: bool = False) -> str:
            name = row["county_name"]
            if suppressed:
                return (f"<b>{name}</b><br>"
                        f"FIPS: {row['county_fips']}<br>"
                        f"Suppressed: fewer than {DISPLAY_SUPPRESS_THRESHOLD} "
                        f"patients<br><i>Data exists but is withheld</i>")
            if grey:
                return (f"<b>{name}</b><br>"
                        f"FIPS: {row['county_fips']}<br>"
                        f"No data")
            v      = row.get(metric_key)
            v_str  = metric_cfg["fmt"](v) if pd.notna(v) else "—"
            # A suppressed count arrives as null from the parquet. Previously
            # that was coerced to 0, which reads as "no telehealth here" — the
            # opposite of the truth. Show "<11" when the row is flagged
            # suppressed, "—" when there is genuinely nothing.
            supp = bool(row.get(SUPPRESS_FLAG_COL, False))

            def _hov(col):
                raw = row.get(col)
                if pd.isna(raw):
                    return f"<{DISPLAY_SUPPRESS_THRESHOLD}" if supp else "—"
                n = int(raw)
                if 0 < n < DISPLAY_SUPPRESS_THRESHOLD:      # stale-parquet guard
                    return f"<{DISPLAY_SUPPRESS_THRESHOLD}"
                return f"{n:,}"

            pts_str = _hov("th_patients")
            cls_str = _hov("th_claims")
            return (
                f"<b>{name}</b><br>"
                f"FIPS: {row['county_fips']}<br>"
                f"TH Patients: {pts_str}<br>"
                f"TH Claims: {cls_str}<br>"
                # f"{metric_cfg['label']}: {v_str}"
            )

        df_colored["hover_text"] = df_colored.apply(
            lambda r: _make_hover(r, grey=False), axis=1)
        df_grey["hover_text"]    = df_grey.apply(
            lambda r: _make_hover(r, grey=True),  axis=1)
        if not df_supp.empty:
            df_supp["hover_text"] = df_supp.apply(
                lambda r: _make_hover(r, grey=True, suppressed=True), axis=1)

        # ── Colour scale cap ──────────────────────────────────────────────────
        vmax = (
            float(df_colored[metric_key].quantile(MAP_CAP_PCT / 100))
            if not df_colored.empty and metric_key in df_colored.columns
            else 1.0
        )

        # ── Centre + zoom from bounding box (offline, no mapbox token) ───────
        all_lons, all_lats = [], []
        for ft in features:
            geom_type = ft["geometry"].get("type", "")
            coords    = ft["geometry"].get("coordinates", [])
            try:
                if geom_type == "Polygon":
                    for ring in coords:
                        for lon, lat in ring: all_lons.append(lon); all_lats.append(lat)
                elif geom_type == "MultiPolygon":
                    for poly in coords:
                        for ring in poly:
                            for lon, lat in ring: all_lons.append(lon); all_lats.append(lat)
            except (TypeError, ValueError):
                pass

        if all_lons:
            c_lon = (min(all_lons) + max(all_lons)) / 2
            c_lat = (min(all_lats) + max(all_lats)) / 2
            span  = max(max(all_lons)-min(all_lons), max(all_lats)-min(all_lats))
            zoom  = 3.0 if span>50 else 4.5 if span>20 else 5.8 if span>8 else 6.8
        else:
            c_lon, c_lat, zoom = -78.5, 37.5, 6.0

        # ── Two-trace choropleth: grey base → coloured data ───────────────────
        # Trace order matters: grey is drawn first so coloured counties render
        # on top with correct edges.
        map_fig = go.Figure()

        if not df_grey.empty:
            map_fig.add_trace(go.Choroplethmapbox(
                geojson      = filtered_geo,
                locations    = df_grey["county_fips"],
                z            = [0] * len(df_grey),
                featureidkey = "id",
                colorscale   = [[0, "#EFEFEF"], [1, "#EFEFEF"]],
                showscale    = False,
                marker_line_width = 0.4,
                marker_line_color = "white",
                text          = df_grey["hover_text"],
                hovertemplate = "%{text}<extra></extra>",
                name          = "No data",
            ))

        # Suppressed counties: a DIFFERENT, darker grey than "no data", because
        # these counties do have telehealth activity — just too little to
        # publish. Drawn between the two so coloured counties stay on top.
        if not df_supp.empty:
            map_fig.add_trace(go.Choroplethmapbox(
                geojson      = filtered_geo,
                locations    = df_supp["county_fips"],
                z            = [0] * len(df_supp),
                featureidkey = "id",
                colorscale   = [[0, "#B0B0B0"], [1, "#B0B0B0"]],
                showscale    = False,
                marker_line_width = 0.6,
                marker_line_color = "white",
                text          = df_supp["hover_text"],
                hovertemplate = "%{text}<extra></extra>",
                name          = f"Suppressed ({SUPPRESS_MARK})",
            ))

        if not df_colored.empty:
            map_fig.add_trace(go.Choroplethmapbox(
                geojson      = filtered_geo,
                locations    = df_colored["county_fips"],
                z            = df_colored[metric_key],
                featureidkey = "id",
                colorscale   = metric_cfg["colorscale"],
                zmin=0, zmax=vmax,
                marker_line_width = 0.4,
                marker_line_color = "white",
                colorbar=dict(
                    title=dict(text=metric_cfg["label"], side="right"),
                    thickness=14, len=0.6,
                ),
                text          = df_colored["hover_text"],
                hovertemplate = "%{text}<extra></extra>",
                name          = "TH Activity",
            ))

        map_fig.update_layout(
            mapbox=dict(style="white-bg", center=dict(lon=c_lon, lat=c_lat), zoom=zoom),
            margin=dict(l=0, r=0, t=70, b=10),
            height=580,
            showlegend=False,
            title=dict(
                text=f"<b>{metric_cfg['label']} by County — {state_name} ({map_yr_label})</b>",
                x=0.5, font=dict(size=22, color="#1B4F8A"),
            ),
        )
        st.plotly_chart(map_fig, width='stretch')
        st.caption(MAP_CITY_CAVEAT)

        # ── Top 10 table ──────────────────────────────────────────────────────
        with st.expander("📋 Top counties by TH patients"):
            # th_per_1000 path: df_colored comes from CTH (no pct_of_state_patients)
            # Other paths: df_colored comes from county_summary (has pct_of_state_patients)
            sort_col = "th_patients" if "th_patients" in df_colored.columns else metric_key
            extra_cols = {
                "pct_of_state_patients": "% of State TH Patients",
                "th_per_1000"          : "TH per 1,000 Claimants",
                "total_claimants"      : "Total Claimants",
            }
            base_cols  = ["county_name", "county_fips", "th_patients", "th_claims"]
            avail_extra = {col: lbl for col, lbl in extra_cols.items()
                           if col in df_colored.columns}
            sel_cols = [c for c in base_cols + list(avail_extra) + [SUPPRESS_FLAG_COL]
                        if c in df_colored.columns]
            rename_map = {"county_name": "County", "county_fips": "FIPS",
                          "th_patients": "TH Patients", "th_claims": "TH Claims",
                          **avail_extra}
            fmt_map = {"TH Patients": "{:,.0f}", "TH Claims": "{:,.0f}",
                       "% of State TH Patients": "{:.2f}%",
                       "TH per 1,000 Claimants": "{:.1f}",
                       "Total Claimants": "{:,.0f}"}

            top_cty = (
                df_colored.sort_values(sort_col, ascending=False)
                .head(20)[sel_cols]
                .rename(columns=rename_map)
            )
            top_cty = _mask_counts(top_cty, ["TH Patients", "TH Claims"])
            fmt_map_remaining = {k: v for k, v in fmt_map.items()
                                 if k not in ("TH Patients", "TH Claims")}
            st.dataframe(
                top_cty.style.format(
                    {k: v for k, v in fmt_map_remaining.items() if k in top_cty.columns},
                    na_rep="—"
                ),
                width='stretch', height=300,
            )

        # ── PNG download (matplotlib, offline-safe) ───────────────────────────
        with st.expander("⬇️ Download map as PNG (matplotlib)"):
            if st.button("Generate PNG"):
                gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                gdf["county_fips"] = [str(ft["id"]).zfill(5) for ft in features]
                gdf = gdf.drop_duplicates("county_fips")
                gdf = gdf.merge(
                    df_map[["county_fips", metric_key, "th_patients"]],
                    on="county_fips", how="left",
                ).reset_index(drop=True)
                import matplotlib.cm as _cm
                cmap_obj  = _cm.get_cmap(metric_cfg["colorscale"])
                norm      = Normalize(vmin=0, vmax=vmax)
                fig_dl, ax_dl = plt.subplots(figsize=(16, 9), facecolor="white")
                gdf[gdf[metric_key].isna()].plot(ax=ax_dl, color="#EEE",
                                                  edgecolor="white", linewidth=0.3)
                gdf[gdf[metric_key].notna()].plot(
                    column=metric_key, ax=ax_dl,
                    cmap=cmap_obj, vmin=0, vmax=vmax,
                    edgecolor="white", linewidth=0.3,
                )
                sm = ScalarMappable(cmap=cmap_obj, norm=norm); sm.set_array([])
                fig_dl.colorbar(sm, ax=ax_dl, fraction=0.025, pad=0.02).set_label(
                    metric_cfg["label"], fontsize=11)
                ax_dl.set_title(
                    f"{metric_cfg['label']} by County — {state_name} ({map_yr_label})",
                    fontsize=18, fontweight="bold", color="#1B4F8A")
                ax_dl.axis("off"); fig_dl.tight_layout()
                buf = BytesIO()
                fig_dl.savefig(buf, format="png", dpi=150,
                               bbox_inches="tight", facecolor="white")
                plt.close(fig_dl); buf.seek(0)
                st.download_button(
                    label="Click to download",
                    data=buf,
                    file_name=(f"county_{metric_key}_{sel_year}"
                               f"{'_' + state_clean if state_clean else ''}.png"),
                    mime="image/png",
                )
