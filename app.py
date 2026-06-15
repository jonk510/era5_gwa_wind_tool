"""
ERA5 + GWA Wind Resource Tool
Combines 10 years of ERA5 hourly wind data at 100m with Global Wind Atlas
spatial accuracy, extrapolated to 150m hub height.

Pipeline:
  1. ERA5 100m hourly  (Open-Meteo)
  2. Height 100→150m   (power-law, diurnal α from ERA5 10m/100m shear,
                        magnitude calibrated to GWA 100m/150m mean ratio)
  3. Weibull correction (quantile transform to match GWA 150m Weibull A & k)
"""

import re
import warnings
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq
from scipy.signal import lfilter
from scipy.special import gamma as sc_gamma
from scipy.stats import weibull_min
from streamlit_folium import st_folium
from timezonefinder import TimezoneFinder

warnings.filterwarnings("ignore")

_TF = TimezoneFinder()


def detect_timezone(lat: float, lon: float) -> str:
    """Return IANA timezone string for a lat/lon, defaulting to 'UTC'."""
    tz = _TF.timezone_at(lat=lat, lng=lon)
    return tz or "UTC"


def localise_df(df: pd.DataFrame, tz_name: str) -> pd.DataFrame:
    """Convert a UTC-naive datetime index to the given timezone."""
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return df.set_axis(idx.tz_convert(tz_name))


def _tz_offset_str(index: pd.DatetimeIndex) -> str:
    """Return e.g. '+08:00' from a timezone-aware index."""
    try:
        offset = index[0].utcoffset()
        total_minutes = int(offset.total_seconds() // 60)
        sign = "+" if total_minutes >= 0 else "-"
        h, m = divmod(abs(total_minutes), 60)
        return f"{sign}{h:02d}:{m:02d}"
    except Exception:
        return ""

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ERA5 + GWA 150m Wind Tool",
    page_icon="💨",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
_LATEST_YEAR = datetime.now().year - 1   # last complete calendar year
GWA_URL = "https://globalwindatlas.info/api/gwa/custom/Lib/"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR = "&copy; <a href='https://www.esri.com'>Esri</a>"


# ── GWC parser ────────────────────────────────────────────────────────────────

def _combined_weibull(f_arr, A_arr, k_arr):
    """
    Compute omnidirectional Weibull A, k from sector arrays using method of moments.
    Returns (A_combined, k_combined, mean_wind_speed).
    """
    f = np.asarray(f_arr, float)
    A = np.asarray(A_arr, float)
    k = np.asarray(k_arr, float)
    f = f / f.sum()

    mu = float(np.sum(f * A * sc_gamma(1 + 1 / k)))
    mu2 = float(np.sum(f * A ** 2 * sc_gamma(1 + 2 / k)))
    var = mu2 - mu ** 2

    if var <= 0 or mu <= 0:
        k_c = 2.0
    else:
        ratio = mu ** 2 / (var + mu ** 2)

        def _obj(kv):
            return sc_gamma(1 + 1 / kv) ** 2 / sc_gamma(1 + 2 / kv) - ratio

        try:
            k_c = brentq(_obj, 0.5, 10.0)
        except Exception:
            k_c = 2.0

    A_c = mu / sc_gamma(1 + 1 / k_c)
    return float(A_c), float(k_c), float(mu)


def parse_gwc(text, ref_roughness: float = 0.1):
    """
    Parse the WAsP GWC text returned by the GWA API.

    Actual format observed from the GWA API:
        Line 0 : n_roughness  n_heights  n_sectors
        Line 1 : roughness_1  roughness_2 ...
        Line 2 : height_1  height_2 ...
        For each roughness class (outer loop):
            [1 line]  12 sector frequencies (%)
            For each height (inner loop):
                [1 line]  12 Weibull A values (m/s)
                [1 line]  12 Weibull k values

    Returns:
        gwc      – dict  { height_m: {'A': float, 'k': float, 'mean': float} }
        heights  – sorted list of heights found in file
        gwa_lat  – GWA grid node latitude
        gwa_lon  – GWA grid node longitude
    """
    # Extract GWA grid coordinates from header before stripping tags
    # Format: <coordinates>lon,lat,elev</coordinates>
    gwa_lat, gwa_lon = None, None
    coord_match = re.search(r"<coordinates>(.*?)</coordinates>", text)
    if coord_match:
        parts = coord_match.group(1).split(",")
        try:
            gwa_lon = float(parts[0])
            gwa_lat = float(parts[1])
        except (IndexError, ValueError):
            pass

    clean = re.sub(r"<[^>]+>", "", text)
    # Keep only lines that are numeric data (start with a digit, minus, or space+digit)
    lines = [
        ln.strip() for ln in clean.splitlines()
        if ln.strip() and re.match(r"^\s*-?\d", ln)
    ]

    if not lines:
        raise ValueError("GWC file contains no numeric data after stripping XML tags.")

    # First numeric line is: n_roughness  n_heights  n_sectors
    meta = lines[0].split()
    if len(meta) < 3:
        raise ValueError(f"Unexpected GWC header: {lines[0]!r}")

    n_rough, n_heights, n_sectors = int(meta[0]), int(meta[1]), int(meta[2])

    rough_vals = list(map(float, lines[1].split()))[:n_rough]
    height_vals = list(map(float, lines[2].split()))[:n_heights]

    # Collect all roughness classes first, then select the one closest to
    # ref_roughness (supplied by the caller, derived from site land cover).
    # r=0.0 (sea) is only used when no land alternatives exist.
    all_data = {}   # (roughness, height) → {A, k, mean}
    idx = 3

    for ri in range(n_rough):
        if idx >= len(lines):
            break
        # One frequency line per roughness class (values in %)
        f_arr = list(map(float, lines[idx].split()))[:n_sectors]
        idx += 1

        for hi in range(n_heights):
            if idx + 1 >= len(lines):
                break
            A_arr = list(map(float, lines[idx].split()))[:n_sectors]
            idx += 1
            k_arr = list(map(float, lines[idx].split()))[:n_sectors]
            idx += 1

            if not A_arr or not k_arr or not f_arr:
                continue

            h = height_vals[hi]
            r = rough_vals[ri] if ri < len(rough_vals) else 999
            A_c, k_c, mean_ws = _combined_weibull(f_arr, A_arr, k_arr)
            all_data[(r, h)] = {"A": A_c, "k": k_c, "mean": mean_ws}

    if not all_data:
        raise ValueError("Could not extract any height data from GWC file.")

    available_roughnesses = sorted({r for r, _ in all_data})
    land_roughnesses = [r for r in available_roughnesses if r > 0.001]
    selected_r = (
        min(land_roughnesses, key=lambda rv: abs(rv - ref_roughness))
        if land_roughnesses else available_roughnesses[0]
    )

    gwc = {
        h: dict(vals, roughness=selected_r)
        for (r, h), vals in all_data.items()
        if r == selected_r
    }

    if not gwc:
        raise ValueError("Could not extract any height data from GWC file.")

    return gwc, sorted(gwc.keys()), gwa_lat, gwa_lon


# ── Data fetching (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_era5(lat: float, lon: float, start_year: int, end_year: int):
    """
    Fetch hourly ERA5 wind at 10m and 100m from Open-Meteo for the given year range.
    Returns (df, era5_lat, era5_lon) — the lat/lon are the actual ERA5 grid node
    coordinates used, which may differ from the input by up to ~0.125°.
    """
    r = requests.get(
        OPENMETEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "hourly": "wind_speed_100m,wind_speed_10m,wind_gusts_10m,wind_direction_100m",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    era5_lat = d.get("latitude", lat)
    era5_lon = d.get("longitude", lon)
    df = pd.DataFrame(
        {
            "ws_100m": d["hourly"]["wind_speed_100m"],
            "ws_10m": d["hourly"]["wind_speed_10m"],
            "ws_gust_10m": d["hourly"]["wind_gusts_10m"],
            "wd_100m": d["hourly"]["wind_direction_100m"],
        },
        index=pd.to_datetime(d["hourly"]["time"]),
    )
    return df.dropna(), era5_lat, era5_lon


@st.cache_data(show_spinner=False)
def fetch_gwa(lat: float, lon: float, ref_roughness: float = 0.1):
    """Fetch and parse GWA GWC file for the given location."""
    r = requests.get(
        GWA_URL,
        params={"lat": lat, "long": lon},
        headers={
            "Referer": "https://globalwindatlas.info",
            "User-Agent": "Mozilla/5.0",
        },
        timeout=30,
    )
    r.raise_for_status()
    return parse_gwc(r.text, ref_roughness)  # returns (gwc, heights, gwa_lat, gwa_lon)


# OSM landuse/natural → aerodynamic roughness length (m) for GWC class selection.
#
# Only three categories matter at 100m+ hub height:
#   • Water/sea    → r ≈ 0.0003  (select r=0.0 GWC class)
#   • Bare/smooth  → r ≈ 0.025   (select r=0.03 GWC class)
#   • All land     → r = 0.1     (default; r=0.1 GWC class gives ~3 % accuracy
#                                  vs GWA website for typical onshore terrain)
#
# We intentionally do NOT map farmland/grassland/forest to lower/higher classes
# because at rotor height the effective roughness converges toward ~0.1 m
# regardless of surface cover — using r=0.4 for forest or r=0.03 for
# farmland empirically worsens accuracy vs the GWA website.
_OSM_TO_ROUGHNESS: dict[str, float] = {
    # Water → sea roughness class
    "water": 0.0003, "bay": 0.0003, "strait": 0.0003,
    "reservoir": 0.0003, "basin": 0.0003,
    # Genuinely bare / featureless terrain → r=0.03 class
    "beach": 0.025, "sand": 0.025, "bare_rock": 0.02, "scree": 0.02,
    # Everything else (farmland, grassland, meadow, forest, urban …) → default land
}
_OSM_DEFAULT_LAND = 0.1   # fallback for any unrecognised land-cover tag


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_site_roughness(lat: float, lon: float) -> tuple[float, str]:
    """
    Estimate aerodynamic roughness length (m) at the site from OpenStreetMap
    land-use/natural tags within a 500 m radius via the Overpass API.

    Returns (roughness_m, description_string).
    Falls back to (0.1, "default") if the query fails or returns no data.
    """
    _fallback = (0.1, "default (r = 0.10 m — open land, no OSM data)")
    query = (
        f"[out:json][timeout:10];"
        f"(way[\"landuse\"](around:500,{lat},{lon});"
        f"way[\"natural\"](around:500,{lat},{lon}););"
        f"out tags;"
    )
    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "ERA5-GWA-WindTool/1.0 (wind resource research)"},
            timeout=12,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        if not elements:
            return _fallback

        rough_vals, tag_names = [], []
        for el in elements:
            tags = el.get("tags", {})
            lu = tags.get("landuse") or tags.get("natural")
            if lu:
                rv = _OSM_TO_ROUGHNESS.get(lu, _OSM_DEFAULT_LAND)
                rough_vals.append(rv)
                tag_names.append(lu)

        if not rough_vals:
            return _fallback

        median_r = float(np.median(rough_vals))
        dominant = max(set(tag_names), key=tag_names.count)
        return median_r, f"OpenStreetMap (dominant: {dominant}, r = {median_r:.4f} m)"

    except Exception:
        return _fallback


# ── AEP helpers ───────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"


@st.cache_data(show_spinner=False)
def load_power_curves() -> pd.DataFrame | None:
    """Load data/power_curves.xlsx. Returns DataFrame[WTG → kW] indexed by wind speed (m/s)."""
    p = _DATA_DIR / "power_curves.xlsx"
    if not p.exists():
        return None
    df = pd.read_excel(p, index_col=0, header=0)
    df.index = df.index.astype(float)
    df.columns = [str(c).strip() for c in df.columns]  # strip any trailing whitespace from WTG names
    return df.sort_index()


@st.cache_data(show_spinner=False)
def load_wake_matrix() -> pd.DataFrame | None:
    """Load data/wake_loss_matrix.xlsx. Returns DataFrame[nameplate_MW → %] indexed by wind speed (m/s).

    Expected layout (matches the supplied file):
      Row 0:  title row (ignored)
      Row 1:  col 0-1 ignored; col 2+ = nameplate capacity values (MW)
      Row 2+: col 0 ignored; col 1 = wind speed (m/s); col 2+ = wake loss (fraction 0–1)
    Values are converted to % (multiplied by 100) for use in calc_aep.
    """
    p = _DATA_DIR / "wake_loss_matrix.xlsx"
    if not p.exists():
        return None
    raw = pd.read_excel(p, header=None)
    nameplate_caps = raw.iloc[1, 2:].values.astype(float)
    wind_speeds = raw.iloc[2:, 1].values.astype(float)
    wake_fractions = raw.iloc[2:, 2:].values.astype(float)
    df = pd.DataFrame(wake_fractions * 100.0, index=wind_speeds, columns=nameplate_caps)
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df.sort_index().sort_index(axis=1)


def calc_aep(
    ws: pd.Series,
    pc_df: pd.DataFrame,
    wtg: str,
    nameplate_mw: float,
    wake_df: pd.DataFrame | None,
) -> dict:
    """Apply power curve + optional wake losses to an hourly wind speed series; return AEP stats."""
    ws_arr = pc_df.index.values.astype(float)
    kw_arr = pc_df[wtg].values.astype(float)
    rated_kw = float(kw_arr.max())

    ws_vals = ws.values
    gross_kw = np.interp(ws_vals, ws_arr, kw_arr, left=0.0, right=rated_kw)
    gross_mw = gross_kw / rated_kw * nameplate_mw

    if wake_df is not None:
        ws_bins = wake_df.index.values.astype(float)
        cap_bins = wake_df.columns.values.astype(float)
        interp_fn = RegularGridInterpolator(
            (ws_bins, cap_bins),
            wake_df.values.astype(float),
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        pts = np.column_stack([
            np.clip(ws_vals, ws_bins.min(), ws_bins.max()),
            np.full(len(ws_vals), float(np.clip(nameplate_mw, cap_bins.min(), cap_bins.max()))),
        ])
        wake_pct = interp_fn(pts).clip(0.0, 100.0)
        net_mw = gross_mw * (1.0 - wake_pct / 100.0)
        producing = gross_mw > 0
        mean_wake = float(wake_pct[producing].mean()) if producing.any() else 0.0
    else:
        net_mw = gross_mw.copy()
        wake_pct = np.zeros_like(gross_mw)
        mean_wake = 0.0

    n_years = len(ws) / 8760.0
    gross_aep = float(gross_mw.sum()) / n_years   # MWh/yr
    net_aep = float(net_mw.sum()) / n_years        # MWh/yr
    cf = net_aep / (nameplate_mw * 8760.0)

    return {
        "gross_mw_ts": pd.Series(gross_mw, index=ws.index),
        "net_mw_ts": pd.Series(net_mw, index=ws.index),
        "wake_pct_ts": pd.Series(wake_pct, index=ws.index),
        "gross_aep_mwh": gross_aep,
        "net_aep_mwh": net_aep,
        "mean_wake_pct": mean_wake,
        "capacity_factor": cf,
        "rated_kw": rated_kw,
        "n_years": n_years,
    }


# ── Processing pipeline ───────────────────────────────────────────────────────

def run_pipeline(df_era5: pd.DataFrame, gwc: dict, heights: list) -> tuple:
    """
    Full processing pipeline:
      1. Derive mean-shear alpha from GWA 100m / 150m means.
      2. Build diurnal alpha profile using ERA5 10m/100m shear pattern,
         normalised to alpha_mean.
      3. Extrapolate ERA5 100m → 150m with diurnal alpha.
      4. Fit Weibull to ERA5 150m estimate.
      5. Apply Weibull quantile transform to match GWA 150m Weibull.

    Returns (df, meta).
    """
    h100 = min(heights, key=lambda x: abs(x - 100))
    h150 = min(heights, key=lambda x: abs(x - 150))

    g100 = gwc[h100]
    g150 = gwc[h150]

    mean_gwa_100 = g100["mean"]
    mean_gwa_150 = g150["mean"]
    A_gwa_150 = g150["A"]
    k_gwa_150 = g150["k"]

    if mean_gwa_150 <= mean_gwa_100:
        raise ValueError(
            f"GWA mean at {h150}m ({mean_gwa_150:.2f} m/s) is not greater than "
            f"at {h100}m ({mean_gwa_100:.2f} m/s). Check GWC parsing."
        )

    # ── Step 1: GWA-derived mean shear exponent ───────────────────────────────
    # Primary: α from 100m→150m (used for actual extrapolation)
    alpha_mean = np.log(mean_gwa_150 / mean_gwa_100) / np.log(150 / 100)

    # Supplementary: α from 50m→150m if GWA 50m is available (spans a wider
    # height range and is more comparable to the "industry standard" α ≈ 0.2)
    alpha_50_150 = None
    mean_gwa_50 = None
    h50_candidates = [h for h in heights if 40 <= h <= 75]
    if h50_candidates:
        h50 = min(h50_candidates, key=lambda x: abs(x - 50))
        g50 = gwc.get(h50)
        if g50 and g50["mean"] > 0 and g50["mean"] < mean_gwa_100:
            mean_gwa_50 = g50["mean"]
            alpha_50_150 = np.log(mean_gwa_150 / mean_gwa_50) / np.log(h150 / h50)

    # ── Step 2: ERA5 diurnal shear pattern ───────────────────────────────────
    df = df_era5.copy()
    df["hour"] = df.index.hour

    valid = df[(df["ws_10m"] > 0.5) & (df["ws_100m"] > 0.5)].copy()
    valid["log_shear"] = np.log(valid["ws_100m"] / valid["ws_10m"]) / np.log(100 / 10)

    diurnal_era5 = (
        valid.groupby("hour")["log_shear"]
        .mean()
        .clip(0.03, 0.8)
        .reindex(range(24), fill_value=alpha_mean)
    )

    # Normalise pattern to GWA magnitude
    diurnal_alpha = diurnal_era5 / diurnal_era5.mean() * alpha_mean

    # ── Step 3: Height extrapolation ─────────────────────────────────────────
    df["alpha_h"] = df["hour"].map(diurnal_alpha).fillna(alpha_mean)
    df["ws_150m_raw"] = df["ws_100m"] * (150.0 / 100.0) ** df["alpha_h"]

    # ── Step 4: Fit Weibull to ERA5 150m estimate ─────────────────────────────
    ws_clean = df["ws_150m_raw"].dropna()
    ws_clean = ws_clean[ws_clean > 0.1]
    k_era5_150, _, A_era5_150 = weibull_min.fit(ws_clean, floc=0)

    # ── Step 5: Weibull quantile transform ────────────────────────────────────
    # v* = A_GWA × (v / A_ERA5)^(k_ERA5 / k_GWA)
    v = df["ws_150m_raw"].clip(lower=0.01).values
    df["ws_150m_corrected"] = A_gwa_150 * (v / A_era5_150) ** (k_era5_150 / k_gwa_150)

    meta = {
        "h100_used": h100,
        "h150_used": h150,
        "alpha_mean": alpha_mean,
        "alpha_50_150": alpha_50_150,
        "diurnal_alpha": diurnal_alpha,
        "mean_era5_100": df["ws_100m"].mean(),
        "mean_era5_150_raw": df["ws_150m_raw"].mean(),
        "mean_corrected": df["ws_150m_corrected"].mean(),
        "mean_gwa_50": mean_gwa_50,
        "mean_gwa_100": mean_gwa_100,
        "mean_gwa_150": mean_gwa_150,
        "A_era5_150": A_era5_150,
        "k_era5_150": k_era5_150,
        "A_gwa_150": A_gwa_150,
        "k_gwa_150": k_gwa_150,
        "gwa_roughness_used": g150.get("roughness"),
    }

    return df, meta


# ── Sub-hourly disaggregation ─────────────────────────────────────────────────

def disaggregate_subhourly(
    df: pd.DataFrame,
    resolution_min: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Stochastically disaggregate hourly 150m wind to sub-hourly resolution.

    Method
    ------
    1. Per-hour turbulence intensity at 150m estimated from ERA5 gust factor:
         TI_10m  = (V_gust / V_10m − 1) / 3.5   (ERA5 ~1-min gust, peak factor ~3.5)
         TI_150m = TI_10m × (10/150)^0.11         (TI decreases with height)
         σ_u     = TI_150m × V_150m_corrected      (per-hour std at hub height)

       For 10-min or 30-min *mean* output, variance is reduced from instantaneous:
         σ_mean = σ_u × sqrt(T_int / T_avg)
       where T_int ≈ 350 s (integral time scale at 150m) and T_avg is the
       averaging period.  Clamped to [0.02 × V, 0.50 × V].

    2. A single AR(1) process is generated at sub-hourly timesteps across the
       entire record (continuous, no jumps at hour boundaries).  The AR(1)
       coefficient is derived from the hourly autocorrelation via a
       continuous-time OU assumption.

    3. Each hourly block's AR(1) noise is mean-corrected so that the sub-hourly
       block mean exactly equals the ERA5 hourly value (mean-preserving).

    Returns
    -------
    (df_sub, info_dict)
    df_sub  — DataFrame at sub-hourly resolution with column ws_150m_subhourly
    info    — dict with TI statistics
    """
    rng = np.random.default_rng(seed)
    n_sub = 60 // resolution_min
    N_total = len(df) * n_sub

    V = df["ws_150m_corrected"].values
    V_10m = df["ws_10m"].values.clip(min=1.0)

    # ── Per-hour sigma at hub height ─────────────────────────────────────────
    if "ws_gust_10m" in df.columns:
        GF = df["ws_gust_10m"].values / V_10m
        TI_10m = np.clip((GF - 1.0) / 3.5, 0.03, 0.45)
        TI_150 = TI_10m * (10.0 / 150.0) ** 0.11
        ti_method = "ERA5 gust factor"
    else:
        # Fallback: terrain-neutral TI profile from IEC class B approximation
        TI_150 = np.where(V > 0.5, 0.14 / (1 + 0.1 * V / V.mean()), 0.14)
        ti_method = "IEC class B approximation"

    # Spectral variance reduction for averaging period vs integral time scale
    T_int_s = 350.0          # ~integral length scale at 150m / typical wind speed
    T_avg_s = resolution_min * 60.0
    spectral_factor = min(1.0, np.sqrt(T_int_s / T_avg_s))
    sigma_h = np.clip(TI_150 * V * spectral_factor, 0.02 * np.maximum(V, 0.1), 0.50 * np.maximum(V, 0.1))

    # ── AR(1) coefficient from hourly autocorrelation ────────────────────────
    phi_1h = float(pd.Series(V).autocorr(lag=1))
    phi_1h = np.clip(phi_1h, 0.50, 0.98)
    T_decorr_min = -60.0 / np.log(phi_1h)
    phi_sub = float(np.exp(-resolution_min / T_decorr_min))

    # ── Generate continuous AR(1) noise ──────────────────────────────────────
    # Use unit-variance AR(1) via lfilter, then scale to per-hour sigma
    sigma_rep = np.repeat(sigma_h, n_sub)          # (N_total,)
    innov_std = np.sqrt(1.0 - phi_sub ** 2)
    white = rng.standard_normal(N_total)
    noise_unit = lfilter([innov_std], [1.0, -phi_sub], white)

    # Rescale to per-hour sigma (normalize unit noise, then apply sigma)
    unit_std = noise_unit.std()
    noise = noise_unit * (sigma_rep / unit_std) if unit_std > 0 else noise_unit * sigma_rep

    # ── Mean-preserve: subtract block mean per hour ──────────────────────────
    noise_reshaped = noise.reshape(len(df), n_sub)
    noise_reshaped -= noise_reshaped.mean(axis=1, keepdims=True)
    noise = noise_reshaped.ravel()

    # ── Build output ─────────────────────────────────────────────────────────
    background = np.repeat(V, n_sub)
    ws_sub = np.maximum(background + noise, 0.0)

    sub_index = pd.date_range(
        df.index[0],
        periods=N_total,
        freq=f"{resolution_min}min",
    )
    df_sub = pd.DataFrame({"ws_150m_subhourly": ws_sub}, index=sub_index)

    info = {
        "ti_method": ti_method,
        "mean_TI_150": float(TI_150.mean()),
        "mean_sigma": float(sigma_h.mean()),
        "phi_sub": phi_sub,
        "spectral_factor": spectral_factor,
        "resolution_min": resolution_min,
        "n_sub": n_sub,
    }
    return df_sub, info


# ── UI ────────────────────────────────────────────────────────────────────────

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1400px;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FAFAFA !important;
    border-right: 1px solid #EAECF0 !important;
}

/* ── Primary button ──────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #0F172A !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    height: 2.6rem !important;
    font-size: 0.88rem !important;
    transition: opacity 0.15s ease !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.85 !important;
}

/* ── Metric tiles ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #94A3B8 !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] > div {
    font-size: 0.78rem !important;
}

/* ── Expanders ───────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    box-shadow: none !important;
}
[data-testid="stExpanderToggleIcon"] { color: #94A3B8 !important; }

/* ── Download button ─────────────────────────────────────── */
.stDownloadButton > button {
    border-radius: 6px !important;
    font-weight: 500 !important;
}

/* ── Section label (small-caps label above a section) ─────── */
.lbl {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 2.5rem 0 0.4rem 0;
    display: block;
}
/* ── Section heading ─────────────────────────────────────── */
.sh {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0 0 0.9rem 0;
    letter-spacing: -0.01em;
    line-height: 1.25;
}
/* ── Thin rule ───────────────────────────────────────────── */
.hr { border: none; border-top: 1px solid #E2E8F0; margin: 1.5rem 0; }

/* ── Annotation / synthesis note ────────────────────────── */
.ann {
    border-left: 2px solid #CBD5E1;
    padding: 6px 12px;
    margin: 0.25rem 0 1.5rem 0;
    font-size: 0.8rem;
    color: #64748B;
    line-height: 1.65;
}
.ann strong { color: #334155; font-weight: 600; }

/* ── Warning annotation ──────────────────────────────────── */
.ann-warn {
    border-left: 2px solid #FCA5A5;
    padding: 6px 12px;
    margin: 0.25rem 0 1.5rem 0;
    font-size: 0.8rem;
    color: #7F1D1D;
    line-height: 1.65;
    background: #FFF5F5;
    border-radius: 0 4px 4px 0;
}
.ann-warn strong { color: #991B1B; font-weight: 600; }

/* ── Pipeline step rows ──────────────────────────────────── */
.step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 11px 14px;
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    margin-bottom: 6px;
    transition: border-color 0.15s;
}
.step:hover { border-color: #C7D2DC; }
.step-n {
    flex-shrink: 0;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #0F172A;
    color: white;
    font-size: 0.68rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 2px;
}
.step-title { font-size: 0.86rem; font-weight: 600; color: #0F172A; }
.step-desc  { font-size: 0.78rem; color: #64748B; line-height: 1.5; margin-top: 3px; }
.tag {
    display: inline-block;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-left: 5px;
    vertical-align: middle;
    text-transform: uppercase;
}
.tag-era5  { background: #EFF6FF; color: #1D4ED8; border: 1px solid #BFDBFE; }
.tag-gwa   { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
.tag-synth { background: #F5F3FF; color: #6D28D9; border: 1px solid #DDD6FE; }
</style>
""", unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding-bottom:1.25rem; margin-bottom:0.5rem; border-bottom:1px solid #E2E8F0;">
  <p style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
            color:#94A3B8; margin:0 0 6px 0;">Wind Resource Tool</p>
  <h1 style="font-size:1.7rem; font-weight:700; color:#0F172A; margin:0; letter-spacing:-0.03em;
             line-height:1.15;">ERA5 × Global Wind Atlas</h1>
  <p style="font-size:0.9rem; color:#64748B; margin:6px 0 0 0; line-height:1.4;">
    ERA5 hourly reanalysis &nbsp;·&nbsp; GWA spatial accuracy &nbsp;·&nbsp;
    <strong style="color:#0F172A;">150 m</strong> hub height &nbsp;·&nbsp; onshore
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("About this tool"):
    st.markdown("""
This tool synthesises a long-term hourly wind speed time series at **150 m hub height**
for any onshore location. It combines two complementary data sources:

- **[ERA5](https://open-meteo.com)** reanalysis (~28 km grid) — provides realistic
  temporal variability: storms, seasonal cycles, diurnal patterns.
- **[Global Wind Atlas](https://globalwindatlas.info)** (250 m grid) — provides local
  spatial accuracy, encoding terrain and roughness effects through Weibull statistics.

A Weibull quantile transform is applied so the final time series matches GWA's
locally-calibrated speed distribution, while preserving ERA5's hour-by-hour sequence.
Sub-hourly output (30-min / 10-min) is stochastically disaggregated via AR(1) and is
synthetic — not a real measurement record.

> **Onshore use only.** Offshore or open-ocean sites are not supported.
    """)

# ── Session state ─────────────────────────────────────────────────────────────
for _key in ("era5_node", "gwa_node", "_prev_lat", "_prev_lon"):
    if _key not in st.session_state:
        st.session_state[_key] = None
# Default lat/lon for the WGS84 number inputs (updated by map clicks)
if "lat_input" not in st.session_state:
    st.session_state["lat_input"] = -31.9505
if "lon_input" not in st.session_state:
    st.session_state["lon_input"] = 115.8605

# Apply any pending map click BEFORE the sidebar widgets render.
# (Streamlit forbids setting a widget key after it has been instantiated,
# so we stage the click in _pending_* and apply it here at the top of the run.)
if st.session_state.get("_pending_lat") is not None:
    st.session_state["lat_input"] = st.session_state.pop("_pending_lat")
    st.session_state["lon_input"] = st.session_state.pop("_pending_lon")

# ── EPSG coordinate presets ───────────────────────────────────────────────────
_EPSG_OPTIONS = {
    "WGS84 / Geographic (EPSG:4326)": None,
    "GDA2020 / MGA Zone 49 (EPSG:7849)": 7849,
    "GDA2020 / MGA Zone 50 (EPSG:7850)": 7850,
    "GDA2020 / MGA Zone 51 (EPSG:7851)": 7851,
    "GDA94 / MGA Zone 49 (EPSG:28349)": 28349,
    "GDA94 / MGA Zone 50 (EPSG:28350)": 28350,
    "GDA94 / MGA Zone 51 (EPSG:28351)": 28351,
    "UTM Zone 50S (EPSG:32750)": 32750,
    "UTM Zone 51S (EPSG:32751)": 32751,
    "Custom EPSG code": "custom",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:0.5rem 0 0.3rem 0;">Location</p>', unsafe_allow_html=True)

    crs_choice = st.selectbox(
        "Coordinate system",
        options=list(_EPSG_OPTIONS.keys()),
        index=0,
        help="Select the coordinate reference system for your input coordinates.",
    )
    epsg_code = _EPSG_OPTIONS[crs_choice]
    if crs_choice == "Custom EPSG code":
        epsg_code = st.number_input("EPSG code", value=32750, min_value=1, max_value=99999, step=1)

    is_projected = epsg_code is not None and epsg_code != "custom"

    if is_projected:
        easting = st.number_input(
            "Easting (m)", value=386000.0, step=100.0, format="%.1f",
        )
        northing = st.number_input(
            "Northing (m)", value=6464000.0, step=100.0, format="%.1f",
        )
        try:
            transformer = Transformer.from_crs(
                f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True
            )
            lon, lat = transformer.transform(easting, northing)
            lat, lon = round(lat, 6), round(lon, 6)
            st.caption(f"→ WGS84: **{lat:.5f}°N, {lon:.5f}°E**")
        except Exception as _e:
            st.error(f"Coordinate conversion failed: {_e}")
            lat, lon = -31.9505, 115.8605
    else:
        lat = st.number_input(
            "Latitude", min_value=-90.0, max_value=90.0,
            step=0.0001, format="%.4f", key="lat_input",
        )
        lon = st.number_input(
            "Longitude", min_value=-180.0, max_value=180.0,
            step=0.0001, format="%.4f", key="lon_input",
        )
        st.caption("Or click the map to set location.")

    if (st.session_state["_prev_lat"], st.session_state["_prev_lon"]) != (lat, lon):
        st.session_state["era5_node"] = None
        st.session_state["gwa_node"] = None
        st.session_state["_prev_lat"] = lat
        st.session_state["_prev_lon"] = lon

    st.divider()
    tz_detected = detect_timezone(lat, lon)
    use_utc = st.checkbox("Show times in UTC", value=False)
    tz_display = "UTC" if use_utc else tz_detected
    st.caption(
        f"Detected: **{tz_detected}**"
        + ("  *(UTC override active)*" if use_utc else "")
    )

    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:0.25rem 0 0.3rem 0;">ERA5 Period</p>', unsafe_allow_html=True)
    _end_max = _LATEST_YEAR
    _end_min = 1980
    end_year = st.number_input(
        "End year", value=_LATEST_YEAR, min_value=_end_min, max_value=_end_max, step=1,
    )
    n_years = st.slider(
        "Number of years", min_value=1, max_value=min(end_year - 1979, 20),
        value=min(10, end_year - 1979),
    )
    start_year = end_year - n_years + 1
    st.caption(f"Period: **{start_year}–{end_year}** ({n_years} yr)")
    _START_YEAR, _END_YEAR = start_year, end_year

    st.info(
        f"**ERA5 period:** {_START_YEAR}–{_END_YEAR}\n\n"
        f"{n_years} years · hourly · {tz_display}"
    )
    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:0.25rem 0 0.3rem 0;">Output Resolution</p>', unsafe_allow_html=True)
    resolution = st.selectbox(
        "Temporal resolution",
        options=["Hourly", "30-min (fake)", "10-min (fake)"],
        index=0,
        help=(
            "Sub-hourly options are stochastically disaggregated from ERA5 hourly "
            "data — they are NOT real measurements. Labelled '(fake)' as a reminder. "
            "Each run produces one plausible realisation using per-hour turbulence "
            "intensity and an AR(1) process calibrated to the site's autocorrelation."
        ),
    )
    st.divider()
    run_btn = st.button("🚀  Fetch & Process Data", type="primary", use_container_width=True)

# ── Friendly display label (strips "(fake)" for headings) ────────────────────
res_label = resolution.replace(" (fake)", "")

# ── Map + method description ──────────────────────────────────────────────────
col_map, col_method = st.columns([3, 2])

with col_map:
    st.markdown('<span class="lbl">Site Location</span>', unsafe_allow_html=True)
    m = folium.Map(location=[lat, lon], zoom_start=9, tiles=ESRI_TILES, attr=ESRI_ATTR)

    # Input site — red filled circle with white border
    folium.CircleMarker(
        [lat, lon],
        radius=9,
        color="white",
        weight=2,
        fill=True,
        fill_color="#EF4444",
        fill_opacity=1.0,
        tooltip=f"Input site: {lat:.4f}°N, {lon:.4f}°E",
        popup=f"<b>Input site</b><br>{lat:.4f}°N, {lon:.4f}°E",
    ).add_to(m)

    if st.session_state["era5_node"]:
        elat, elon = st.session_state["era5_node"]
        folium.CircleMarker(
            [elat, elon],
            radius=8,
            color="white",
            weight=2,
            fill=True,
            fill_color="#3B82F6",
            fill_opacity=1.0,
            tooltip=f"ERA5 grid node (~0.25°): {elat:.4f}°N, {elon:.4f}°E",
            popup=f"<b>ERA5 grid node</b><br>{elat:.4f}°N, {elon:.4f}°E<br>~28 km resolution",
        ).add_to(m)
        if (elat, elon) != (lat, lon):
            folium.PolyLine(
                [[lat, lon], [elat, elon]],
                color="white", weight=1.5, dash_array="5",
                opacity=0.7,
                tooltip="Site → ERA5 node",
            ).add_to(m)

    if st.session_state["gwa_node"]:
        glat, glon = st.session_state["gwa_node"]
        folium.CircleMarker(
            [glat, glon],
            radius=8,
            color="white",
            weight=2,
            fill=True,
            fill_color="#F59E0B",
            fill_opacity=1.0,
            tooltip=f"GWA grid node (250 m): {glat:.4f}°N, {glon:.4f}°E",
            popup=f"<b>GWA grid node</b><br>{glat:.4f}°N, {glon:.4f}°E<br>250 m resolution",
        ).add_to(m)

    _map_out = st_folium(
        m, height=430, use_container_width=True,
        returned_objects=["last_clicked"], key="site_map",
    )

    # Handle map clicks (WGS84 mode only).
    # Write to _pending_* staging keys — NOT the widget keys directly —
    # because widgets have already rendered by this point in the script.
    if not is_projected and _map_out and _map_out.get("last_clicked"):
        _click = _map_out["last_clicked"]
        _clat = round(_click["lat"], 4)
        _clng = round(_click["lng"], 4)
        if (_clat, _clng) != (
            round(st.session_state["lat_input"], 4),
            round(st.session_state["lon_input"], 4),
        ):
            st.session_state["_pending_lat"] = _clat
            st.session_state["_pending_lon"] = _clng
            st.session_state["era5_node"] = None
            st.session_state["gwa_node"] = None
            st.rerun()

    if st.session_state["era5_node"] or st.session_state["gwa_node"]:
        st.caption("🔴 Input site  🔵 ERA5 grid node (~0.25°, ~28 km)  🟡 GWA grid node (250 m)")
    elif is_projected:
        st.caption("Click Fetch & Process Data to show ERA5 and GWA grid nodes.")
    else:
        st.caption("Click the map to move the site · Fetch & Process Data to show grid nodes.")

with col_method:
    st.markdown('<span class="lbl">Processing Pipeline</span><p class="sh">How the synthesis works</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step">
      <div class="step-n">1</div>
      <div>
        <div class="step-title">ERA5 @ 100 m <span class="tag tag-era5">ERA5</span></div>
        <div class="step-desc">10-year hourly reanalysis via Open-Meteo (~28 km grid). Real temporal variability — storms, seasons, diurnal cycles.</div>
      </div>
    </div>
    <div class="step">
      <div class="step-n">2</div>
      <div>
        <div class="step-title">Height extrapolation → 150 m <span class="tag tag-era5">ERA5</span><span class="tag tag-gwa">GWA</span></div>
        <div class="step-desc">Power-law with a diurnal shear exponent α. Shape from ERA5 10m/100m ratio; mean magnitude calibrated to GWA 100m/150m shear.</div>
      </div>
    </div>
    <div class="step">
      <div class="step-n">3</div>
      <div>
        <div class="step-title">Weibull correction <span class="tag tag-synth">Synthesised</span></div>
        <div class="step-desc">Quantile transform morphs the ERA5 distribution to match GWA 150 m Weibull (A, k). Rank order and all temporal patterns are preserved exactly.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Equations"):
        st.markdown(r"""
**Height extrapolation:**
$$V_{150}(t) = V_{100}(t) \times \left(\frac{150}{100}\right)^{\alpha(h)}$$

**Weibull quantile transform:**
$$V^* = A_{\text{GWA}} \times \left(\frac{V}{A_{\text{ERA5}}}\right)^{k_{\text{ERA5}} / k_{\text{GWA}}}$$

*α(h) varies by hour — stronger at night (stable BL), weaker by day (convective mixing).*
        """)

# ── Results section ───────────────────────────────────────────────────────────
if run_btn:
    try:
        with st.spinner(f"Fetching {_END_YEAR - _START_YEAR + 1} years of ERA5 data…"):
            df_era5_utc, era5_lat, era5_lon = fetch_era5(lat, lon, _START_YEAR, _END_YEAR)

        with st.spinner("Fetching Global Wind Atlas data…"):
            site_roughness, rough_source = fetch_site_roughness(lat, lon)
            gwc, heights, gwa_lat, gwa_lon = fetch_gwa(lat, lon, site_roughness)

        # Store grid node coordinates for the map (persists across re-renders)
        st.session_state["era5_node"] = (era5_lat, era5_lon)
        if gwa_lat is not None and gwa_lon is not None:
            st.session_state["gwa_node"] = (gwa_lat, gwa_lon)

        # Convert timestamps to display timezone before pipeline so that
        # diurnal hour grouping uses local hours (physically correct)
        df_era5 = localise_df(df_era5_utc, tz_display)

        with st.spinner("Running processing pipeline…"):
            df, meta = run_pipeline(df_era5, gwc, heights)

        tz_label = "UTC" if tz_display == "UTC" else f"{tz_detected} (UTC{_tz_offset_str(df.index)})"

        # ── Sub-hourly disaggregation ─────────────────────────────────────────
        df_sub = None
        sub_info = None
        if res_label != "Hourly":
            res_min = int(res_label.split("-")[0])
            with st.spinner(f"Disaggregating to {res_label}…"):
                df_sub, sub_info = disaggregate_subhourly(df, res_min)

        # Persist wind results so the AEP section survives re-renders without re-fetching
        st.session_state["aep_df"] = df
        st.session_state["aep_lat"] = lat
        st.session_state["aep_lon"] = lon

        n_records = len(df_sub) if df_sub is not None else len(df)
        st.success(
            f"Done — {n_records:,} {res_label.lower()} records  "
            f"({_START_YEAR}–{_END_YEAR} · {tz_label})"
        )

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown('<span class="lbl">Results</span><p class="sh">Summary Statistics</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric(
            "ERA5 raw @ 100 m",
            f"{meta['mean_era5_100']:.2f} m/s",
        )
        c2.metric(
            f"ERA5 height-extrap. @ 150 m",
            f"{meta['mean_era5_150_raw']:.2f} m/s",
            delta=f"{meta['mean_era5_150_raw'] - meta['mean_era5_100']:+.2f} m/s",
        )
        c3.metric(
            "GWA-corrected @ 150 m",
            f"{meta['mean_corrected']:.2f} m/s",
            delta=f"{meta['mean_corrected'] - meta['mean_era5_150_raw']:+.2f} m/s vs extrap.",
        )

        c4, c5, c6 = st.columns(3)
        c4.metric("GWA mean @ 100 m", f"{meta['mean_gwa_100']:.2f} m/s")
        c5.metric("GWA mean @ 150 m", f"{meta['mean_gwa_150']:.2f} m/s")
        c6.metric(
            "Wind shear α (100→150 m)",
            f"{meta['alpha_mean']:.3f}",
            delta=f"α 50→150 m = {meta['alpha_50_150']:.3f}" if meta["alpha_50_150"] else None,
            delta_color="off",
        )

        _rough_used = meta.get("gwa_roughness_used")
        _rough_note = (
            f" GWA Weibull uses roughness class <strong>r = {_rough_used:.3f} m</strong> "
            f"(source: {rough_source}). Small residual vs GWA website is normal — "
            f"GWA applies additional terrain/orography corrections at 250 m resolution."
            if _rough_used is not None else ""
        )
        st.markdown(f"""
        <div class="ann">
        <strong>Reading these numbers:</strong> ERA5 raw 100 m is the original reanalysis.
        ERA5 extrap. 150 m applies the diurnal power-law shear. GWA-corrected 150 m is the
        final synthesised output — Weibull-transformed to match GWA's site statistics.{_rough_note}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Weibull Parameters"):
            wb_tbl = pd.DataFrame(
                {
                    "Parameter": ["A — scale (m/s)", "k — shape"],
                    "ERA5 150 m (extrapolated)": [
                        f"{meta['A_era5_150']:.3f}",
                        f"{meta['k_era5_150']:.3f}",
                    ],
                    "GWA target 150 m": [
                        f"{meta['A_gwa_150']:.3f}",
                        f"{meta['k_gwa_150']:.3f}",
                    ],
                }
            ).set_index("Parameter")
            st.dataframe(wb_tbl, use_container_width=True)
            st.markdown("""
            <div class="ann" style="margin-top:10px;">
            The quantile transform maps each ERA5 150 m value to the equivalent quantile
            in the GWA Weibull — rank order (and all temporal patterns) are preserved
            while the speed distribution is reshaped to match GWA's A and k.
            </div>
            """, unsafe_allow_html=True)

        # ── Sub-hourly disaggregation panel ──────────────────────────────────
        if df_sub is not None and sub_info is not None:
            st.markdown(
                f'<span class="lbl">Sub-hourly</span><p class="sh">Disaggregation — {res_label} (stochastic)</p>',
                unsafe_allow_html=True,
            )

            st.markdown("""
            <div class="ann-warn">
            <strong>Not real measurements.</strong> Sub-hourly values are stochastically
            generated via AR(1) — each run is one plausible realisation, not the actual
            historical record.
            </div>
            """, unsafe_allow_html=True)

            si = sub_info
            sb1, sb2, sb3, sb4 = st.columns(4)
            sb1.metric("Output resolution", res_label)
            sb2.metric("Mean TI @ 150 m", f"{si['mean_TI_150']*100:.1f} %")
            sb3.metric("Mean σ_u @ 150 m", f"{si['mean_sigma']:.2f} m/s")
            sb4.metric("AR(1) φ per step", f"{si['phi_sub']:.3f}")

            with st.expander("Disaggregation method"):
                st.markdown(
                    f"""
**Turbulence intensity source:** {si['ti_method']}

The per-hour σᵤ at 150 m is estimated from the ERA5 gust factor at 10 m:

$$TI_{{10m}} = \\frac{{V_{{gust}}/V_{{10m}} - 1}}{{3.5}}, \\quad
  TI_{{150m}} = TI_{{10m}} \\times \\left(\\frac{{10}}{{150}}\\right)^{{0.11}}$$

The standard deviation for {res_label} *mean* output is then reduced from
instantaneous TI using the ratio of the integral time scale (≈ 350 s at 150 m)
to the averaging period ({si['resolution_min'] * 60} s):

$$\\sigma_{{\\text{{{res_label}}}}} = TI_{{150m}} \\times V_{{150m}} \\times
  \\sqrt{{T_{{int}} / T_{{avg}}}} = \\times {si['spectral_factor']:.2f}\\; \\text{{of instantaneous}}$$

A **continuous AR(1) process** is then generated at {res_label} timesteps across
the full 10-year record. The AR(1) coefficient (φ = {si['phi_sub']:.3f} per {si['resolution_min']}-min step)
is derived from the hourly autocorrelation assuming a continuous-time
Ornstein-Uhlenbeck process. Finally, each hourly block's noise is
mean-corrected to exactly preserve the ERA5 hourly means.
                    """
                )

            # Sample plot: 5 days from middle of record
            mid = len(df) // 2
            n_days = 5
            sample_h = df["ws_150m_corrected"].iloc[mid : mid + 24 * n_days]
            sample_sub = df_sub["ws_150m_subhourly"].loc[
                sample_h.index[0] : sample_h.index[-1]
            ]

            fig_sub = go.Figure()
            fig_sub.add_trace(
                go.Scatter(
                    x=sample_sub.index,
                    y=sample_sub.values,
                    mode="lines",
                    name=f"{res_label} stochastic (fake)",
                    line=dict(color="#10B981", width=0.9),
                )
            )
            fig_sub.add_trace(
                go.Scatter(
                    x=sample_h.index,
                    y=sample_h.values,
                    mode="lines+markers",
                    name="Hourly ERA5+GWA",
                    line=dict(color="#0F172A", width=2.0, dash="dash"),
                    marker=dict(size=4),
                )
            )
            fig_sub.update_layout(
                template="plotly_white",
                title=dict(text=f"{res_label} stochastic disaggregation — 5-day sample (fake)", font=dict(size=13, color="#0F172A")),
                xaxis_title=f"Date/Time ({tz_display})",
                yaxis_title="Wind Speed @ 150 m (m/s)",
                height=320,
                margin=dict(t=40, b=50, l=55, r=20),
                legend=dict(orientation="h", y=-0.3),
                font=dict(color="#64748B", size=11),
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

        # ── Diurnal shear profile ─────────────────────────────────────────────
        st.markdown('<span class="lbl">Shear</span><p class="sh">Wind Shear Exponent α — 100 m → 150 m</p>', unsafe_allow_html=True)

        da = meta["diurnal_alpha"]
        alpha_min, alpha_max = float(da.min()), float(da.max())

        sh_cols = st.columns(4) if meta["alpha_50_150"] else st.columns(3)
        sh_cols[0].metric("α (GWA 100→150 m)", f"{meta['alpha_mean']:.3f}")
        if meta["alpha_50_150"]:
            sh_cols[1].metric("α (GWA 50→150 m)", f"{meta['alpha_50_150']:.3f}")
            sh_cols[2].metric(f"Min α  (hour {int(da.idxmin()):02d}:00)", f"{alpha_min:.3f}")
            sh_cols[3].metric(f"Max α  (hour {int(da.idxmax()):02d}:00)", f"{alpha_max:.3f}")
        else:
            sh_cols[1].metric(f"Min α  (hour {int(da.idxmin()):02d}:00)", f"{alpha_min:.3f}")
            sh_cols[2].metric(f"Max α  (hour {int(da.idxmax()):02d}:00)", f"{alpha_max:.3f}")

        with st.expander("How is shear calculated?"):
            _gwa50_eq = ""
            if meta["alpha_50_150"] and meta["mean_gwa_50"]:
                _gwa50_eq = f"""
**GWA 50→150 m (supplementary):**

$$\\alpha_{{50\\text{{-}}150}} = \\frac{{\\ln({meta['mean_gwa_150']:.2f}/{meta['mean_gwa_50']:.2f})}}{{\\ln(150/50)}}
= {meta['alpha_50_150']:.3f}$$

This spans a wider height range and tends to be closer to the standard wind industry
value of ~0.2. It is shown for reference only — the 100→150 m α is used for extrapolation
because it most accurately represents the shear in the layer we are extrapolating across.
"""
            st.markdown(
                f"""
The shear exponent α is **diurnal** — it varies hour-by-hour, not a single constant.

**Step 1 — Mean magnitude from GWA (100→150 m)**

$$\\alpha_{{\\text{{mean}}}} = \\frac{{\\ln(V_{{150}}/V_{{100}})}}{{\\ln(150/100)}}
= \\frac{{\\ln({meta['mean_gwa_150']:.2f}/{meta['mean_gwa_100']:.2f})}}{{\\ln(1.5)}}
= {meta['alpha_mean']:.3f}$$
{_gwa50_eq}
**Step 2 — Diurnal pattern from ERA5**
The hourly shape of α is inferred from ERA5 10m/100m ratios — stable nights produce
stronger shear (higher α); convective days reduce it. This pattern is normalised so its
mean equals α_mean from Step 1.

**Result** — every hourly record extrapolated with its own hour-specific α:

$$V_{{150}}(t) = V_{{100}}(t) \\times \\left(\\frac{{150}}{{100}}\\right)^{{\\alpha(h)}}
\\quad h = \\text{{hour of day ({tz_display})}}$$

Diurnal range: **{alpha_min:.3f} – {alpha_max:.3f}** ({
"low" if (alpha_max - alpha_min) < 0.05 else "moderate" if (alpha_max - alpha_min) < 0.15 else "strong"
} signal).
                """
            )

        fig_alpha = go.Figure()
        fig_alpha.add_trace(
            go.Scatter(
                x=da.index.tolist(),
                y=da.values.tolist(),
                mode="lines+markers",
                line=dict(color="#4F46E5", width=2.5),
                marker=dict(size=6, color="#4F46E5"),
                name="Diurnal α",
                fill="tozeroy",
                fillcolor="rgba(79,70,229,0.07)",
            )
        )
        fig_alpha.add_hline(
            y=meta["alpha_mean"],
            line_dash="dash",
            line_color="#94A3B8",
            annotation_text=f"GWA mean α = {meta['alpha_mean']:.3f}",
            annotation_position="top right",
            annotation_font_color="#64748B",
        )
        fig_alpha.update_layout(
            template="plotly_white",
            xaxis=dict(title=f"Hour of Day ({tz_display})", tickmode="linear", tick0=0, dtick=3, gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Shear Exponent α", gridcolor="rgba(0,0,0,0.05)"),
            height=280,
            margin=dict(t=15, b=40, l=55, r=20),
            showlegend=False,
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(fig_alpha, use_container_width=True)
        _a50_note = (
            f" The 50→150 m span (α = {meta['alpha_50_150']:.3f}) is typically closer to the "
            f"industry-standard ≈ 0.2 — the 100→150 m value is lower because shear decreases "
            f"with height in a well-mixed boundary layer."
            if meta["alpha_50_150"] else ""
        )
        st.markdown(f"""
        <div class="ann">
        <strong>Mean α ({meta['alpha_mean']:.3f})</strong> is anchored to GWA's 100→150 m speed ratio,
        which sets the long-term shear used for height extrapolation. Hourly variation comes
        from ERA5's 10m/100m ratio, capturing the real stability cycle.{_a50_note}
        </div>
        """, unsafe_allow_html=True)

        # ── Monthly mean time series ──────────────────────────────────────────
        st.markdown('<span class="lbl">Time Series</span><p class="sh">Monthly Mean Wind Speed</p>', unsafe_allow_html=True)

        monthly = df.resample("ME").mean()[
            ["ws_100m", "ws_150m_raw", "ws_150m_corrected"]
        ]
        monthly.columns = [
            "ERA5 raw 100 m",
            "ERA5 extrap. 150 m",
            "GWA-corrected 150 m",
        ]

        fig_monthly = go.Figure()
        palette = ["#CBD5E1", "#94A3B8", "#4F46E5"]
        widths   = [1.2, 1.2, 2.2]
        for col, colour, w in zip(monthly.columns, palette, widths):
            fig_monthly.add_trace(
                go.Scatter(
                    x=monthly.index,
                    y=monthly[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colour, width=w),
                )
            )
        fig_monthly.update_layout(
            template="plotly_white",
            xaxis=dict(title="Date", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Mean Wind Speed (m/s)", gridcolor="rgba(0,0,0,0.05)"),
            height=330,
            margin=dict(t=15, b=60, l=55, r=20),
            legend=dict(orientation="h", y=-0.3),
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # ── Mean monthly seasonality ──────────────────────────────────────────
        st.markdown('<p class="sh" style="margin-top:1.5rem;">Mean Monthly Seasonality (all years averaged)</p>', unsafe_allow_html=True)
        month_avg = df.copy()
        month_avg["month"] = month_avg.index.month
        seasonal = (
            month_avg.groupby("month")[
                ["ws_100m", "ws_150m_corrected"]
            ]
            .mean()
        )
        seasonal.columns = ["ERA5 raw 100 m", "GWA-corrected 150 m"]

        fig_seasonal = go.Figure()
        for col, colour in zip(seasonal.columns, ["#CBD5E1", "#4F46E5"]):
            fig_seasonal.add_trace(
                go.Bar(
                    x=MONTH_LABELS,
                    y=seasonal[col].values,
                    name=col,
                    marker_color=colour,
                )
            )
        fig_seasonal.update_layout(
            template="plotly_white",
            barmode="group",
            xaxis=dict(title="Month", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Mean Wind Speed (m/s)", gridcolor="rgba(0,0,0,0.05)"),
            height=300,
            margin=dict(t=15, b=60, l=55, r=20),
            legend=dict(orientation="h", y=-0.3),
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

        # ── Weibull distribution ──────────────────────────────────────────────
        st.markdown('<span class="lbl">Distribution</span><p class="sh">Wind Speed Distribution @ 150 m</p>', unsafe_allow_html=True)
        st.caption(
            "Bars show empirical frequency; dashed line is the GWA Weibull target. "
            "The green bars should closely follow the dashed line."
        )

        bins = np.arange(0, 31, 0.5)
        bc = (bins[:-1] + bins[1:]) / 2
        bin_w = 0.5

        h_raw, _ = np.histogram(
            df["ws_150m_raw"].dropna(), bins=bins, density=True
        )
        h_corr, _ = np.histogram(
            df["ws_150m_corrected"].dropna(), bins=bins, density=True
        )
        pdf_gwa = weibull_min.pdf(bc, c=meta["k_gwa_150"], scale=meta["A_gwa_150"])

        fig_wb = go.Figure()
        fig_wb.add_trace(
            go.Bar(
                x=bc,
                y=h_raw * bin_w,
                name="ERA5 extrap. 150 m",
                marker_color="rgba(148,163,184,0.45)",
                width=bin_w,
            )
        )
        fig_wb.add_trace(
            go.Bar(
                x=bc,
                y=h_corr * bin_w,
                name="GWA-corrected 150 m",
                marker_color="rgba(79,70,229,0.55)",
                width=bin_w,
            )
        )
        fig_wb.add_trace(
            go.Scatter(
                x=bc,
                y=pdf_gwa * bin_w,
                mode="lines",
                name="GWA Weibull target",
                line=dict(color="#0F172A", width=2.0, dash="dash"),
            )
        )
        fig_wb.update_layout(
            template="plotly_white",
            barmode="overlay",
            xaxis=dict(title="Wind Speed (m/s)", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Probability", gridcolor="rgba(0,0,0,0.05)"),
            height=330,
            margin=dict(t=15, b=60, l=55, r=20),
            legend=dict(orientation="h", y=-0.3),
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(fig_wb, use_container_width=True)
        st.markdown(f"""
        <div class="ann">
        Blue bars = ERA5 extrapolated (pre-correction). Green bars = GWA-corrected output.
        Red dashed line = GWA Weibull target (A = {meta['A_gwa_150']:.2f} m/s, k = {meta['k_gwa_150']:.2f}).
        Green should track the dashed line closely.
        </div>
        """, unsafe_allow_html=True)

        # ── Daily time series ─────────────────────────────────────────────────
        st.markdown('<span class="lbl">Full Record</span><p class="sh">Daily Mean Wind Speed</p>', unsafe_allow_html=True)

        daily = df.resample("D").mean()[["ws_100m", "ws_150m_corrected"]]
        daily.columns = ["ERA5 raw 100 m", "GWA-corrected 150 m"]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=daily.index, y=daily["ERA5 raw 100 m"],
            mode="lines", name="ERA5 raw 100 m",
            line=dict(color="#CBD5E1", width=0.9),
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily.index, y=daily["GWA-corrected 150 m"],
            mode="lines", name="GWA-corrected 150 m",
            line=dict(color="#4F46E5", width=1.2),
        ))
        if df_sub is not None:
            daily_sub = df_sub["ws_150m_subhourly"].resample("D").mean()
            fig_ts.add_trace(go.Scatter(
                x=daily_sub.index, y=daily_sub.values,
                mode="lines", name=f"{res_label} daily mean (fake)",
                line=dict(color="#10B981", width=0.8, dash="dot"),
            ))
        fig_ts.update_layout(
            template="plotly_white",
            xaxis=dict(title="Date", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Wind Speed (m/s)", gridcolor="rgba(0,0,0,0.05)"),
            height=310,
            margin=dict(t=15, b=60, l=55, r=20),
            legend=dict(orientation="h", y=-0.3),
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(fig_ts, use_container_width=True)
        if df_sub is not None:
            st.markdown("""
            <div class="ann">
            The dotted green line is the sub-hourly data resampled to daily means. Because
            the disaggregation is mean-preserving, it overlaps exactly with the GWA-corrected
            hourly — confirming the AR(1) process introduces no bias.
            </div>
            """, unsafe_allow_html=True)

        # ── Example day ───────────────────────────────────────────────────────
        if df_sub is not None:
            st.markdown(
                f'<span class="lbl">Example Day</span>'
                f'<p class="sh">ERA5 vs GWA-corrected vs {res_label} (fake) — Jan 1, {_START_YEAR}</p>',
                unsafe_allow_html=True,
            )

            # First Jan 1 in the dataset
            jan1 = pd.Timestamp(f"{_START_YEAR}-01-01", tz=df.index.tz)
            jan1_end = pd.Timestamp(f"{_START_YEAR}-01-01 23:59", tz=df.index.tz)
            day_h = df.loc[jan1:jan1_end]
            day_sub = df_sub["ws_150m_subhourly"].loc[jan1:jan1_end]

            fig_day = go.Figure()

            # ERA5 raw 100 m — grey steps
            fig_day.add_trace(go.Scatter(
                x=day_h.index, y=day_h["ws_100m"],
                mode="lines", name="ERA5 raw 100 m",
                line=dict(color="#CBD5E1", width=2.0, shape="hv"),
            ))
            # GWA-corrected 150 m — indigo steps
            fig_day.add_trace(go.Scatter(
                x=day_h.index, y=day_h["ws_150m_corrected"],
                mode="lines", name="GWA-corrected 150 m",
                line=dict(color="#4F46E5", width=2.5, shape="hv"),
            ))
            # Sub-hourly fake — emerald continuous line
            fig_day.add_trace(go.Scatter(
                x=day_sub.index, y=day_sub.values,
                mode="lines", name=f"{res_label} (fake)",
                line=dict(color="#10B981", width=1.2),
            ))

            fig_day.update_layout(
                template="plotly_white",
                xaxis=dict(
                    title=f"Time ({tz_display})",
                    gridcolor="rgba(0,0,0,0.05)",
                    tickformat="%H:%M",
                ),
                yaxis=dict(title="Wind Speed (m/s)", gridcolor="rgba(0,0,0,0.05)"),
                height=340,
                margin=dict(t=15, b=60, l=55, r=20),
                legend=dict(orientation="h", y=-0.3),
                font=dict(color="#64748B", size=11),
            )
            st.plotly_chart(fig_day, use_container_width=True)
            st.markdown(f"""
            <div class="ann">
            Stepped lines show the hourly ERA5 values — constant within each hour.
            The {res_label} green trace shows the stochastic variation generated
            within each hour. The green line's block means exactly equal the indigo
            GWA-corrected value for that hour.
            </div>
            """, unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.markdown('<span class="lbl">Export</span><p class="sh">Download</p>', unsafe_allow_html=True)

        # Helper: strip tz offset from index, keep it only in the column name
        _tz_suffix = f"UTC{_tz_offset_str(df.index)}" if tz_display != "UTC" else "UTC"
        def _prep_index(frame):
            out = frame.copy()
            out.index = out.index.tz_localize(None)
            out.index.name = f"datetime_{_tz_suffix}"
            return out

        _has_wd = "wd_100m" in df.columns

        # Build metadata header lines (# prefix so parsers can skip easily)
        _era5_node = st.session_state.get("era5_node")
        _gwa_node  = st.session_state.get("gwa_node")
        _header_lines = [
            "# ERA5 x GWA Wind Resource Synthesis — synthesised time series",
            "#",
            f"# Site (input):       {lat:.4f}N, {lon:.4f}E",
            (f"# ERA5 grid node:     {_era5_node[0]:.4f}N, {_era5_node[1]:.4f}E  (~0.25 deg grid, ~28 km)"
             if _era5_node else "# ERA5 grid node:     unknown (fetch data to populate)"),
            (f"# GWA grid node:      {_gwa_node[0]:.4f}N, {_gwa_node[1]:.4f}E  (250 m grid)"
             if _gwa_node else "# GWA grid node:      unknown (fetch data to populate)"),
            f"# Timezone:           {tz_display} ({_tz_suffix})",
            f"# Period:             {_START_YEAR}-01-01 to {_END_YEAR}-12-31",
            "#",
            "# DATA SOURCE:        SYNTHESISED — NOT A MEASUREMENT RECORD",
            "# Wind speeds are derived by combining ERA5 reanalysis (temporal variability)",
            "# with Global Wind Atlas statistics (spatial accuracy via Weibull correction).",
            "# Sub-hourly values (if present) are stochastic disaggregations, not observations.",
            "# Results are indicative only. Do not use as a substitute for on-site measurement.",
            "#",
        ]
        _file_header = "\n".join(_header_lines) + "\n"

        if df_sub is not None:
            # Sub-hourly speed + hourly direction repeated for each sub-hourly step
            dl = _prep_index(df_sub[["ws_150m_subhourly"]].copy())
            dl.columns = ["gwa_corrected_ws_150m_ms"]
            dl["gwa_corrected_ws_150m_ms"] = dl["gwa_corrected_ws_150m_ms"].round(1)
            if _has_wd:
                # Forward-fill hourly direction to sub-hourly timestamps
                wd_sub = df["wd_100m"].reindex(df_sub.index, method="ffill")
                dl["era5_wd_100m_deg"] = wd_sub.round(0).astype(int).values
            csv_bytes = (_file_header + dl.to_csv()).encode()
            dl_label = f"Download {res_label} time series (CSV)"
            dl_caption = (
                f"ws: GWA-corrected 150 m (m/s, 1 dp, {res_label} stochastic fake) · "
                f"wd: ERA5 100 m (°, hourly repeated) · Timestamps: {_tz_suffix}"
            )
            dl_fname = f"wind_150m_{res_label}_{lat:.4f}_{lon:.4f}_{_START_YEAR}_{_END_YEAR}.csv"
        else:
            # Hourly: speed columns + wind direction
            cols = ["ws_100m", "ws_150m_raw", "ws_150m_corrected"]
            col_names = ["era5_ws_100m_ms", "era5_ws_150m_extrap_ms", "gwa_corrected_ws_150m_ms"]
            if _has_wd:
                cols.append("wd_100m")
                col_names.append("era5_wd_100m_deg")
            dl = _prep_index(df[cols])
            dl.columns = col_names
            dl[["era5_ws_100m_ms", "era5_ws_150m_extrap_ms", "gwa_corrected_ws_150m_ms"]] = (
                dl[["era5_ws_100m_ms", "era5_ws_150m_extrap_ms", "gwa_corrected_ws_150m_ms"]].round(1)
            )
            if _has_wd:
                dl["era5_wd_100m_deg"] = dl["era5_wd_100m_deg"].round(0).astype(int)
            csv_bytes = (_file_header + dl.to_csv()).encode()
            dl_label = "Download hourly time series (CSV)"
            dl_caption = (
                f"Speeds in m/s (1 dp) · Direction in ° (0–360) · Timestamps: {_tz_suffix}"
            )
            dl_fname = f"wind_150m_hourly_{lat:.4f}_{lon:.4f}_{_START_YEAR}_{_END_YEAR}.csv"

        st.download_button(
            label=dl_label,
            data=csv_bytes,
            file_name=dl_fname,
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(dl_caption)

    except requests.HTTPError as exc:
        st.error(f"API request failed: {exc}")
    except ValueError as exc:
        st.error(f"Data processing error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        raise

# ── AEP Calculator ────────────────────────────────────────────────────────────
if "aep_df" in st.session_state:
    _df_aep = st.session_state["aep_df"]
    _aep_lat = st.session_state.get("aep_lat", lat)
    _aep_lon = st.session_state.get("aep_lon", lon)

    _pc_df = load_power_curves()
    _wake_df = load_wake_matrix()

    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown(
        '<span class="lbl">AEP</span>'
        '<p class="sh">Annual Energy Production Calculator</p>',
        unsafe_allow_html=True,
    )

    if _pc_df is None:
        st.info(
            "No power curves loaded. Add `data/power_curves.xlsx` to enable AEP calculation. "
            "Format: column 1 = wind speed (m/s), remaining columns = power (kW) per WTG, "
            "row 1 = WTG names."
        )
    else:
        _ac1, _ac2, _ac3 = st.columns([2, 1, 1])
        with _ac1:
            _selected_wtg = st.selectbox("Wind turbine model", options=list(_pc_df.columns))
        with _ac2:
            _nameplate_mw = st.number_input(
                "Nameplate capacity (MW)",
                min_value=0.1, max_value=5000.0, value=100.0, step=0.5, format="%.1f",
            )
        with _ac3:
            _apply_wake = st.checkbox(
                "Apply wake losses",
                value=_wake_df is not None,
                disabled=_wake_df is None,
                help="Requires data/wake_loss_matrix.xlsx" if _wake_df is None else "",
            )

        # Power curve preview scaled to nameplate
        _ws_c = _pc_df.index.values
        _kw_c = _pc_df[_selected_wtg].values
        _rated_kw_preview = float(_kw_c.max())
        _mw_c = _kw_c / _rated_kw_preview * _nameplate_mw

        _fig_pc = go.Figure()
        _fig_pc.add_trace(go.Scatter(
            x=_ws_c, y=_mw_c,
            mode="lines",
            line=dict(color="#4F46E5", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(79,70,229,0.07)",
            name=f"{_selected_wtg} — scaled to {_nameplate_mw:.1f} MW",
        ))
        _fig_pc.update_layout(
            template="plotly_white",
            xaxis=dict(title="Wind Speed (m/s)", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Power (MW)", gridcolor="rgba(0,0,0,0.05)"),
            height=240,
            margin=dict(t=10, b=40, l=55, r=20),
            legend=dict(orientation="h", y=-0.4),
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(_fig_pc, use_container_width=True)

        # Run AEP calculation
        _ws_aep = _df_aep["ws_150m_corrected"].dropna()
        _aep = calc_aep(
            _ws_aep, _pc_df, _selected_wtg, _nameplate_mw,
            _wake_df if _apply_wake else None,
        )

        # Metrics
        def _energy_str(mwh: float) -> str:
            return f"{mwh / 1000:.2f} GWh/yr" if mwh >= 1000 else f"{mwh:.0f} MWh/yr"

        _am1, _am2, _am3, _am4 = st.columns(4)
        _am1.metric("Gross AEP", _energy_str(_aep["gross_aep_mwh"]))
        _am2.metric(
            "Net AEP",
            _energy_str(_aep["net_aep_mwh"]),
            delta=f"−{_aep['mean_wake_pct']:.1f}% wake" if _aep["mean_wake_pct"] > 0 else None,
            delta_color="inverse",
        )
        _am3.metric("Capacity Factor", f"{_aep['capacity_factor'] * 100:.1f}%")
        _am4.metric("Rated (from curve)", f"{_aep['rated_kw'] / 1000:.2f} MW")

        st.markdown(
            f'<div class="ann">'
            f'<strong>{_selected_wtg}</strong> normalised to rated output and scaled to '
            f'<strong>{_nameplate_mw:.1f} MW</strong> nameplate capacity. '
            f'{"Wake losses applied via 2D interpolation of the wind-speed × nameplate capacity matrix." if _apply_wake and _wake_df is not None else "No wake correction applied — add <code>data/wake_loss_matrix.xlsx</code> to enable."}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Monthly net energy
        st.markdown(
            '<p class="sh" style="margin-top:1.5rem;">Mean Monthly Net Energy</p>',
            unsafe_allow_html=True,
        )
        _monthly_net = _aep["net_mw_ts"].resample("ME").sum()
        _monthly_gross = _aep["gross_mw_ts"].resample("ME").sum()
        _mnet_avg = _monthly_net.groupby(_monthly_net.index.month).mean()
        _mgross_avg = _monthly_gross.groupby(_monthly_gross.index.month).mean()

        _fig_maep = go.Figure()
        _fig_maep.add_trace(go.Bar(
            x=MONTH_LABELS, y=_mgross_avg.values,
            name="Gross energy", marker_color="rgba(148,163,184,0.5)",
        ))
        _fig_maep.add_trace(go.Bar(
            x=MONTH_LABELS, y=_mnet_avg.values,
            name="Net energy (after wake)", marker_color="rgba(79,70,229,0.7)",
        ))
        _fig_maep.update_layout(
            template="plotly_white",
            barmode="overlay",
            xaxis=dict(title="Month", gridcolor="rgba(0,0,0,0.05)"),
            yaxis=dict(title="Mean Monthly Energy (MWh)", gridcolor="rgba(0,0,0,0.05)"),
            height=290,
            margin=dict(t=10, b=60, l=55, r=20),
            legend=dict(orientation="h", y=-0.3),
            font=dict(color="#64748B", size=11),
        )
        st.plotly_chart(_fig_maep, use_container_width=True)

        # AEP CSV download
        _aep_tz_sfx = (
            f"UTC{_tz_offset_str(_df_aep.index)}" if tz_display != "UTC" else "UTC"
        )
        _aep_out = pd.DataFrame(
            {
                "wind_speed_ms": _ws_aep.round(1),
                "gross_power_mw": _aep["gross_mw_ts"].round(3),
                "wake_loss_pct": _aep["wake_pct_ts"].round(2),
                "net_power_mw": _aep["net_mw_ts"].round(3),
            }
        )
        _aep_out.index = _aep_out.index.tz_localize(None)
        _aep_out.index.name = f"datetime_{_aep_tz_sfx}"
        _aep_fname = (
            f"aep_{_selected_wtg.replace(' ', '_')}_{_nameplate_mw:.1f}MW"
            f"_{_aep_lat:.4f}_{_aep_lon:.4f}.csv"
        )
        st.download_button(
            label="Download AEP time series (CSV)",
            data=_aep_out.to_csv().encode(),
            file_name=_aep_fname,
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            f"{_selected_wtg} · {_nameplate_mw:.1f} MW nameplate · "
            f"{'Wake-corrected' if _apply_wake and _wake_df is not None else 'No wake correction'} · "
            f"{_aep['n_years']:.1f}-year record"
        )
