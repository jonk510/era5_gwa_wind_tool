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

import io
import re
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from pyproj import Transformer
from report_gen import generate_pdf_report
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


def _prevailing_wd(df: pd.DataFrame) -> float | None:
    """Vector-mean wind direction from ERA5 100m records."""
    if "wd_100m" not in df.columns:
        return None
    wd_rad = np.radians(df["wd_100m"].dropna().values)
    if len(wd_rad) == 0:
        return None
    return float(np.degrees(np.arctan2(np.sin(wd_rad).mean(), np.cos(wd_rad).mean())) % 360)


def _offset_latlon(lat: float, lon: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Return (lat, lon) displaced from origin by distance_m in compass bearing (degrees CW from N)."""
    R = 6371000.0
    d = distance_m / R
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    brng = np.radians(bearing_deg % 360)
    lat2 = np.arcsin(np.sin(lat_r) * np.cos(d) + np.cos(lat_r) * np.sin(d) * np.cos(brng))
    lon2 = lon_r + np.arctan2(
        np.sin(brng) * np.sin(d) * np.cos(lat_r),
        np.cos(d) - np.sin(lat_r) * np.sin(lat2),
    )
    return float(np.degrees(lat2)), float(np.degrees(lon2))


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


# ── GWC / height helpers ──────────────────────────────────────────────────────

def _gwa_at_height(gwc: dict, h: float) -> tuple[float, float, float]:
    """Return (A, k, mean_ws) at height h by log-linear interpolation between GWC heights."""
    heights = sorted(gwc.keys())
    if h in gwc:
        d = gwc[h]
        return d["A"], d["k"], d["mean"]
    if h <= heights[0]:
        d = gwc[heights[0]]
        return d["A"], d["k"], d["mean"]
    if h >= heights[-1]:
        d = gwc[heights[-1]]
        return d["A"], d["k"], d["mean"]
    h_lo = max(hv for hv in heights if hv < h)
    h_hi = min(hv for hv in heights if hv > h)
    t = np.log(h / h_lo) / np.log(h_hi / h_lo)
    A = gwc[h_lo]["A"] * (gwc[h_hi]["A"] / gwc[h_lo]["A"]) ** t
    k = gwc[h_lo]["k"] + t * (gwc[h_hi]["k"] - gwc[h_lo]["k"])
    mean = gwc[h_lo]["mean"] * (gwc[h_hi]["mean"] / gwc[h_lo]["mean"]) ** t
    return float(A), float(k), float(mean)


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
    Fetch hourly ERA5 wind + 2 m temperature from Open-Meteo for the given year range.
    Returns (df, era5_lat, era5_lon, elevation) — lat/lon are the actual ERA5 grid node
    coordinates; elevation (m ASL) is the site elevation reported by Open-Meteo.
    """
    r = requests.get(
        OPENMETEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "hourly": "wind_speed_100m,wind_speed_10m,wind_gusts_10m,wind_direction_100m,temperature_2m,boundary_layer_height",
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        },
        timeout=120,
    )
    r.raise_for_status()
    d = r.json()
    era5_lat = d.get("latitude", lat)
    era5_lon = d.get("longitude", lon)
    elevation = float(d.get("elevation", 0.0))
    _n = len(d["hourly"]["time"])
    df = pd.DataFrame(
        {
            "ws_100m": d["hourly"]["wind_speed_100m"],
            "ws_10m": d["hourly"]["wind_speed_10m"],
            "ws_gust_10m": d["hourly"]["wind_gusts_10m"],
            "wd_100m": d["hourly"]["wind_direction_100m"],
            "temp_2m": d["hourly"]["temperature_2m"],
            "blh": d["hourly"].get("boundary_layer_height", [None] * _n),
        },
        index=pd.to_datetime(d["hourly"]["time"]),
    )
    # Drop only on core wind columns so BLH NaNs don't discard valid records
    return df.dropna(subset=["ws_100m", "ws_10m", "ws_gust_10m", "wd_100m", "temp_2m"]), era5_lat, era5_lon, elevation


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
def fetch_site_roughness(
    lat: float,
    lon: float,
    prevailing_wd_deg: float | None = None,
    fetch_radius_m: int = 10000,
) -> tuple[float, str]:
    """
    Estimate aerodynamic roughness length (m) from OpenStreetMap land-use tags.

    When prevailing_wd_deg is supplied, queries a 120°-wide sector centred on
    that direction (the upwind fetch) out to fetch_radius_m.  Falls back to an
    omnidirectional circle of the same radius if no direction is available.

    fetch_radius_m defaults to 10 km; callers should pass 100 × hub_height
    (capped at 15 km) for a physically motivated fetch distance.

    Returns (roughness_m, description_string).
    """
    _fallback = (0.1, "default (r = 0.10 m — open land, no OSM data)")

    if prevailing_wd_deg is not None:
        # Build a closed sector polygon: site → arc at fetch_radius → back to site
        half = 60  # ±60° gives a 120° sector covering the dominant upwind quadrant
        wd = prevailing_wd_deg % 360
        pts = [f"{lat} {lon}"]
        for angle in range(int(wd - half), int(wd + half) + 1, 5):
            p = _offset_latlon(lat, lon, float(angle % 360), fetch_radius_m)
            pts.append(f"{p[0]:.6f} {p[1]:.6f}")
        pts.append(f"{lat} {lon}")
        poly_str = " ".join(pts)
        query = (
            f'[out:json][timeout:20];'
            f'(way["landuse"](poly:"{poly_str}");'
            f'way["natural"](poly:"{poly_str}"););'
            f'out tags;'
        )
        fetch_desc = f"upwind sector {wd:.0f}° ±60°, {fetch_radius_m/1000:.0f} km"
    else:
        query = (
            f'[out:json][timeout:20];'
            f'(way["landuse"](around:{fetch_radius_m},{lat},{lon});'
            f'way["natural"](around:{fetch_radius_m},{lat},{lon}););'
            f'out tags;'
        )
        fetch_desc = f"omnidirectional {fetch_radius_m/1000:.0f} km radius"

    try:
        resp = requests.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
            headers={"User-Agent": "ERA5-GWA-WindTool/1.0 (wind resource research)"},
            timeout=25,
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
        return median_r, f"OpenStreetMap ({fetch_desc}, dominant: {dominant}, r = {median_r:.4f} m)"

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
    density_series: pd.Series | None = None,
) -> dict:
    """Apply power curve + optional air-density correction + optional wake losses; return AEP stats.

    Air density correction (IEC method): V_eq = V × (ρ/1.225)^⅓ — equivalent wind speed
    at standard density is used for power curve lookup. Wake lookup uses actual wind speed.
    """
    ws_arr = pc_df.index.values.astype(float)
    kw_arr = pc_df[wtg].values.astype(float)
    rated_kw = float(kw_arr.max())

    ws_vals = ws.values

    # Air-density correction: convert actual wind speed to standard-density equivalent
    if density_series is not None:
        rho = density_series.reindex(ws.index).fillna(1.225).values
        ws_eq = ws_vals * (rho / 1.225) ** (1.0 / 3.0)
    else:
        rho = np.full(len(ws_vals), 1.225)
        ws_eq = ws_vals

    gross_kw = np.interp(ws_eq, ws_arr, kw_arr, left=0.0, right=rated_kw)
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
            np.clip(ws_vals, ws_bins.min(), ws_bins.max()),  # actual speed for wake
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
        "ws_equiv_ts": pd.Series(ws_eq, index=ws.index),
        "gross_aep_mwh": gross_aep,
        "net_aep_mwh": net_aep,
        "mean_wake_pct": mean_wake,
        "capacity_factor": cf,
        "rated_kw": rated_kw,
        "n_years": n_years,
        "mean_air_density": float(rho.mean()),
    }


# ── Processing pipeline ───────────────────────────────────────────────────────

def run_pipeline(
    df_era5: pd.DataFrame,
    gwc: dict,
    heights: list,
    hub_height: float = 150.0,
    elevation: float = 0.0,
    amplitude_scale: float = 1.0,
    alpha_clip_lo: float = 0.02,
    alpha_clip_hi: float = 1.0,
    alpha_mean_lo: float = 0.03,
    alpha_mean_hi: float = 0.70,
) -> tuple:
    """
    Full processing pipeline for an arbitrary hub height:
      1. Derive mean-shear alpha from GWA 100m → hub_height means.
      2. Build diurnal alpha profile using ERA5 10m/100m shear pattern.
      3. Extrapolate ERA5 100m → hub_height with diurnal alpha.
      4. Fit Weibull to ERA5 hub_height estimate.
      5. Apply Weibull quantile transform to match GWA hub_height Weibull.
      6. Compute per-timestep air density at hub height from ERA5 T₂ₘ + ISA lapse.

    Returns (df, meta). Internal df columns ws_150m_raw / ws_150m_corrected
    always refer to the hub height (the name is historical).
    """
    h100 = min(heights, key=lambda x: abs(x - 100))
    g100 = gwc[h100]
    mean_gwa_100 = g100["mean"]

    # GWA Weibull at hub height (interpolated between GWC heights if needed)
    A_gwa_hub, k_gwa_hub, mean_gwa_hub = _gwa_at_height(gwc, hub_height)

    # ── Step 1: GWA-derived mean shear exponent (100 m → hub_height) ─────────
    log_h_ratio = np.log(hub_height / h100)
    if abs(log_h_ratio) < 1e-6:
        # hub_height ≈ ERA5 reference height — use a neighbouring GWC height
        other = [hv for hv in heights if abs(hv - h100) > 5]
        if other:
            h_ref2 = min(other, key=lambda x: abs(x - h100))
            alpha_mean = np.log(gwc[h_ref2]["mean"] / mean_gwa_100) / np.log(h_ref2 / h100)
        else:
            alpha_mean = 0.2
    else:
        alpha_mean = np.log(mean_gwa_hub / mean_gwa_100) / log_h_ratio

    _alpha_raw = alpha_mean
    alpha_mean = max(alpha_mean_lo, min(alpha_mean, alpha_mean_hi))

    # Supplementary: α from ~50m → hub_height (reference only, shown in UI)
    alpha_50_150 = None
    mean_gwa_50 = None
    h50_candidates = [h for h in heights if 40 <= h <= 75]
    if h50_candidates:
        h50 = min(h50_candidates, key=lambda x: abs(x - 50))
        g50 = gwc.get(h50)
        if g50 and g50["mean"] > 0 and g50["mean"] < mean_gwa_100:
            mean_gwa_50 = g50["mean"]
            log_50_hub = np.log(hub_height / h50)
            if abs(log_50_hub) > 1e-6:
                alpha_50_150 = np.log(mean_gwa_hub / g50["mean"]) / log_50_hub

    # ── Step 2: Per-timestep stability → α(t) ────────────────────────────────
    df = df_era5.copy()
    df["hour"] = df.index.hour

    # Compute ERA5 10/100m instantaneous shear std — used to calibrate amplitude
    # for both BLH and hour-of-day paths.
    valid = df[(df["ws_10m"] > 0.5) & (df["ws_100m"] > 0.5)].copy()
    if len(valid) > 10:
        valid["log_shear"] = np.log(valid["ws_100m"] / valid["ws_10m"]) / np.log(100 / 10)
        alpha_std_era5 = max(float(valid["log_shear"].clip(0.0, 1.5).std()), 0.03)
    else:
        alpha_std_era5 = 0.10

    has_blh = "blh" in df.columns and df["blh"].notna().mean() > 0.5

    if has_blh:
        # BLH stability index: hub_height / BLH
        #   > 1 → rotor above stable nocturnal BL top → strong shear (high α)
        #   ≈ 0 → rotor well inside convective mixed layer → weak shear (low α)
        blh_filled = df["blh"].fillna(df["blh"].median()).clip(lower=10.0)
        si = (hub_height / blh_filled).clip(0.0, 5.0)
        si_std = float(si.std())
        si_norm = (si - si.mean()) / si_std if si_std > 1e-6 else pd.Series(0.0, index=df.index)
        df["alpha_h"] = (alpha_mean + amplitude_scale * alpha_std_era5 * si_norm).clip(
            alpha_clip_lo, alpha_clip_hi
        )
        alpha_method = "ERA5 boundary layer height (per-timestep)"
    else:
        # Fallback: hour-of-day climatological shear from ERA5 10m/100m ratio
        if len(valid) > 10:
            diurnal_era5 = (
                valid.groupby("hour")["log_shear"]
                .mean()
                .reindex(range(24), fill_value=alpha_mean)
            )
            deviation = diurnal_era5 - diurnal_era5.mean()
            diurnal_scaled = (alpha_mean + amplitude_scale * deviation).clip(
                alpha_clip_lo, alpha_clip_hi
            )
        else:
            diurnal_scaled = pd.Series(alpha_mean, index=range(24))
        df["alpha_h"] = df["hour"].map(diurnal_scaled).fillna(alpha_mean)
        alpha_method = "ERA5 10m/100m shear (hour-of-day, BLH unavailable)"

    # ── Step 3: Height extrapolation to hub_height ───────────────────────────
    df["ws_150m_raw"] = df["ws_100m"] * (hub_height / 100.0) ** df["alpha_h"]

    # ── Step 4: Fit Weibull to ERA5 at hub_height ────────────────────────────
    ws_clean = df["ws_150m_raw"].dropna()
    ws_clean = ws_clean[ws_clean > 0.1]
    k_era5_150, _, A_era5_150 = weibull_min.fit(ws_clean, floc=0)

    # ── Step 5: Weibull quantile transform ────────────────────────────────────
    v = df["ws_150m_raw"].clip(lower=0.01).values
    if A_era5_150 < 0.01:
        # Degenerate Weibull fit (near-zero wind site) — skip quantile transform
        df["ws_150m_corrected"] = df["ws_150m_raw"].clip(lower=0)
    else:
        df["ws_150m_corrected"] = A_gwa_hub * (v / A_era5_150) ** (k_era5_150 / k_gwa_hub)

    # ── Step 6: Air density at hub height ────────────────────────────────────
    # Hub height above sea level determines both pressure and temperature.
    # Temperature: ERA5 2 m value extrapolated up to hub using ISA lapse rate.
    # Pressure: standard barometric formula (ISA, constant lapse to tropopause).
    h_asl = elevation + hub_height
    if "temp_2m" in df.columns:
        T_hub_K = (df["temp_2m"] + 273.15) - 0.0065 * max(0.0, hub_height - 2.0)
    else:
        T_hub_K = 288.15 - 0.0065 * h_asl  # ISA fallback when no ERA5 temperature
    P_hub = 101325.0 * (1.0 - 2.2558e-5 * h_asl) ** 5.2559
    air_density_raw = P_hub / (287.05 * T_hub_K)
    df["air_density"] = air_density_raw.clip(0.9, 1.4)
    try:
        _density_clip_frac = float(((air_density_raw < 0.9) | (air_density_raw > 1.4)).mean())
    except AttributeError:
        _density_clip_frac = 0.0

    meta = {
        "hub_height": hub_height,
        "h100_used": h100,
        "h150_used": hub_height,
        "alpha_mean": alpha_mean,
        "alpha_raw": _alpha_raw,
        "alpha_50_150": alpha_50_150,
        "diurnal_alpha": df.groupby(df.index.hour)["alpha_h"].mean().reindex(range(24), fill_value=alpha_mean),
        "alpha_method": alpha_method,
        "mean_era5_100": df["ws_100m"].mean(),
        "mean_era5_150_raw": df["ws_150m_raw"].mean(),
        "mean_corrected": df["ws_150m_corrected"].mean(),
        "mean_gwa_50": mean_gwa_50,
        "mean_gwa_100": mean_gwa_100,
        "mean_gwa_150": mean_gwa_hub,
        "A_era5_150": A_era5_150,
        "k_era5_150": k_era5_150,
        "A_gwa_150": A_gwa_hub,
        "k_gwa_150": k_gwa_hub,
        "gwa_roughness_used": g100.get("roughness"),
        "mean_air_density": float(df["air_density"].mean()),
        "density_clip_frac": _density_clip_frac,
        "site_elevation": elevation,
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
st.markdown(f"""
<div style="padding-bottom:1.25rem; margin-bottom:0.5rem; border-bottom:1px solid #E2E8F0;">
  <p style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
            color:#94A3B8; margin:0 0 6px 0;">Wind Resource Tool</p>
  <h1 style="font-size:1.7rem; font-weight:700; color:#0F172A; margin:0; letter-spacing:-0.03em;
             line-height:1.15;">Synthetic wind data and Energy time series with wake approximation<br>
    <span style="font-size:1.1rem; font-weight:600; color:#475569;">(ERA5 × Global Wind Atlas)</span></h1>
  <p style="font-size:0.9rem; color:#64748B; margin:6px 0 0 0; line-height:1.4;">
    ERA5 hourly reanalysis &nbsp;·&nbsp; GWA spatial accuracy &nbsp;·&nbsp;
    <strong style="color:#0F172A;">{st.session_state.get('hub_height', 150):.0f} m</strong> hub height &nbsp;·&nbsp; onshore
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("About this tool — methodology & pipeline"):
    st.markdown("""
This tool produces a long-term hourly (or sub-hourly) wind speed time series at a
user-specified hub height for any onshore location, by fusing two complementary datasets:
**ERA5 reanalysis** for temporal variability and **Global Wind Atlas (GWA)** for local
spatial accuracy. The pipeline is described in full below.

---

### Data Sources

**ERA5 — temporal backbone**

ERA5 is the European Centre for Medium-Range Weather Forecasts (ECMWF) global atmospheric
reanalysis, covering 1940 to near-present at ~28 km horizontal resolution and 1-hour
timesteps. It is accessed here via the [Open-Meteo archive API](https://open-meteo.com),
which returns the following variables for the ERA5 grid node nearest to the input site:

- Wind speed at **10 m** and **100 m** above ground level (m/s)
- Wind direction at 100 m (°)
- Wind gust at 10 m (m/s) — used for sub-hourly turbulence estimation
- Air temperature at 2 m (°C) — used for air density calculation
- **Boundary layer height (BLH, m)** — the depth of the turbulent mixed layer, used as
  the primary atmospheric stability signal for per-timestep shear estimation

ERA5 captures the full temporal structure of the wind climate — inter-annual variability,
seasonal cycles, storm events, diurnal patterns, and calm periods — but its coarse
resolution means it cannot represent local terrain channelling, coastal effects, or
roughness changes at the sub-kilometre scale. ERA5 wind speeds are therefore systematically
biased relative to what a mast or turbine at the site would actually experience.

**Global Wind Atlas (GWA) — spatial calibration**

The [Global Wind Atlas](https://globalwindatlas.info) is produced by the Technical
University of Denmark (DTU) using WAsP mesoscale modelling driven by ERA5, downscaled to a
**250 m grid**. At each grid point GWA provides Weibull scale (A) and shape (k) parameters
at multiple heights (typically 10, 50, 100, 150, 200 m) for each of 12 wind direction
sectors, as well as sector frequencies. These statistics encode the effects of local terrain,
land cover, and roughness at much finer resolution than ERA5. Critically, GWA Weibull
parameters represent the *long-term mean* wind climate at the site — they have no temporal
dimension, but they are far more spatially accurate than ERA5 alone.

---

### Processing Pipeline

**Step 1 — Height extrapolation: ERA5 100 m → hub height**

ERA5 provides wind at 100 m. To reach the hub height (default 150 m, user-adjustable), a
**power-law extrapolation** is applied:

$$V_{hub}(t) = V_{100}(t) \\times \\left(\\frac{h_{hub}}{100}\\right)^{\\alpha(t)}$$

The shear exponent α varies **per timestep** (not hour-of-day) to capture individual
stability episodes:

- **Magnitude:** The long-term mean α is anchored to GWA. Specifically, the GWA mean wind
  speeds at 100 m and hub height (log-linearly interpolated from the GWC Weibull parameters
  at available heights) give: α_mean = ln(V_hub_GWA / V_100_GWA) / ln(h_hub / 100).
  This ensures the extrapolated mean matches GWA's locally-calibrated estimate of shear.

- **Per-timestep stability from ERA5 boundary layer height (primary method):** Each
  hourly ERA5 BLH value is used to compute a stability index SI(t) = h_hub / BLH(t).
  When BLH is below hub height (SI > 1), the rotor is above the stable nocturnal boundary
  layer — the classic low-level jet regime — and shear is strong (high α). When BLH greatly
  exceeds hub height (SI ≪ 1), the atmosphere is well-mixed and shear is low. The per-timestep
  α is then:

$$\\alpha(t) = \\alpha_{mean} + s \\cdot \\sigma_{\\alpha} \\cdot SI_{norm}(t)$$

  where σ_α is the standard deviation of ERA5 instantaneous 10–100 m shear (amplitude
  calibration), SI_norm is the zero-mean unit-std normalised stability index, and **s** is
  the amplitude scale parameter (default 1.0, adjustable in Advanced settings).

  Unlike a simple hour-of-day grouping, this approach captures episode-to-episode stability
  variability: a stormy winter night with deep BLH has low shear; a calm clear-sky winter
  night with BLH of 50 m has very high shear — even at the same clock hour.

- **Fallback (hour-of-day):** If BLH is unavailable, the diurnal shape of α is derived
  from the ERA5 10 m / 100 m wind speed ratio, grouped by hour of day and normalised to
  α_mean. The amplitude scale parameter is applied to the deviation in both cases.

**Step 2 — Weibull quantile transform: bias correction to GWA**

After height extrapolation, the ERA5-derived distribution at hub height will still differ
from GWA because of ERA5's spatial resolution bias. A **Weibull quantile transform**
re-shapes the ERA5 distribution to match GWA's locally-calibrated Weibull:

$$V^*(t) = A_{GWA} \\times \\left(\\frac{V_{hub}(t)}{A_{ERA5}}\\right)^{k_{ERA5}/k_{GWA}}$$

where A_ERA5 and k_ERA5 are fitted to the ERA5 hub-height series, and A_GWA and k_GWA come
from the GWA grid node (interpolated to hub height from the GWC file). This transform is
**rank-preserving**: the hour-by-hour sequence, storm timing, and seasonal patterns are all
unchanged — only the speed distribution is reshaped to match GWA. The result is a time
series that has ERA5's temporal fidelity and GWA's spatial accuracy.

**Roughness class and directional fetch:** The GWA GWC file contains Weibull parameters
for multiple roughness classes (0.0, 0.03, 0.1, 0.4, 3.0 m). The tool automatically
selects the appropriate roughness class using a directional upwind fetch query:

1. The ERA5 100 m vector-mean wind direction across the full record is computed to identify
   the prevailing inflow direction.
2. OpenStreetMap landuse and natural tags are queried within a **120°-wide sector polygon**
   in that upwind direction, at a radius of **100 × hub_height** (bounded between 5 km and
   15 km). For a 150 m turbine this is typically a 15 km radius sector — far larger than the
   old 500 m omnidirectional circle.
3. The dominant OSM tag determines roughness: water → r = 0.0003 m; beach/bare_rock →
   r = 0.025 m; all other land → r = 0.1 m. The closest GWC roughness class is then
   selected.
4. If BLH or direction data is unavailable, the query falls back to an omnidirectional
   circle at the same radius.

This directional approach targets the actual upwind fetch that determines the boundary-layer
profile reaching the turbine, rather than averaging over all directions equally.

---

### Sub-hourly Disaggregation (optional)

When 30-min or 10-min output is selected, hourly ERA5+GWA values are **stochastically
disaggregated** to produce a synthetic sub-hourly time series. This is a plausible
realisation of what the wind could have done within each hour — it is explicitly *not* the
real historical record and is labelled accordingly.

**Turbulence intensity (TI) at hub height**

Per-hour TI is estimated from the ERA5 gust factor at 10 m:

$$TI_{10m} = \\frac{V_{gust} / V_{10m} - 1}{3.5}$$

TI decreases with height following a standard power law:

$$TI_{hub} = TI_{10m} \\times \\left(\\frac{10}{h_{hub}}\\right)^{0.11}$$

When ERA5 gust data is unavailable, a fallback TI profile based on mean wind speed is used
(higher TI at low speeds, lower at high speeds, following IEC 61400-1 Class B behaviour).

**Standard deviation for sub-hourly means**

Instantaneous TI gives the standard deviation of instantaneous wind speed fluctuations.
For a *mean* over an averaging period T_avg, the variance is reduced by the ratio of the
integral time scale T_int to T_avg (von Kármán spectral theory):

$$\\sigma_{T_{avg}} = TI_{hub} \\times V_{hub} \\times \\sqrt{T_{int} / T_{avg}}$$

where T_int ≈ 350 s represents the integral length scale at hub height divided by a
typical wind speed. For 30-min means this gives a spectral reduction factor of ~0.24;
for 10-min means ~0.42.

**AR(1) process**

Sub-hourly wind speed values are generated using a first-order autoregressive (AR(1))
process — a discrete-time approximation to the Ornstein-Uhlenbeck continuous-time
stochastic process. The AR(1) coefficient φ per sub-hourly timestep is derived from the
ERA5 hourly autocorrelation assuming exponential decay: φ = exp(−Δt / T_int). Gaussian
noise scaled to σ_{T_avg} is added at each step. Each hourly block is then mean-corrected
so that the sub-hourly values average exactly to the ERA5+GWA hourly value — there is no
bias introduced by disaggregation.

---

### Advanced Settings

The following parameters can be adjusted in the **Advanced settings** expander in the
sidebar. Default values are calibrated to typical conditions; they can be fine-tuned if
site measurements are available (see *Calibration* below).

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| **Diurnal amplitude scale (s)** | 1.0 | 0.5 – 3.0 | Multiplier on the BLH stability signal. Increase above 1.0 if the tool underestimates the day/night wind speed swing versus measurements; decrease if the swing is exaggerated. |
| **α per-timestep clip (low)** | 0.02 | — | Minimum α applied at any single timestep. Prevents unphysically smooth shear profiles. |
| **α per-timestep clip (high)** | 1.0 | — | Maximum α applied at any single timestep. Strong low-level jets can produce α > 0.5 above 100 m. |
| **Mean α clamp (low)** | 0.03 | — | Minimum allowed value for α_mean (GWA-calibrated long-term average shear). |
| **Mean α clamp (high)** | 0.70 | — | Maximum allowed value for α_mean. |

---

### Calibration against site measurements

A standalone script `calibrate.py` is provided in the tool directory for fitting the
diurnal amplitude scale against site mast or LiDAR measurements. It requires:

- A measured wind speed CSV (any datetime-indexed format)
- The tool's hourly output CSV for the same site

Usage:
```
python calibrate.py --measured measured.csv --modelled model_output.csv [--plot]
```

The script aligns both records to a common hourly index, then uses `scipy.optimize`
(bounded 1-D minimisation) to find the `amplitude_scale` that minimises the RMSE between
the measured and modelled 24-hour mean diurnal wind speed profiles. With `--plot` it saves
a comparison chart and the RMSE-vs-scale curve as `calibration_result.png`.

The optimal scale value is reported at the end and can be directly entered into the
Advanced settings panel without re-running the ERA5 fetch or GWA correction.

---

### Limitations and appropriate use

- **Not a measured record.** Output is synthesised from modelled data. Uncertainty is
  greater than for a site-specific mast measurement campaign.
- **ERA5 grid spacing ~28 km.** Local terrain effects (ridgelines, valleys, coastal
  gradients) within this radius are not captured by ERA5 and only partially captured by GWA.
- **GWA is a long-term climatology.** The GWA Weibull parameters represent a multi-decadal
  mean; they have no inter-annual variability of their own. Year-to-year variation in the
  output comes entirely from ERA5.
- **ERA5 BLH at ~28 km resolution.** The boundary layer height used for stability is also
  ERA5-native resolution. Terrain-driven stability variations (valley drainage, coastal
  internal boundary layers) at sub-28 km scales are not resolved. The amplitude scale
  parameter exists partly to compensate for this systematic underestimation.
- **BLH approach requires BLH data.** If the Open-Meteo API does not return BLH (rare),
  the tool falls back to the hour-of-day ERA5 10/100 m shear grouping, which has less
  episode-to-episode accuracy.
- **Shear above 100 m.** Both the BLH approach and the 10–100 m fallback are calibrated
  at ERA5 native levels. The nocturnal low-level jet (NLLJ), which often peaks between
  50–300 m AGL, can produce shear exponents above 0.4–0.6 in the 100 m–hub height layer
  that neither ERA5 variable directly measures. The amplitude scale can be increased to
  partially compensate, but site mast data at or near hub height is the only reliable way
  to validate this.
- **Sub-hourly output is synthetic.** Each run produces one plausible realisation. Do not
  treat it as a real historical record or use it for fatigue-load analysis without
  understanding this limitation.
- **Onshore use only.** The roughness-class selection and GWA terrain modelling are not
  designed for offshore environments.

> Results are indicative and should be used to inform — not replace — site-specific
> measurement campaigns and bankable wind resource assessments.
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
if "hub_height" not in st.session_state:
    st.session_state["hub_height"] = 150

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
    app_mode = st.radio(
        "Mode",
        ["Single Site", "Batch"],
        horizontal=True,
        help="Single Site: analyse one location interactively.\nBatch: process multiple sites from an uploaded file.",
    )
    st.divider()

    if app_mode == "Single Site":
        st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:0.5rem 0 0.3rem 0;">Location</p>', unsafe_allow_html=True)

        st.text_input(
            "Site name",
            placeholder="e.g. Augusta WTG01",
            key="site_name_input",
        )

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
    else:
        # Batch mode — location and hub height come from the uploaded file
        lat, lon = -31.9505, 115.8605
        crs_choice = "WGS84 (lat/lon)"
        epsg_code = None
        is_projected = False
        tz_detected = "UTC"
        use_utc = False
        tz_display = "UTC"
        st.caption("📋 Location and hub height are read from the batch file. ERA5 period and resolution below apply to all sites.")

    st.divider()
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:0.25rem 0 0.3rem 0;">ERA5 Period</p>', unsafe_allow_html=True)
    _end_max = _LATEST_YEAR
    _end_min = 1980
    end_year = st.number_input(
        "End year", value=_LATEST_YEAR, min_value=_end_min, max_value=_end_max, step=1,
    )
    n_years = st.slider(
        "Number of years", min_value=1, max_value=min(end_year - 1979, 20),
        value=min(5, end_year - 1979),
    )
    start_year = end_year - n_years + 1
    st.caption(f"Period: **{start_year}–{end_year}** ({n_years} yr)")
    _START_YEAR, _END_YEAR = start_year, end_year

    if app_mode == "Single Site":
        st.info(
            f"**ERA5 period:** {_START_YEAR}–{_END_YEAR}\n\n"
            f"{n_years} years · hourly · {tz_display}"
        )
    else:
        st.info(
            f"**ERA5 period:** {_START_YEAR}–{_END_YEAR}\n\n"
            f"{n_years} years · hourly · timezone auto per site"
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
    st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:0.25rem 0 0.3rem 0;">Hub Height</p>', unsafe_allow_html=True)
    hub_height = st.number_input(
        "Hub height (m)",
        min_value=10,
        max_value=300,
        step=10,
        key="hub_height",
        help="Height for wind speed extrapolation and GWA Weibull correction. Used as the default in batch mode when no hub_height column is present.",
    )
    st.divider()
    with st.expander("⚙️ Advanced settings"):
        st.caption(
            "Adjust diurnal shear amplitude and α bounds. Leave at defaults unless "
            "calibrating against site measurements."
        )
        amplitude_scale = st.slider(
            "Diurnal amplitude scale",
            min_value=0.5, max_value=3.0, value=1.0, step=0.05,
            help=(
                "Multiplies the per-timestep α deviation from the long-term mean. "
                "1.0 = ERA5 native amplitude. Increase (e.g. 1.3–1.6) if measured "
                "diurnal variation exceeds model output."
            ),
        )
        _adv_c1, _adv_c2 = st.columns(2)
        with _adv_c1:
            alpha_clip_lo = st.number_input(
                "α per-timestep — lower clip", min_value=0.01, max_value=0.10,
                value=0.02, step=0.01, format="%.2f",
                help="Floor on instantaneous shear exponent. Prevents unphysically low shear.",
            )
            alpha_mean_lo = st.number_input(
                "Mean α — lower clamp", min_value=0.01, max_value=0.10,
                value=0.03, step=0.01, format="%.2f",
                help="Floor on the GWA-derived long-term mean α.",
            )
        with _adv_c2:
            alpha_clip_hi = st.number_input(
                "α per-timestep — upper clip", min_value=0.50, max_value=2.0,
                value=1.0, step=0.05, format="%.2f",
                help="Ceiling on instantaneous shear exponent.",
            )
            alpha_mean_hi = st.number_input(
                "Mean α — upper clamp", min_value=0.40, max_value=1.0,
                value=0.70, step=0.05, format="%.2f",
                help="Ceiling on the GWA-derived long-term mean α.",
            )
    st.divider()
    if app_mode == "Single Site":
        run_btn = st.button("🚀  Fetch & Process Data", type="primary", use_container_width=True)
    else:
        run_btn = False

# ── Friendly display label (strips "(fake)" for headings) ────────────────────
res_label = resolution.replace(" (fake)", "")

# ── Batch processing ──────────────────────────────────────────────────────────
if app_mode == "Batch":
    st.markdown(
        "Upload an **Excel or CSV** with columns `site_name`, `latitude`, `longitude`, "
        "`hub_height`, `turbine_type` (WGS84 decimal degrees). "
        "Add `name_plate` (or `nameplate_mw`) with the **total park capacity in MW** to enable wake loss "
        "estimation via the wind-speed × nameplate matrix — wake losses require nameplate > 8 MW "
        "(single-turbine values will yield 0% wake). "
        "ERA5 period and resolution are taken from the sidebar."
    )
    _bf = st.file_uploader("Site list", type=["xlsx", "csv"], label_visibility="collapsed")

    if _bf is not None:
        if st.session_state.get("_batch_fname") != _bf.name:
            st.session_state.pop("batch_zip", None)
            st.session_state.pop("batch_summary", None)
            st.session_state["_batch_fname"] = _bf.name

        try:
            _bdf_raw = (
                pd.read_csv(_bf) if _bf.name.lower().endswith(".csv")
                else pd.read_excel(_bf)
            )
            _bdf_raw.columns = [str(c).strip().lower() for c in _bdf_raw.columns]
            if "site_name" not in _bdf_raw.columns:
                _bdf_raw.insert(0, "site_name", [f"Site_{i+1}" for i in range(len(_bdf_raw))])

            _missing_cols = [c for c in ["latitude", "longitude"] if c not in _bdf_raw.columns]
            if _missing_cols:
                st.error(f"Missing required columns: {', '.join(_missing_cols)}")
            else:
                _has_wtg = "turbine_type" in _bdf_raw.columns
                # Accept "name_plate", "nameplate", or "nameplate_mw"
                if "nameplate_mw" in _bdf_raw.columns:
                    _cap_col = "nameplate_mw"
                elif "nameplate" in _bdf_raw.columns:
                    _cap_col = "nameplate"
                elif "name_plate" in _bdf_raw.columns:
                    _cap_col = "name_plate"
                else:
                    _cap_col = None
                _has_cap = _cap_col is not None
                _has_aep = _has_wtg
                _has_hh_col = "hub_height" in _bdf_raw.columns

                _keep_cols = ["site_name", "latitude", "longitude"]
                if _has_hh_col:
                    _keep_cols.append("hub_height")
                if _has_wtg:
                    _keep_cols.append("turbine_type")
                if _has_cap:
                    _keep_cols.append(_cap_col)

                _bdf = _bdf_raw[_keep_cols].dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
                st.dataframe(_bdf, use_container_width=True, hide_index=True)

                # Load and validate power curves if AEP columns present
                _b_pc_df = load_power_curves() if _has_aep else None
                _b_wake_df = load_wake_matrix() if _has_aep else None

                if _has_aep and _b_pc_df is None:
                    st.warning("AEP columns detected but `data/power_curves.xlsx` not found — AEP will be skipped.")
                    _has_aep = False

                if _has_aep:
                    _unknown_wtgs = sorted({
                        str(r["turbine_type"]).strip()
                        for _, r in _bdf.iterrows()
                        if pd.notna(r.get("turbine_type")) and str(r["turbine_type"]).strip() not in _b_pc_df.columns
                    })
                    if _unknown_wtgs:
                        st.warning(
                            f"Unknown turbine type(s): **{', '.join(_unknown_wtgs)}**. "
                            f"Available: {', '.join(_b_pc_df.columns)}. AEP skipped for those rows."
                        )

                st.caption(
                    f"{len(_bdf)} sites · ERA5 {_START_YEAR}–{_END_YEAR} · {res_label}"
                    + (" · per-site hub height" if _has_hh_col else f" · hub height {hub_height:.0f} m")
                    + (" · AEP enabled (wind + AEP CSV per site)" if _has_aep else "")
                )

                _batch_go = st.button("🚀 Process All Sites", type="primary", use_container_width=True)

                if _batch_go:
                    _zip_buf = io.BytesIO()
                    _prog = st.progress(0, text="Starting…")
                    _summary_rows, _batch_errors = [], []
                    _batch_site_data: dict = {}

                    with zipfile.ZipFile(_zip_buf, "w", zipfile.ZIP_DEFLATED) as _zf:
                        for _bi, _brow in _bdf.iterrows():
                            _bname = str(_brow["site_name"])
                            _bname_safe = re.sub(r'[/\\:*?"<>|]', '-', _bname).replace(' ', '_')
                            _blat, _blon = float(_brow["latitude"]), float(_brow["longitude"])
                            _prog.progress(_bi / len(_bdf), text=f"Processing {_bname} ({_bi + 1}/{len(_bdf)})…")

                            try:
                                _b_hub_h = float(_brow["hub_height"]) if _has_hh_col and pd.notna(_brow.get("hub_height")) else float(hub_height)
                                _b_hh_int = int(_b_hub_h)
                                _b_era5, _b_elat, _b_elon, _b_elevation = fetch_era5(_blat, _blon, _START_YEAR, _END_YEAR)
                                _b_prevailing = _prevailing_wd(_b_era5)
                                _b_fetch_radius = int(min(max(5000, 100 * _b_hub_h), 15000))
                                _b_rough, _ = fetch_site_roughness(_blat, _blon, _b_prevailing, _b_fetch_radius)
                                _b_gwc, _b_heights, _b_glat, _b_glon = fetch_gwa(_blat, _blon, _b_rough)
                                _b_tz = detect_timezone(_blat, _blon)
                                _b_tz_disp = "UTC" if use_utc else _b_tz
                                _b_df_era5 = localise_df(_b_era5, _b_tz_disp)
                                _b_df, _b_meta = run_pipeline(
                                    _b_df_era5, _b_gwc, _b_heights, hub_height=_b_hub_h, elevation=_b_elevation,
                                    amplitude_scale=amplitude_scale, alpha_clip_lo=alpha_clip_lo,
                                    alpha_clip_hi=alpha_clip_hi, alpha_mean_lo=alpha_mean_lo, alpha_mean_hi=alpha_mean_hi,
                                )
                                _b_tz_sfx = f"UTC{_tz_offset_str(_b_df.index)}" if _b_tz_disp != "UTC" else "UTC"

                                # Sub-hourly disaggregation (mirrors single-site behaviour)
                                _b_sub_df = None
                                if res_label != "Hourly":
                                    _b_res_min = int(res_label.split("-")[0])
                                    _b_sub_df, _ = disaggregate_subhourly(_b_df, _b_res_min)

                                # ── Wind CSV ───────────────────────────────────────────────────────
                                _b_hdr = "\n".join([
                                    f"# ERA5 x GWA Wind Resource Synthesis - {_bname}",
                                    "#",
                                    f"# Site:               {_bname}",
                                    f"# Latitude:           {_blat:.4f}",
                                    f"# Longitude:          {_blon:.4f}",
                                    f"# ERA5 grid node:     {_b_elat:.4f}N, {_b_elon:.4f}E  (~0.25 deg grid, ~28 km)",
                                    (f"# GWA grid node:      {_b_glat:.4f}N, {_b_glon:.4f}E  (250 m grid)"
                                     if _b_glat else "# GWA grid node:      unknown"),
                                    f"# Timezone:           {_b_tz_disp} ({_b_tz_sfx})",
                                    f"# Period:             {_START_YEAR}-01-01 to {_END_YEAR}-12-31",
                                    f"# Hub height:         {_b_hh_int} m",
                                    f"# GWA mean @ 100m:    {_b_meta['mean_gwa_100']:.2f} m/s",
                                    f"# GWA mean @ {_b_hh_int}m:    {_b_meta['mean_gwa_150']:.2f} m/s",
                                    f"# GWA-corrected mean: {_b_meta['mean_corrected']:.2f} m/s",
                                    f"# Wind shear alpha:     {_b_meta['alpha_mean']:.3f}  (100->{_b_hh_int} m)",
                                    "#",
                                    "# DATA SOURCE:        SYNTHESISED - NOT A MEASUREMENT RECORD",
                                    "#",
                                ]) + "\n"

                                _b_col_extrap = f"era5_ws_{_b_hh_int}m_extrap_ms"
                                _b_col_corr = f"gwa_corrected_ws_{_b_hh_int}m_ms"

                                if _b_sub_df is not None:
                                    # Sub-hourly: corrected wind at sub-hourly res, hourly direction/density repeated
                                    _bdl = _b_sub_df[["ws_150m_subhourly"]].copy()
                                    _bdl.columns = [_b_col_corr]
                                    _bdl[_b_col_corr] = _bdl[_b_col_corr].round(1)
                                    if "wd_100m" in _b_df.columns:
                                        _bdl["era5_wd_100m_deg"] = (
                                            _b_df["wd_100m"].reindex(_bdl.index, method="ffill").round(0).astype(int)
                                        )
                                    if "air_density" in _b_df.columns:
                                        _bdl["air_density_kg_m3"] = (
                                            _b_df["air_density"].reindex(_bdl.index, method="ffill").round(4)
                                        )
                                    _bdl.index = _bdl.index.tz_localize(None)
                                    _bdl.index.name = f"datetime_{_b_tz_sfx}"
                                else:
                                    # Hourly: all columns
                                    _bcols = ["ws_100m", "ws_150m_raw", "ws_150m_corrected"]
                                    _bcnames = ["era5_ws_100m_ms", _b_col_extrap, _b_col_corr]
                                    if "wd_100m" in _b_df.columns:
                                        _bcols.append("wd_100m")
                                        _bcnames.append("era5_wd_100m_deg")
                                    if "air_density" in _b_df.columns:
                                        _bcols.append("air_density")
                                        _bcnames.append("air_density_kg_m3")
                                    _bdl = _b_df[_bcols].copy()
                                    _bdl.columns = _bcnames
                                    _bdl.index = _bdl.index.tz_localize(None)
                                    _bdl.index.name = f"datetime_{_b_tz_sfx}"
                                    _bdl[["era5_ws_100m_ms", _b_col_extrap, _b_col_corr]] = (
                                        _bdl[["era5_ws_100m_ms", _b_col_extrap, _b_col_corr]].round(1)
                                    )
                                    if "era5_wd_100m_deg" in _bdl.columns:
                                        _bdl["era5_wd_100m_deg"] = _bdl["era5_wd_100m_deg"].round(0).astype(int)
                                    if "air_density_kg_m3" in _bdl.columns:
                                        _bdl["air_density_kg_m3"] = _bdl["air_density_kg_m3"].round(4)

                                _zf.writestr(
                                    f"{_bname_safe}_{_blat:.4f}_{_blon:.4f}_wind.csv",
                                    (_b_hdr + _bdl.to_csv()).encode(),
                                )

                                # ── AEP CSV (if turbine_type provided) ─────────────────────────
                                _sum_aep: dict = {}
                                _b_aep_store = None   # captured for PDF report
                                _b_wtg = str(_brow.get("turbine_type", "")).strip() if _has_aep else ""

                                if _has_aep and _b_wtg in _b_pc_df.columns:
                                    # nameplate_mw is optional — fall back to rated kW from curve
                                    _b_cap_raw = _brow.get(_cap_col) if _has_cap else None
                                    if _b_cap_raw is None or pd.isna(_b_cap_raw):
                                        _b_cap = float(_b_pc_df[_b_wtg].max()) / 1000.0
                                        _b_cap_note = "fallback: rated kW from power curve (single turbine) — wake losses will be zero; set name_plate to total park MW"
                                    else:
                                        _b_cap = float(_b_cap_raw)
                                        _b_cap_note = f"from name_plate column"

                                    if _b_sub_df is not None:
                                        _b_ws_aep = _b_sub_df["ws_150m_subhourly"].dropna()
                                        _b_density = (
                                            _b_df["air_density"].reindex(_b_ws_aep.index, method="ffill")
                                            if "air_density" in _b_df.columns else None
                                        )
                                    else:
                                        _b_ws_aep = _b_df["ws_150m_corrected"].dropna()
                                        _b_density = _b_df["air_density"] if "air_density" in _b_df.columns else None
                                    _b_aep = calc_aep(_b_ws_aep, _b_pc_df, _b_wtg, _b_cap, _b_wake_df, density_series=_b_density)
                                    _b_aep_store = {
                                        "gross_mw": _b_aep["gross_mw_ts"],
                                        "net_mw":   _b_aep["net_mw_ts"],
                                        "gross_aep_mwh": _b_aep["gross_aep_mwh"],
                                        "net_aep_mwh":   _b_aep["net_aep_mwh"],
                                        "mean_wake_pct": _b_aep["mean_wake_pct"],
                                        "capacity_factor": _b_aep["capacity_factor"],
                                        "rated_kw": _b_aep["rated_kw"],
                                        "n_years": _b_aep["n_years"],
                                        "mean_air_density": _b_aep["mean_air_density"],
                                    }

                                    def _b_es(mwh: float) -> str:
                                        return f"{mwh/1000:.2f} GWh/yr" if mwh >= 1000 else f"{mwh:.0f} MWh/yr"

                                    _b_wake_note = (
                                        "not applied (wake_loss_matrix.xlsx missing)" if _b_wake_df is None
                                        else f"applied — mean {_b_aep['mean_wake_pct']:.1f}%"
                                        + (" (zero because nameplate ≤ 8 MW — set name_plate to total park MW)" if _b_cap <= 8.0 else "")
                                    )
                                    _b_n_yr_display = (_b_df.index[-1] - _b_df.index[0]).days / 365.25
                                    _b_aep_hdr = "\n".join([
                                        f"# ERA5 x GWA Wind Resource Synthesis - AEP - {_bname}",
                                        "#",
                                        f"# Site:               {_bname}",
                                        f"# Latitude:           {_blat:.4f}",
                                        f"# Longitude:          {_blon:.4f}",
                                        f"# Timezone:           {_b_tz_disp} ({_b_tz_sfx})",
                                        f"# Wind record:        {_b_df.index[0].strftime('%Y-%m-%d')} to {_b_df.index[-1].strftime('%Y-%m-%d')} ({_b_n_yr_display:.1f} yr)",
                                        "#",
                                        f"# Wind turbine:       {_b_wtg}",
                                        f"# Rated capacity:     {_b_aep['rated_kw']/1000:.2f} MW (from power curve)",
                                        f"# Nameplate capacity: {_b_cap:.1f} MW - {_b_cap_note}",
                                        f"# Wake losses:        {_b_wake_note}",
                                        f"# Air density:        {_b_aep['mean_air_density']:.4f} kg/m3 mean at hub height (IEC V_eq = V*(rho/1.225)^(1/3) applied)",
                                        "#",
                                        f"# Gross AEP:          {_b_es(_b_aep['gross_aep_mwh'])}",
                                        f"# Net AEP:            {_b_es(_b_aep['net_aep_mwh'])}",
                                        f"# Mean wake loss:     {_b_aep['mean_wake_pct']:.1f} %",
                                        f"# Capacity factor:    {_b_aep['capacity_factor']*100:.1f} %",
                                        "#",
                                        "# INDICATIVE ONLY - wind speeds are synthesised from ERA5 + GWA, not measured.",
                                        "#",
                                    ]) + "\n"

                                    _b_aep_out = pd.DataFrame({
                                        "wind_speed_ms": _b_ws_aep.round(1),
                                        "equiv_wind_speed_ms": _b_aep["ws_equiv_ts"].round(1),
                                        "gross_power_mw": _b_aep["gross_mw_ts"].round(3),
                                        "wake_loss_pct": _b_aep["wake_pct_ts"].round(2),
                                        "net_power_mw": _b_aep["net_mw_ts"].round(3),
                                    })
                                    _b_aep_out.index = _b_aep_out.index.tz_localize(None)
                                    _b_aep_out.index.name = f"datetime_{_b_tz_sfx}"

                                    _zf.writestr(
                                        f"{_bname_safe}_{_blat:.4f}_{_blon:.4f}_aep.csv",
                                        (_b_aep_hdr + _b_aep_out.to_csv()).encode(),
                                    )

                                    _sum_aep = {
                                        "turbine_type": _b_wtg,
                                        "nameplate_mw": _b_cap,
                                        "gross_aep_mwh_yr": round(_b_aep["gross_aep_mwh"], 0),
                                        "net_aep_mwh_yr": round(_b_aep["net_aep_mwh"], 0),
                                        "capacity_factor_pct": round(_b_aep["capacity_factor"] * 100, 1),
                                        "mean_wake_loss_pct": round(_b_aep["mean_wake_pct"], 1),
                                        "mean_air_density_kg_m3": round(_b_aep["mean_air_density"], 4),
                                    }

                                _summary_rows.append({
                                    "site_name": _bname,
                                    "latitude": _blat,
                                    "longitude": _blon,
                                    "elevation_m_asl": round(_b_elevation, 0),
                                    "hub_height_m": _b_hh_int,
                                    "era5_grid_lat": round(_b_elat, 4),
                                    "era5_grid_lon": round(_b_elon, 4),
                                    "gwa_grid_lat": round(_b_glat, 4) if _b_glat else "",
                                    "gwa_grid_lon": round(_b_glon, 4) if _b_glon else "",
                                    "era5_mean_100m_ms": round(_b_meta["mean_era5_100"], 2),
                                    "gwa_mean_100m_ms": round(_b_meta["mean_gwa_100"], 2),
                                    "gwa_mean_hub_ms": round(_b_meta["mean_gwa_150"], 2),
                                    "gwa_corrected_mean_hub_ms": round(_b_meta["mean_corrected"], 2),
                                    "wind_shear_alpha": round(_b_meta["alpha_mean"], 3),
                                    "mean_air_density_kg_m3": round(_b_meta["mean_air_density"], 4),
                                    **_sum_aep,
                                })

                                # Store data for PDF report (minimum columns needed)
                                _store_cols = ["ws_150m_raw", "ws_150m_corrected", "ws_100m"]
                                if "air_density" in _b_df.columns:
                                    _store_cols.append("air_density")
                                _batch_site_data[_bname] = {
                                    "ws_raw":       _b_df["ws_150m_raw"].copy(),
                                    "ws_corr":      _b_df["ws_150m_corrected"].copy(),
                                    "ws_100m":      _b_df["ws_100m"].copy(),
                                    "air_density":  _b_df["air_density"].copy() if "air_density" in _b_df.columns else None,
                                    "meta":         _b_meta,
                                    "aep":          _b_aep_store,
                                    "wtg":          _b_wtg,
                                    "nameplate_mw": _b_cap if _b_aep_store is not None else None,
                                    "lat":          _blat,
                                    "lon":          _blon,
                                    "elevation":    _b_elevation,
                                }

                            except Exception as _be:
                                _batch_errors.append(f"{_bname}: {_be}")

                        if _summary_rows:
                            _zf.writestr(
                                "_summary.csv",
                                pd.DataFrame(_summary_rows).to_csv(index=False).encode(),
                            )

                    _prog.progress(1.0, text=f"Done — {len(_summary_rows)} sites processed.")
                    if _batch_errors:
                        st.warning(f"{len(_batch_errors)} site(s) failed:\n" + "\n".join(_batch_errors))

                    st.session_state["batch_zip"] = _zip_buf.getvalue()
                    st.session_state["batch_zip_name"] = f"wind_batch_{_START_YEAR}_{_END_YEAR}.zip"
                    st.session_state["batch_n"] = len(_summary_rows)
                    st.session_state["batch_summary"] = _summary_rows
                    st.session_state["batch_site_data"] = _batch_site_data

                if "batch_zip" in st.session_state:
                    st.download_button(
                        label=f"⬇️  Download ZIP ({st.session_state['batch_n']} sites + _summary.csv)",
                        data=st.session_state["batch_zip"],
                        file_name=st.session_state["batch_zip_name"],
                        mime="application/zip",
                        use_container_width=True,
                    )

                if "batch_summary" in st.session_state and st.session_state["batch_summary"]:
                    # ── PDF Report ─────────────────────────────────────────────
                    if st.button("📄  Generate PDF QA Report", use_container_width=True,
                                 help="Creates a multi-page PDF with satellite map, charts, tables and methodology"):
                        with st.spinner("Generating PDF report…"):
                            try:
                                _pdf_bytes = generate_pdf_report(
                                    summary_rows=st.session_state["batch_summary"],
                                    site_data=st.session_state.get("batch_site_data", {}),
                                    wake_df=load_wake_matrix(),
                                    pc_df=load_power_curves(),
                                    start_year=_START_YEAR,
                                    end_year=_END_YEAR,
                                )
                                st.session_state["batch_pdf"] = _pdf_bytes
                            except Exception as _pdf_err:
                                st.error(f"PDF generation failed: {_pdf_err}")
                                raise

                    if "batch_pdf" in st.session_state:
                        st.download_button(
                            label="⬇️  Download PDF Report",
                            data=st.session_state["batch_pdf"],
                            file_name=f"wind_assessment_report_{_START_YEAR}_{_END_YEAR}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                    _sr = pd.DataFrame(st.session_state["batch_summary"])

                    st.markdown("---")
                    st.markdown("**Results Summary**")
                    st.dataframe(_sr, use_container_width=True, hide_index=True)

                    # ── Map ────────────────────────────────────────────────────
                    _mlat = _sr["latitude"].mean()
                    _mlon = _sr["longitude"].mean()
                    _bm = folium.Map(location=[_mlat, _mlon], zoom_start=7,
                                     tiles=ESRI_TILES, attr=ESRI_ATTR)

                    _ws_col = "gwa_corrected_mean_hub_ms"
                    _ws_vals = _sr[_ws_col].dropna()
                    _ws_min = _ws_vals.min() if len(_ws_vals) else 0
                    _ws_max = _ws_vals.max() if len(_ws_vals) else 1

                    for _, _sr_row in _sr.iterrows():
                        _ws = _sr_row.get(_ws_col)
                        if pd.isna(_ws) if _ws is not None else True:
                            _colour = "#94A3B8"
                        else:
                            _t = (_ws - _ws_min) / max(_ws_max - _ws_min, 0.01)
                            r = int(220 - _t * 170)
                            g = int(80 + _t * 150)
                            _colour = f"#{r:02x}{g:02x}50"

                        _pop_lines = [
                            f"<b>{_sr_row['site_name']}</b>",
                            f"{_sr_row['latitude']:.4f}N, {_sr_row['longitude']:.4f}E",
                            f"Hub height: {_sr_row.get('hub_height_m', '—')} m",
                            f"GWA mean @ hub: {_sr_row.get('gwa_mean_hub_ms', '—'):.2f} m/s" if pd.notna(_sr_row.get('gwa_mean_hub_ms')) else "",
                            f"GWA-corrected: <b>{_ws:.2f} m/s</b>" if _ws is not None and pd.notna(_ws) else "",
                            f"Shear α: {_sr_row.get('wind_shear_alpha', '—'):.3f}" if pd.notna(_sr_row.get('wind_shear_alpha')) else "",
                        ]
                        if pd.notna(_sr_row.get("gross_aep_mwh_yr")):
                            _pop_lines += [
                                "—",
                                f"Turbine: {_sr_row.get('turbine_type', '—')}",
                                f"Nameplate: {_sr_row.get('nameplate_mw', '—'):.1f} MW",
                                f"Gross AEP: {_sr_row['gross_aep_mwh_yr']/1000:.1f} GWh/yr",
                                f"Net AEP: {_sr_row['net_aep_mwh_yr']/1000:.1f} GWh/yr",
                                f"CF: {_sr_row['capacity_factor_pct']:.1f} %",
                            ]
                        _popup_html = "<br>".join(l for l in _pop_lines if l)

                        folium.CircleMarker(
                            location=[_sr_row["latitude"], _sr_row["longitude"]],
                            radius=10,
                            color="white",
                            weight=1.5,
                            fill=True,
                            fill_color=_colour,
                            fill_opacity=0.9,
                            tooltip=f"{_sr_row['site_name']} — {_ws:.2f} m/s" if _ws is not None and pd.notna(_ws) else _sr_row['site_name'],
                            popup=folium.Popup(_popup_html, max_width=240),
                        ).add_to(_bm)

                    st_folium(_bm, height=420, use_container_width=True,
                              returned_objects=[], key="batch_map")

        except Exception as _be_outer:
            st.error(f"Could not read file: {_be_outer}")

# ── Site location map (full width) ───────────────────────────────────────────
if app_mode == "Single Site":
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


# ── Results section ───────────────────────────────────────────────────────────
if run_btn:
    try:
        with st.spinner(f"Fetching {_END_YEAR - _START_YEAR + 1} years of ERA5 data…"):
            df_era5_utc, era5_lat, era5_lon, site_elevation = fetch_era5(lat, lon, _START_YEAR, _END_YEAR)

        with st.spinner("Fetching Global Wind Atlas data…"):
            _prevailing = _prevailing_wd(df_era5_utc)
            _fetch_radius = int(min(max(5000, 100 * hub_height), 15000))
            site_roughness, rough_source = fetch_site_roughness(lat, lon, _prevailing, _fetch_radius)
            gwc, heights, gwa_lat, gwa_lon = fetch_gwa(lat, lon, site_roughness)

        # Store grid node coordinates for the map (persists across re-renders)
        st.session_state["era5_node"] = (era5_lat, era5_lon)
        if gwa_lat is not None and gwa_lon is not None:
            st.session_state["gwa_node"] = (gwa_lat, gwa_lon)

        # Convert timestamps to display timezone before pipeline so that
        # diurnal hour grouping uses local hours (physically correct)
        df_era5 = localise_df(df_era5_utc, tz_display)

        with st.spinner("Running processing pipeline…"):
            df, meta = run_pipeline(
                df_era5, gwc, heights, hub_height=hub_height, elevation=site_elevation,
                amplitude_scale=amplitude_scale, alpha_clip_lo=alpha_clip_lo,
                alpha_clip_hi=alpha_clip_hi, alpha_mean_lo=alpha_mean_lo, alpha_mean_hi=alpha_mean_hi,
            )

        if meta.get("alpha_raw", meta["alpha_mean"]) != meta["alpha_mean"]:
            st.warning(
                f"Wind shear exponent α = {meta['alpha_raw']:.3f} (from GWA) is outside the "
                f"plausible range [0.05, 0.60] and has been clamped to {meta['alpha_mean']:.3f}. "
                f"Check GWA statistics for this location."
            )
        if meta.get("density_clip_frac", 0) > 0.01:
            st.warning(
                f"Air density at hub height was outside 0.9–1.4 kg/m³ for "
                f"{meta['density_clip_frac']*100:.0f}% of hours and has been clipped. "
                f"At very high-altitude sites (>3000 m ASL) the ISA model may underestimate density."
            )

        # Persist for PDF report (survives re-renders)
        st.session_state["site_df"]   = df
        st.session_state["site_meta"] = meta

        tz_label = "UTC" if tz_display == "UTC" else f"{tz_detected} (UTC{_tz_offset_str(df.index)})"

        # ── Sub-hourly disaggregation ─────────────────────────────────────────
        df_sub = None
        sub_info = None
        if res_label != "Hourly":
            res_min = int(res_label.split("-")[0])
            with st.spinner(f"Disaggregating to {res_label}…"):
                df_sub, sub_info = disaggregate_subhourly(df, res_min)

        # Persist wind results so the AEP section survives re-renders without re-fetching.
        # Use sub-hourly df if disaggregation was requested, so the AEP CSV matches the wind CSV resolution.
        if df_sub is not None:
            _aep_save = df_sub[["ws_150m_subhourly"]].rename(columns={"ws_150m_subhourly": "ws_150m_corrected"})
            if "air_density" in df.columns:
                _aep_save["air_density"] = df["air_density"].reindex(_aep_save.index, method="ffill")
            st.session_state["aep_df"] = _aep_save
        else:
            st.session_state["aep_df"] = df
        st.session_state["aep_lat"] = lat
        st.session_state["aep_lon"] = lon
        st.session_state["aep_hub_height"] = hub_height

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
            f"ERA5 height-extrap. @ {hub_height:.0f} m",
            f"{meta['mean_era5_150_raw']:.2f} m/s",
            delta=f"{meta['mean_era5_150_raw'] - meta['mean_era5_100']:+.2f} m/s",
        )
        c3.metric(
            f"GWA-corrected @ {hub_height:.0f} m",
            f"{meta['mean_corrected']:.2f} m/s",
            delta=f"{meta['mean_corrected'] - meta['mean_era5_150_raw']:+.2f} m/s vs extrap.",
        )

        c4, c5, c6 = st.columns(3)
        c4.metric("GWA mean @ 100 m", f"{meta['mean_gwa_100']:.2f} m/s")
        c5.metric(f"GWA mean @ {hub_height:.0f} m", f"{meta['mean_gwa_150']:.2f} m/s")
        c6.metric(
            f"Wind shear α (100→{hub_height:.0f} m)",
            f"{meta['alpha_mean']:.3f}",
            delta=f"α 50→{hub_height:.0f} m = {meta['alpha_50_150']:.3f}" if meta["alpha_50_150"] else None,
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
        ERA5 extrap. {hub_height:.0f} m applies the diurnal power-law shear. GWA-corrected {hub_height:.0f} m is the
        final synthesised output — Weibull-transformed to match GWA's site statistics.{_rough_note}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("Weibull Parameters"):
            wb_tbl = pd.DataFrame(
                {
                    "Parameter": ["A — scale (m/s)", "k — shape"],
                    f"ERA5 {hub_height:.0f} m (extrapolated)": [
                        f"{meta['A_era5_150']:.3f}",
                        f"{meta['k_era5_150']:.3f}",
                    ],
                    f"GWA target {hub_height:.0f} m": [
                        f"{meta['A_gwa_150']:.3f}",
                        f"{meta['k_gwa_150']:.3f}",
                    ],
                }
            ).set_index("Parameter")
            st.dataframe(wb_tbl, use_container_width=True)
            st.markdown(f"""
            <div class="ann" style="margin-top:10px;">
            The quantile transform maps each ERA5 {hub_height:.0f} m value to the equivalent quantile
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
            sb2.metric(f"Mean TI @ {hub_height:.0f} m", f"{si['mean_TI_150']*100:.1f} %")
            sb3.metric(f"Mean σ_u @ {hub_height:.0f} m", f"{si['mean_sigma']:.2f} m/s")
            sb4.metric("AR(1) φ per step", f"{si['phi_sub']:.3f}")

            with st.expander("Disaggregation method"):
                st.markdown(
                    f"""
**Turbulence intensity source:** {si['ti_method']}

The per-hour σᵤ at {hub_height:.0f} m is estimated from the ERA5 gust factor at 10 m:

$$TI_{{10m}} = \\frac{{V_{{gust}}/V_{{10m}} - 1}}{{3.5}}$$

$$TI_{{{hub_height:.0f}m}} = TI_{{10m}} \\times \\left(\\frac{{10}}{{{hub_height:.0f}}}\\right)^{{0.11}}$$

The standard deviation for {res_label} *mean* output is reduced from instantaneous TI using the ratio of the integral time scale (≈ 350 s at {hub_height:.0f} m) to the averaging period ({si['resolution_min'] * 60} s):

$$\\sigma_{{\\text{{{res_label}}}}} = TI_{{{hub_height:.0f}m}} \\times V_{{{hub_height:.0f}m}} \\times \\sqrt{{T_{{int}} / T_{{avg}}}} \\approx {si['spectral_factor']:.2f} \\times \\sigma_{{\\text{{instantaneous}}}}$$

A **continuous AR(1) process** is then generated at {res_label} timesteps across the full record. The AR(1) coefficient (φ = {si['phi_sub']:.3f} per {si['resolution_min']}-min step) is derived from the hourly autocorrelation assuming a continuous-time Ornstein-Uhlenbeck process. Each hourly block's noise is mean-corrected to exactly preserve the ERA5 hourly means.
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
                yaxis_title=f"Wind Speed @ {hub_height:.0f} m (m/s)",
                height=320,
                margin=dict(t=40, b=50, l=55, r=20),
                legend=dict(orientation="h", y=-0.3),
                font=dict(color="#64748B", size=11),
                xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

        # ── Diurnal shear profile ─────────────────────────────────────────────
        st.markdown(f'<span class="lbl">Shear</span><p class="sh">Wind Shear Exponent α — 100 m → {hub_height:.0f} m</p>', unsafe_allow_html=True)

        da = meta["diurnal_alpha"]
        alpha_min, alpha_max = float(da.min()), float(da.max())

        sh_cols = st.columns(4) if meta["alpha_50_150"] else st.columns(3)
        sh_cols[0].metric(f"α (GWA 100→{hub_height:.0f} m)", f"{meta['alpha_mean']:.3f}")
        if meta["alpha_50_150"]:
            sh_cols[1].metric(f"α (GWA 50→{hub_height:.0f} m)", f"{meta['alpha_50_150']:.3f}")
            sh_cols[2].metric(f"Min α  (hour {int(da.idxmin()):02d}:00)", f"{alpha_min:.3f}")
            sh_cols[3].metric(f"Max α  (hour {int(da.idxmax()):02d}:00)", f"{alpha_max:.3f}")
        else:
            sh_cols[1].metric(f"Min α  (hour {int(da.idxmin()):02d}:00)", f"{alpha_min:.3f}")
            sh_cols[2].metric(f"Max α  (hour {int(da.idxmax()):02d}:00)", f"{alpha_max:.3f}")

        with st.expander("How is shear calculated?"):
            _gwa50_eq = ""
            if meta["alpha_50_150"] and meta["mean_gwa_50"]:
                _log_denom_50 = f"ln({hub_height:.0f}/{50})"
                _gwa50_eq = f"""
**GWA 50→{hub_height:.0f} m (supplementary):**

$$\\alpha_{{50\\text{{-}}{hub_height:.0f}}} = \\frac{{\\ln({meta['mean_gwa_150']:.2f}/{meta['mean_gwa_50']:.2f})}}{{{_log_denom_50}}} = {meta['alpha_50_150']:.3f}$$

This spans a wider height range and tends to be closer to the standard wind industry
value of ~0.2. It is shown for reference only — the 100→{hub_height:.0f} m α is used for extrapolation
because it most accurately represents the shear in the layer we are extrapolating across.
"""
            _diurnal_signal = "low" if (alpha_max - alpha_min) < 0.05 else "moderate" if (alpha_max - alpha_min) < 0.15 else "strong"
            _log_ratio = f"ln({hub_height:.0f}/100)"
            _alpha_method = meta.get("alpha_method", "")
            _using_blh = "boundary layer height" in _alpha_method

            if _using_blh:
                _step2_text = f"""
**Step 2 — Per-timestep stability from ERA5 boundary layer height (BLH)**

ERA5 BLH is the depth of the turbulent mixed layer. When BLH is below hub height the
rotor sits above the stable nocturnal boundary layer — the classic low-level jet regime
where shear is strongest. When BLH greatly exceeds hub height the atmosphere is well
mixed and shear is low.

A stability index is derived per timestep:

$$SI(t) = \\frac{{h_{{hub}}}}{{BLH(t)}}$$

clipped to [0, 5]. This is normalised to zero mean, unit standard deviation, then scaled
to produce per-timestep α:

$$\\alpha(t) = \\alpha_{{\\text{{mean}}}} + s \\cdot \\sigma_{{\\alpha}} \\cdot SI_{{\\text{{norm}}}}(t)$$

where σ_α is the standard deviation of ERA5 10m/100m instantaneous shear and *s* is
the amplitude scale (currently **{amplitude_scale:.2f}**). The chart shows the 24-hour
binned mean of the per-timestep α for interpretability.
"""
            else:
                _step2_text = f"""
**Step 2 — Diurnal pattern from ERA5 10m/100m shear (hour-of-day fallback)**

*(BLH data unavailable — using climatological hour-of-day grouping)*

The hourly shape of α is inferred from ERA5 10m/100m ratios — stable nights produce
stronger shear (higher α); convective days reduce it. The deviation from the mean is
scaled by the amplitude factor (*s* = **{amplitude_scale:.2f}**) and anchored to α_mean.
"""

            st.markdown(
                f"""
The shear exponent α varies **per timestep**, not a single constant.

**Step 1 — Mean magnitude from GWA (100→{hub_height:.0f} m)**

$$\\alpha_{{\\text{{mean}}}} = \\frac{{\\ln({meta['mean_gwa_150']:.2f}/{meta['mean_gwa_100']:.2f})}}{{{_log_ratio}}} = {meta['alpha_mean']:.3f}$$
{_gwa50_eq}
{_step2_text}
**Result** — every record extrapolated with its own α(t):

$$V_{{{hub_height:.0f}}}(t) = V_{{100}}(t) \\times \\left(\\frac{{{hub_height:.0f}}}{{100}}\\right)^{{\\alpha(t)}}$$

Displayed diurnal range (hourly mean): **{alpha_min:.3f} – {alpha_max:.3f}** ({_diurnal_signal} signal).
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
            f" The 50→{hub_height:.0f} m span (α = {meta['alpha_50_150']:.3f}) is typically closer to the "
            f"industry-standard ≈ 0.2 — the 100→{hub_height:.0f} m value is lower because shear decreases "
            f"with height in a well-mixed boundary layer."
            if meta["alpha_50_150"] else ""
        )
        st.markdown(f"""
        <div class="ann">
        <strong>Mean α ({meta['alpha_mean']:.3f})</strong> is anchored to GWA's 100→{hub_height:.0f} m speed ratio,
        which sets the long-term shear used for height extrapolation. Per-timestep variation comes
        from {meta.get('alpha_method', 'ERA5 stability signal')}, capturing real stability episodes.{_a50_note}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Stability source: {meta.get('alpha_method', '—')} · Amplitude scale: {amplitude_scale:.2f}")

        # ── Monthly mean time series ──────────────────────────────────────────
        st.markdown('<span class="lbl">Time Series</span><p class="sh">Monthly Mean Wind Speed</p>', unsafe_allow_html=True)

        monthly = df.resample("ME").mean()[
            ["ws_100m", "ws_150m_raw", "ws_150m_corrected"]
        ]
        monthly.columns = [
            "ERA5 raw 100 m",
            f"ERA5 extrap. {hub_height:.0f} m",
            f"GWA-corrected {hub_height:.0f} m",
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
        seasonal.columns = ["ERA5 raw 100 m", f"GWA-corrected {hub_height:.0f} m"]

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
        st.markdown(f'<span class="lbl">Distribution</span><p class="sh">Wind Speed Distribution @ {hub_height:.0f} m</p>', unsafe_allow_html=True)
        st.caption(
            "Bars show empirical frequency; dashed line is the GWA Weibull target. "
            "The green bars should closely follow the dashed line."
        )

        bins = np.arange(0, 31, 0.5)
        bc = (bins[:-1] + bins[1:]) / 2
        bin_w = 0.5

        _ws_raw_vals = df["ws_150m_raw"].dropna()
        _ws_corr_vals = df["ws_150m_corrected"].dropna()
        if len(_ws_raw_vals) < 10 or len(_ws_corr_vals) < 10:
            st.info("Insufficient wind data to plot distribution.")
        else:
            h_raw, _ = np.histogram(_ws_raw_vals, bins=bins, density=True)
            h_corr, _ = np.histogram(_ws_corr_vals, bins=bins, density=True)
            pdf_gwa = weibull_min.pdf(bc, c=meta["k_gwa_150"], scale=meta["A_gwa_150"])

            fig_wb = go.Figure()
            fig_wb.add_trace(
                go.Bar(
                    x=bc,
                    y=h_raw * bin_w,
                    name=f"ERA5 extrap. {hub_height:.0f} m",
                    marker_color="rgba(148,163,184,0.45)",
                    width=bin_w,
                )
            )
            fig_wb.add_trace(
                go.Bar(
                    x=bc,
                    y=h_corr * bin_w,
                    name=f"GWA-corrected {hub_height:.0f} m",
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
        daily.columns = ["ERA5 raw 100 m", f"GWA-corrected {hub_height:.0f} m"]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=daily.index, y=daily["ERA5 raw 100 m"],
            mode="lines", name="ERA5 raw 100 m",
            line=dict(color="#CBD5E1", width=0.9),
        ))
        fig_ts.add_trace(go.Scatter(
            x=daily.index, y=daily[f"GWA-corrected {hub_height:.0f} m"],
            mode="lines", name=f"GWA-corrected {hub_height:.0f} m",
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
            # GWA-corrected hub_height m — indigo steps
            fig_day.add_trace(go.Scatter(
                x=day_h.index, y=day_h["ws_150m_corrected"],
                mode="lines", name=f"GWA-corrected {hub_height:.0f} m",
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
        _site_name_csv = st.session_state.get("site_name_input", "").strip()
        _header_lines = [
            "# ERA5 x GWA Wind Resource Synthesis - synthesised time series",
            "#",
            *([ f"# Site name:          {_site_name_csv}" ] if _site_name_csv else []),
            f"# Latitude (input):   {lat:.4f}",
            f"# Longitude (input):  {lon:.4f}",
            (f"# ERA5 grid node:     {_era5_node[0]:.4f}N, {_era5_node[1]:.4f}E  (~0.25 deg grid, ~28 km)"
             if _era5_node else "# ERA5 grid node:     unknown (fetch data to populate)"),
            (f"# GWA grid node:      {_gwa_node[0]:.4f}N, {_gwa_node[1]:.4f}E  (250 m grid)"
             if _gwa_node else "# GWA grid node:      unknown (fetch data to populate)"),
            f"# Timezone:           {tz_display} ({_tz_suffix})",
            f"# Period:             {_START_YEAR}-01-01 to {_END_YEAR}-12-31",
            f"# Site elevation:     {meta['site_elevation']:.0f} m ASL",
            f"# Hub height:         {hub_height:.0f} m AGL  ({meta['site_elevation'] + hub_height:.0f} m ASL)",
            f"# Mean air density:   {meta['mean_air_density']:.4f} kg/m3 at hub height (ERA5 T2m + ISA lapse, standard: 1.225)",
            "#",
            "# DATA SOURCE:        SYNTHESISED - NOT A MEASUREMENT RECORD",
            "# Wind speeds are derived by combining ERA5 reanalysis (temporal variability)",
            "# with Global Wind Atlas statistics (spatial accuracy via Weibull correction).",
            "# Sub-hourly values (if present) are stochastic disaggregations, not observations.",
            "# Results are indicative only. Do not use as a substitute for on-site measurement.",
            "#",
        ]
        _file_header = "\n".join(_header_lines) + "\n"

        _hh_int = int(hub_height)
        _col_extrap = f"era5_ws_{_hh_int}m_extrap_ms"
        _col_corr = f"gwa_corrected_ws_{_hh_int}m_ms"

        if df_sub is not None:
            # Sub-hourly speed + hourly direction repeated for each sub-hourly step
            dl = _prep_index(df_sub[["ws_150m_subhourly"]].copy())
            dl.columns = [_col_corr]
            dl[_col_corr] = dl[_col_corr].round(1)
            if _has_wd:
                wd_sub = df["wd_100m"].reindex(df_sub.index, method="ffill")
                dl["era5_wd_100m_deg"] = wd_sub.round(0).astype(int).values
            csv_bytes = (_file_header + dl.to_csv()).encode()
            dl_label = f"Download {res_label} time series (CSV)"
            dl_caption = (
                f"ws: GWA-corrected {_hh_int} m (m/s, 1 dp, {res_label} stochastic fake) · "
                f"wd: ERA5 100 m (°, hourly repeated) · Timestamps: {_tz_suffix}"
            )
            dl_fname = f"wind_{_hh_int}m_{res_label}_{lat:.4f}_{lon:.4f}_{_START_YEAR}_{_END_YEAR}.csv"
        else:
            # Hourly: speed columns + wind direction + air density
            cols = ["ws_100m", "ws_150m_raw", "ws_150m_corrected"]
            col_names = ["era5_ws_100m_ms", _col_extrap, _col_corr]
            if _has_wd:
                cols.append("wd_100m")
                col_names.append("era5_wd_100m_deg")
            if "air_density" in df.columns:
                cols.append("air_density")
                col_names.append("air_density_kg_m3")
            dl = _prep_index(df[cols])
            dl.columns = col_names
            dl[["era5_ws_100m_ms", _col_extrap, _col_corr]] = (
                dl[["era5_ws_100m_ms", _col_extrap, _col_corr]].round(1)
            )
            if _has_wd:
                dl["era5_wd_100m_deg"] = dl["era5_wd_100m_deg"].round(0).astype(int)
            if "air_density_kg_m3" in dl.columns:
                dl["air_density_kg_m3"] = dl["air_density_kg_m3"].round(4)
            csv_bytes = (_file_header + dl.to_csv()).encode()
            dl_label = "Download hourly time series (CSV)"
            dl_caption = (
                f"Speeds in m/s (1 dp) · Direction in ° (0–360) · Timestamps: {_tz_suffix}"
            )
            dl_fname = f"wind_{_hh_int}m_hourly_{lat:.4f}_{lon:.4f}_{_START_YEAR}_{_END_YEAR}.csv"

        # Persist so the download button survives re-renders after clicking it
        st.session_state["wind_csv"] = {
            "bytes": csv_bytes, "fname": dl_fname,
            "label": dl_label, "caption": dl_caption,
        }

    except requests.HTTPError as exc:
        st.error(f"API request failed: {exc}")
    except ValueError as exc:
        st.error(f"Data processing error: {exc}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        raise

# ── Persistent single-site downloads + PDF ────────────────────────────────────
if "wind_csv" in st.session_state and app_mode == "Single Site":
    _wcsv = st.session_state["wind_csv"]
    st.download_button(
        label=_wcsv["label"],
        data=_wcsv["bytes"],
        file_name=_wcsv["fname"],
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(_wcsv["caption"])

    st.markdown("---")

    if st.button("📄  Generate PDF QA Report", use_container_width=True,
                 key="ss_pdf_btn",
                 help="Single-site PDF with wind charts and (if AEP was run) energy results"):
        _ss_df   = st.session_state.get("site_df")
        _ss_meta = st.session_state.get("site_meta")
        _ss_aep  = st.session_state.get("site_aep_results")
        _ss_lat  = st.session_state.get("aep_lat", lat)
        _ss_lon  = st.session_state.get("aep_lon", lon)
        _ss_hub  = int(st.session_state.get("aep_hub_height", hub_height))
        _ss_name_input = st.session_state.get("site_name_input", "").strip()
        _ss_name = _ss_name_input or f"Site ({_ss_lat:.4f}, {_ss_lon:.4f})"

        if _ss_df is not None and _ss_meta is not None:
            _ss_row = {
                "site_name":                 _ss_name,
                "latitude":                  _ss_lat,
                "longitude":                 _ss_lon,
                "elevation_m_asl":           _ss_meta.get("site_elevation", 0),
                "hub_height_m":              _ss_hub,
                "era5_mean_100m_ms":         round(_ss_meta.get("mean_era5_100", 0), 2),
                "gwa_mean_100m_ms":          round(_ss_meta.get("mean_gwa_100", 0), 2),
                "gwa_mean_hub_ms":           round(_ss_meta.get("mean_gwa_150", 0), 2),
                "gwa_corrected_mean_hub_ms": round(_ss_meta.get("mean_corrected", 0), 2),
                "wind_shear_alpha":          round(_ss_meta.get("alpha_mean", 0), 3),
                "mean_air_density_kg_m3":    round(_ss_meta.get("mean_air_density", 1.225), 4),
            }
            _ss_entry = {
                "ws_corr":    _ss_df["ws_150m_corrected"],
                "ws_raw":     _ss_df["ws_150m_raw"] if "ws_150m_raw" in _ss_df.columns else _ss_df["ws_150m_corrected"],
                "meta":       _ss_meta,
                "air_density": _ss_df["air_density"] if "air_density" in _ss_df.columns else None,
            }
            if _ss_aep is not None:
                _ar = _ss_aep["aep"]
                _ss_net_adj = _ss_aep.get("net_aep_adj", _ar["net_aep_mwh"])
                _ss_cf_adj  = _ss_aep.get("cf_adj", _ar["capacity_factor"])
                _ss_row.update({
                    "gross_aep_mwh_yr":     _ar["gross_aep_mwh"],
                    "net_aep_mwh_yr":       _ss_net_adj,
                    "turbine_type":         _ss_aep["wtg"],
                    "nameplate_mw":         _ss_aep["nameplate_mw"],
                    "mean_wake_loss_pct":   _ar["mean_wake_pct"],
                    "capacity_factor_pct":  _ss_cf_adj * 100,
                })
                _ss_entry.update({
                    "aep": {
                        "gross_mw":        _ar["gross_mw_ts"],
                        "net_mw":          _ar["net_mw_ts"],
                        "gross_aep_mwh":   _ar["gross_aep_mwh"],
                        "net_aep_mwh":     _ss_net_adj,
                        "mean_wake_pct":   _ar["mean_wake_pct"],
                        "capacity_factor": _ss_cf_adj,
                        "rated_kw":        _ar["rated_kw"],
                    },
                    "wtg":         _ss_aep["wtg"],
                    "nameplate_mw": _ss_aep["nameplate_mw"],
                })
            with st.spinner("Generating PDF report…"):
                try:
                    _ss_pdf_bytes = generate_pdf_report(
                        summary_rows=[_ss_row],
                        site_data={_ss_name: _ss_entry},
                        wake_df=load_wake_matrix(),
                        pc_df=load_power_curves(),
                        start_year=_START_YEAR,
                        end_year=_END_YEAR,
                    )
                    st.session_state["single_site_pdf"] = _ss_pdf_bytes
                except Exception as _e:
                    st.error(f"PDF generation failed: {_e}")
                    raise
        else:
            st.warning("No wind data in memory — fetch data first.")

    if "single_site_pdf" in st.session_state:
        st.download_button(
            label="⬇️  Download PDF Report",
            data=st.session_state["single_site_pdf"],
            file_name=f"wind_report_{_START_YEAR}_{_END_YEAR}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

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

        st.markdown('<p style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#94A3B8;margin:1rem 0 0.3rem 0;">Energy Losses</p>', unsafe_allow_html=True)
        _lc1, _lc2, _lc3, _lc4 = st.columns(4)
        with _lc1:
            _avail_loss = st.number_input("Availability loss [%]", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")
        with _lc2:
            _elec_loss = st.number_input("Electrical loss [%]", min_value=0.0, max_value=50.0, value=2.0, step=0.1, format="%.1f")
        with _lc3:
            _tp_loss = st.number_input("Turbine performance loss [%]", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")
        with _lc4:
            _deg_loss = st.number_input("Degradation [%]", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")

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
        _density_aep = _df_aep["air_density"] if "air_density" in _df_aep.columns else None
        _aep = calc_aep(
            _ws_aep, _pc_df, _selected_wtg, _nameplate_mw,
            _wake_df if _apply_wake else None,
            density_series=_density_aep,
        )

        # Apply additional losses (multiplicative)
        _other_loss_factor = (
            (1 - _avail_loss / 100)
            * (1 - _elec_loss / 100)
            * (1 - _tp_loss / 100)
            * (1 - _deg_loss / 100)
        )
        _other_loss_pct = (1 - _other_loss_factor) * 100
        _net_aep_adj = _aep["net_aep_mwh"] * _other_loss_factor
        _cf_adj = _net_aep_adj / (_nameplate_mw * 8760.0)
        _total_loss_pct = (1 - (_aep["net_aep_mwh"] / _aep["gross_aep_mwh"]) * _other_loss_factor) * 100 if _aep["gross_aep_mwh"] > 0 else 0.0

        st.session_state["site_aep_results"] = {
            "aep": _aep,
            "wtg": _selected_wtg,
            "nameplate_mw": _nameplate_mw,
            "net_aep_adj": _net_aep_adj,
            "cf_adj": _cf_adj,
            "avail_loss": _avail_loss,
            "elec_loss": _elec_loss,
            "tp_loss": _tp_loss,
            "deg_loss": _deg_loss,
            "other_loss_pct": _other_loss_pct,
        }

        # Metrics
        def _energy_str(mwh: float) -> str:
            return f"{mwh / 1000:.2f} GWh/yr" if mwh >= 1000 else f"{mwh:.0f} MWh/yr"

        _am1, _am2, _am3, _am4, _am5 = st.columns(5)
        _am1.metric("Gross AEP", _energy_str(_aep["gross_aep_mwh"]))
        _am2.metric(
            "Net AEP (post-losses)",
            _energy_str(_net_aep_adj),
            delta=f"−{_total_loss_pct:.1f}% total losses" if _total_loss_pct > 0 else None,
            delta_color="inverse",
        )
        _am3.metric("Capacity Factor", f"{_cf_adj * 100:.1f}%")
        _am4.metric("Rated (from curve)", f"{_aep['rated_kw'] / 1000:.2f} MW")
        _am5.metric(
            "Air Density",
            f"{_aep['mean_air_density']:.4f} kg/m³",
            delta=f"{(_aep['mean_air_density'] - 1.225) / 1.225 * 100:+.1f}% vs 1.225",
            delta_color="off",
        )

        _loss_breakdown_parts = []
        if _apply_wake and _wake_df is not None and _aep["mean_wake_pct"] > 0:
            _loss_breakdown_parts.append(f"wake {_aep['mean_wake_pct']:.1f}%")
        if _avail_loss > 0:
            _loss_breakdown_parts.append(f"availability {_avail_loss:.1f}%")
        if _elec_loss > 0:
            _loss_breakdown_parts.append(f"electrical {_elec_loss:.1f}%")
        if _tp_loss > 0:
            _loss_breakdown_parts.append(f"turbine performance {_tp_loss:.1f}%")
        if _deg_loss > 0:
            _loss_breakdown_parts.append(f"degradation {_deg_loss:.1f}%")
        _loss_note = (
            f" Losses applied: {', '.join(_loss_breakdown_parts)} (combined −{_total_loss_pct:.1f}%)."
            if _loss_breakdown_parts else " No losses applied."
        )

        st.markdown(
            f'<div class="ann">'
            f'<strong>{_selected_wtg}</strong> normalised to rated output and scaled to '
            f'<strong>{_nameplate_mw:.1f} MW</strong> nameplate capacity.'
            f'{_loss_note}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Monthly net energy
        st.markdown(
            '<p class="sh" style="margin-top:1.5rem;">Mean Monthly Net Energy</p>',
            unsafe_allow_html=True,
        )
        _interval_h = (_aep["gross_mw_ts"].index[1] - _aep["gross_mw_ts"].index[0]).total_seconds() / 3600
        _monthly_net = (_aep["net_mw_ts"] * _other_loss_factor).resample("ME").sum() * _interval_h
        _monthly_gross = _aep["gross_mw_ts"].resample("ME").sum() * _interval_h
        _mnet_avg = _monthly_net.groupby(_monthly_net.index.month).mean()
        _mgross_avg = _monthly_gross.groupby(_monthly_gross.index.month).mean()

        _fig_maep = go.Figure()
        _fig_maep.add_trace(go.Bar(
            x=MONTH_LABELS, y=_mgross_avg.values,
            name="Gross energy", marker_color="rgba(148,163,184,0.5)",
        ))
        _fig_maep.add_trace(go.Bar(
            x=MONTH_LABELS, y=_mnet_avg.values,
            name="Net energy (after all losses)", marker_color="rgba(79,70,229,0.7)",
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
        _aep_n_yr = (_df_aep.index[-1] - _df_aep.index[0]).days / 365.25
        _aep_site_name = st.session_state.get("site_name_input", "").strip()
        _aep_header_lines = [
            "# ERA5 x GWA Wind Resource Synthesis - AEP time series",
            "#",
            *([ f"# Site name:          {_aep_site_name}" ] if _aep_site_name else []),
            f"# Latitude (input):   {_aep_lat:.4f}",
            f"# Longitude (input):  {_aep_lon:.4f}",
            f"# Timezone:           {tz_display} ({_aep_tz_sfx})",
            f"# Wind record:        {_df_aep.index[0].strftime('%Y-%m-%d')} to {_df_aep.index[-1].strftime('%Y-%m-%d')} ({_aep_n_yr:.1f} yr)",
            "#",
            f"# Wind turbine:       {_selected_wtg}",
            f"# Rated capacity:     {_aep['rated_kw']/1000:.2f} MW (from power curve)",
            f"# Nameplate capacity: {_nameplate_mw:.1f} MW (scaled)",
            f"# Wake losses:        {'applied (2D interpolation from wake matrix)' if _apply_wake and _wake_df is not None else 'not applied'}",
            f"# Air density:        {_aep['mean_air_density']:.4f} kg/m3 mean at hub height (IEC V_eq = V*(rho/1.225)^(1/3) applied)",
            "#",
            f"# Gross AEP:          {_energy_str(_aep['gross_aep_mwh'])}",
            f"# Net AEP (wake only): {_energy_str(_aep['net_aep_mwh'])}",
            f"# Mean wake loss:     {_aep['mean_wake_pct']:.1f} %",
            f"# Availability loss:  {_avail_loss:.1f} %",
            f"# Electrical loss:    {_elec_loss:.1f} %",
            f"# Turbine perf. loss: {_tp_loss:.1f} %",
            f"# Degradation:        {_deg_loss:.1f} %",
            f"# Total other losses: {_other_loss_pct:.2f} %",
            f"# Net AEP (all losses): {_energy_str(_net_aep_adj)}",
            f"# Capacity factor:    {_cf_adj*100:.1f} %",
            "#",
            "# INDICATIVE ONLY - wind speeds are synthesised from ERA5 + GWA, not measured.",
            "#",
        ]
        _aep_file_header = "\n".join(_aep_header_lines) + "\n"
        _aep_out = pd.DataFrame(
            {
                "wind_speed_ms": _ws_aep.round(1),
                "equiv_wind_speed_ms": _aep["ws_equiv_ts"].round(1),
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
            data=(_aep_file_header + _aep_out.to_csv()).encode(),
            file_name=_aep_fname,
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            f"{_selected_wtg} · {_nameplate_mw:.1f} MW nameplate · "
            f"{'Wake-corrected' if _apply_wake and _wake_df is not None else 'No wake correction'}"
            + (f" · other losses {_other_loss_pct:.1f}%" if _other_loss_pct > 0 else "")
            + f" · {_aep_n_yr:.1f}-year record"
        )
