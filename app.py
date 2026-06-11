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

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
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
_END_YEAR = datetime.now().year - 1
_START_YEAR = _END_YEAR - 9
GWA_URL = "https://globalwindatlas.info/api/gwa/custom/Lib/"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
    "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
)


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


def parse_gwc(text):
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

    gwc = {}
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

            # Keep the entry with the lowest roughness (free-stream resource)
            if h not in gwc or r < gwc[h]["roughness"]:
                gwc[h] = {"A": A_c, "k": k_c, "mean": mean_ws, "roughness": r}

    if not gwc:
        raise ValueError("Could not extract any height data from GWC file.")

    return gwc, sorted(gwc.keys()), gwa_lat, gwa_lon


# ── Data fetching (cached) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def fetch_era5(lat: float, lon: float):
    """
    Fetch 10-year hourly ERA5 wind at 10m and 100m from Open-Meteo.
    Returns (df, era5_lat, era5_lon) — the lat/lon are the actual ERA5 grid node
    coordinates used, which may differ from the input by up to ~0.125°.
    """
    r = requests.get(
        OPENMETEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{_START_YEAR}-01-01",
            "end_date": f"{_END_YEAR}-12-31",
            "hourly": "wind_speed_100m,wind_speed_10m,wind_gusts_10m",
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
        },
        index=pd.to_datetime(d["hourly"]["time"]),
    )
    return df.dropna(), era5_lat, era5_lon


@st.cache_data(show_spinner=False)
def fetch_gwa(lat: float, lon: float):
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
    return parse_gwc(r.text)  # returns (gwc, heights, gwa_lat, gwa_lon)


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
    alpha_mean = np.log(mean_gwa_150 / mean_gwa_100) / np.log(150 / 100)

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
        "diurnal_alpha": diurnal_alpha,
        "mean_era5_100": df["ws_100m"].mean(),
        "mean_era5_150_raw": df["ws_150m_raw"].mean(),
        "mean_corrected": df["ws_150m_corrected"].mean(),
        "mean_gwa_100": mean_gwa_100,
        "mean_gwa_150": mean_gwa_150,
        "A_era5_150": A_era5_150,
        "k_era5_150": k_era5_150,
        "A_gwa_150": A_gwa_150,
        "k_gwa_150": k_gwa_150,
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

st.title("ERA5 + GWA Wind Resource Tool — 150 m")
st.caption(
    "Combines 10 years of ERA5 hourly wind variability with Global Wind Atlas "
    "downscaled spatial accuracy, extrapolated to 150 m hub height."
)

# ── Session state — persist grid node markers across re-renders ───────────────
for _key in ("era5_node", "gwa_node", "_prev_lat", "_prev_lon"):
    if _key not in st.session_state:
        st.session_state[_key] = None

# Sidebar
with st.sidebar:
    st.header("Location")
    lat = st.number_input(
        "Latitude", value=-31.9505, min_value=-90.0, max_value=90.0,
        step=0.0001, format="%.4f",
    )
    lon = st.number_input(
        "Longitude", value=115.8605, min_value=-180.0, max_value=180.0,
        step=0.0001, format="%.4f",
    )
    # Clear cached node markers when the user moves to a new location
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
        f"Detected timezone: **{tz_detected}**"
        + ("  *(overridden to UTC)*" if use_utc else "")
    )
    st.info(
        f"**ERA5 period**\n{_START_YEAR} – {_END_YEAR}\n\n"
        f"10 years · hourly · {tz_display}"
    )
    st.divider()
    st.header("Output Resolution")
    resolution = st.selectbox(
        "Temporal resolution",
        options=["Hourly", "30-min", "10-min"],
        index=0,
        help=(
            "30-min and 10-min outputs are stochastically disaggregated from "
            "ERA5 hourly data using per-hour turbulence intensity (from ERA5 "
            "gust factor) and an AR(1) process calibrated to the site's "
            "mesoscale autocorrelation. Each run produces a plausible "
            "realisation — not a historical record."
        ),
    )
    st.divider()
    run_btn = st.button("Fetch & Process Data", type="primary", use_container_width=True)

# Map + method description
col_map, col_method = st.columns([3, 2])

with col_map:
    st.subheader("Site Location")
    m = folium.Map(location=[lat, lon], zoom_start=9, tiles=ESRI_TILES, attr=ESRI_ATTR)

    # Input site (red)
    folium.Marker(
        [lat, lon],
        tooltip=f"Input site: {lat:.4f}°, {lon:.4f}°",
        popup=f"<b>Input site</b><br>{lat:.4f}°N, {lon:.4f}°E",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

    # ERA5 grid node (blue) — shown after first fetch
    if st.session_state["era5_node"]:
        elat, elon = st.session_state["era5_node"]
        folium.Marker(
            [elat, elon],
            tooltip=f"ERA5 node (~0.25° grid): {elat:.4f}°, {elon:.4f}°",
            popup=f"<b>ERA5 grid node</b><br>{elat:.4f}°N, {elon:.4f}°E<br>Resolution ~28 km",
            icon=folium.Icon(color="blue", icon="cloud", prefix="fa"),
        ).add_to(m)
        if (elat, elon) != (lat, lon):
            folium.PolyLine(
                [[lat, lon], [elat, elon]],
                color="#4a90d9", weight=1.5, dash_array="6",
                tooltip="Site → ERA5 node",
            ).add_to(m)

    # GWA grid node (orange) — shown after first fetch
    if st.session_state["gwa_node"]:
        glat, glon = st.session_state["gwa_node"]
        folium.Marker(
            [glat, glon],
            tooltip=f"GWA node (250 m grid): {glat:.4f}°, {glon:.4f}°",
            popup=f"<b>GWA grid node</b><br>{glat:.4f}°N, {glon:.4f}°E<br>Resolution 250 m",
            icon=folium.Icon(color="orange", icon="sun-o", prefix="fa"),
        ).add_to(m)

    st_folium(m, height=420, use_container_width=True, returned_objects=[])

    if st.session_state["era5_node"] or st.session_state["gwa_node"]:
        st.caption("🔴 Input site  🔵 ERA5 grid node (~0.25°, ~28 km)  🟠 GWA grid node (250 m)")

with col_method:
    st.subheader("Processing Method")
    st.markdown(
        """
        **① ERA5 @ 100 m**
        10-year hourly wind speed from Open-Meteo ERA5 archive.

        **② Height extrapolation 100 m → 150 m**
        Power-law with a *diurnal shear exponent* — the 24-hour pattern of α
        is inferred from the ERA5 10 m / 100 m ratio (captures the daily
        stability cycle), then normalised so the long-term mean matches GWA's
        100 m / 150 m shear:

        $$V_{150}(t) = V_{100}(t) \\times \\left(\\frac{150}{100}\\right)^{\\alpha(h)}$$

        **③ Weibull distribution correction**
        Quantile transform to match GWA 150 m Weibull *A* & *k*:

        $$V^* = A_{\\text{GWA}} \\times \\left(\\frac{V}{A_{\\text{ERA5}}}\\right)^{k_{\\text{ERA5}} / k_{\\text{GWA}}}$$

        **Result:** ERA5 temporal structure + GWA spatial accuracy @ 150 m.
        """
    )

# ── Results section ───────────────────────────────────────────────────────────
if run_btn:
    try:
        with st.spinner(f"Fetching {_END_YEAR - _START_YEAR + 1} years of ERA5 data…"):
            df_era5_utc, era5_lat, era5_lon = fetch_era5(lat, lon)

        with st.spinner("Fetching Global Wind Atlas data…"):
            gwc, heights, gwa_lat, gwa_lon = fetch_gwa(lat, lon)

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
        if resolution != "Hourly":
            res_min = int(resolution.split("-")[0])
            with st.spinner(f"Disaggregating to {resolution}…"):
                df_sub, sub_info = disaggregate_subhourly(df, res_min)

        n_records = len(df_sub) if df_sub is not None else len(df)
        st.success(
            f"Done — {n_records:,} {resolution.lower()} records  "
            f"({_START_YEAR}–{_END_YEAR} · {tz_label})"
        )

        # ── Summary metrics ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Summary Statistics")

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
        c6.metric("Wind shear α (GWA mean)", f"{meta['alpha_mean']:.3f}")

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

        # ── Sub-hourly disaggregation panel ──────────────────────────────────
        if df_sub is not None and sub_info is not None:
            st.divider()
            st.subheader(f"Sub-hourly Disaggregation — {resolution}")

            si = sub_info
            sb1, sb2, sb3, sb4 = st.columns(4)
            sb1.metric("Output resolution", resolution)
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

The standard deviation for {resolution} *mean* output is then reduced from
instantaneous TI using the ratio of the integral time scale (≈ 350 s at 150 m)
to the averaging period ({si['resolution_min'] * 60} s):

$$\\sigma_{{\\text{{{resolution}}}}} = TI_{{150m}} \\times V_{{150m}} \\times
  \\sqrt{{T_{{int}} / T_{{avg}}}} = \\times {si['spectral_factor']:.2f}\\; \\text{{of instantaneous}}$$

A **continuous AR(1) process** is then generated at {resolution} timesteps across
the full 10-year record. The AR(1) coefficient (φ = {si['phi_sub']:.3f} per {si['resolution_min']}-min step)
is derived from the hourly autocorrelation assuming a continuous-time
Ornstein-Uhlenbeck process. Finally, each hourly block's noise is
mean-corrected to exactly preserve the ERA5 hourly means.

⚠️ **This is a single stochastic realisation.** The temporal sequence within each
hour is plausible but not the actual historical record.
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
                    name=f"{resolution} (disaggregated)",
                    line=dict(color="#27ae60", width=1.0),
                )
            )
            fig_sub.add_trace(
                go.Scatter(
                    x=sample_h.index,
                    y=sample_h.values,
                    mode="lines+markers",
                    name="Hourly ERA5+GWA",
                    line=dict(color="#c0392b", width=2.0, dash="dash"),
                    marker=dict(size=5),
                )
            )
            fig_sub.update_layout(
                title=f"Sample {n_days}-day window showing {resolution} disaggregation",
                xaxis_title=f"Date/Time ({tz_display})",
                yaxis_title="Wind Speed @ 150 m (m/s)",
                height=320,
                margin=dict(t=40, b=50, l=60, r=20),
                legend=dict(orientation="h", y=-0.3),
            )
            st.plotly_chart(fig_sub, use_container_width=True)

        # ── Diurnal shear profile ─────────────────────────────────────────────
        st.divider()
        st.subheader("Wind Shear Exponent α  (100 m → 150 m)")

        da = meta["diurnal_alpha"]
        alpha_min, alpha_max = float(da.min()), float(da.max())

        sh_c1, sh_c2, sh_c3 = st.columns(3)
        sh_c1.metric("Mean α (from GWA)", f"{meta['alpha_mean']:.3f}")
        sh_c2.metric(f"Min α  (hour {int(da.idxmin()):02d}:00)", f"{alpha_min:.3f}")
        sh_c3.metric(f"Max α  (hour {int(da.idxmax()):02d}:00)", f"{alpha_max:.3f}")

        with st.expander("How is shear calculated?"):
            st.markdown(
                f"""
The shear exponent α is **diurnal** — it varies hour-by-hour across the 24-hour cycle,
not a single constant value.

**Step 1 — Mean magnitude from GWA**
The long-term mean α is derived from the GWA mean wind speeds at 100 m and 150 m:

$$\\alpha_{{\\text{{mean}}}} = \\frac{{\\ln(V_{{150}}/V_{{100}})}}{{\\ln(150/100)}}
= \\frac{{\\ln({meta['mean_gwa_150']:.2f}/{meta['mean_gwa_100']:.2f})}}{{\\ln(1.5)}}
= {meta['alpha_mean']:.3f}$$

**Step 2 — Diurnal pattern from ERA5**
The hourly shape of α (which hours are high, which are low) is inferred from the ERA5
10 m / 100 m wind ratio.  The stable night-time boundary layer produces stronger shear
(higher α); daytime convective mixing reduces it. This pattern is normalised so its mean
equals α_mean from Step 1.

**Result**
Every hourly ERA5 record is extrapolated to 150 m using its own hour-specific α:

$$V_{{150}}(t) = V_{{100}}(t) \\times \\left(\\frac{{150}}{{100}}\\right)^{{\\alpha(h)}}
\\quad \\text{{where }} h = \\text{{hour of day in {tz_display}}}$$

The diurnal range here is **{alpha_min:.3f} – {alpha_max:.3f}**, a
{"low" if (alpha_max - alpha_min) < 0.05 else "moderate" if (alpha_max - alpha_min) < 0.15 else "strong"}
diurnal signal.
                """
            )

        fig_alpha = go.Figure()
        fig_alpha.add_trace(
            go.Scatter(
                x=da.index.tolist(),
                y=da.values.tolist(),
                mode="lines+markers",
                line=dict(color="#1a73e8", width=2.5),
                marker=dict(size=7),
                name="Diurnal α",
                fill="tozeroy",
                fillcolor="rgba(26,115,232,0.1)",
            )
        )
        fig_alpha.add_hline(
            y=meta["alpha_mean"],
            line_dash="dash",
            line_color="#888",
            annotation_text=f"Mean α = {meta['alpha_mean']:.3f}",
            annotation_position="top right",
        )
        fig_alpha.update_layout(
            xaxis=dict(title=f"Hour of Day ({tz_display})", tickmode="linear", tick0=0, dtick=3),
            yaxis_title="Shear Exponent α",
            height=280,
            margin=dict(t=15, b=40, l=60, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_alpha, use_container_width=True)

        # ── Monthly mean time series ──────────────────────────────────────────
        st.divider()
        st.subheader("Monthly Mean Wind Speed")

        monthly = df.resample("ME").mean()[
            ["ws_100m", "ws_150m_raw", "ws_150m_corrected"]
        ]
        monthly.columns = [
            "ERA5 raw 100 m",
            "ERA5 extrap. 150 m",
            "GWA-corrected 150 m",
        ]

        fig_monthly = go.Figure()
        palette = ["#90bfd8", "#f39c12", "#27ae60"]
        for col, colour in zip(monthly.columns, palette):
            fig_monthly.add_trace(
                go.Scatter(
                    x=monthly.index,
                    y=monthly[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colour, width=1.8),
                )
            )
        fig_monthly.update_layout(
            xaxis_title="Date",
            yaxis_title="Mean Wind Speed (m/s)",
            height=330,
            margin=dict(t=15, b=60, l=60, r=20),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

        # ── Mean monthly seasonality ──────────────────────────────────────────
        st.subheader("Mean Monthly Seasonality (all years averaged)")
        month_avg = df.copy()
        month_avg["month"] = month_avg.index.month
        seasonal = (
            month_avg.groupby("month")[
                ["ws_100m", "ws_150m_corrected"]
            ]
            .mean()
        )
        seasonal.columns = ["ERA5 raw 100 m", "GWA-corrected 150 m"]

        MONTH_LABELS = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        fig_seasonal = go.Figure()
        for col, colour in zip(seasonal.columns, ["#90bfd8", "#27ae60"]):
            fig_seasonal.add_trace(
                go.Bar(
                    x=MONTH_LABELS,
                    y=seasonal[col].values,
                    name=col,
                    marker_color=colour,
                    opacity=0.85,
                )
            )
        fig_seasonal.update_layout(
            barmode="group",
            xaxis_title="Month",
            yaxis_title="Mean Wind Speed (m/s)",
            height=300,
            margin=dict(t=15, b=60, l=60, r=20),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)

        # ── Weibull distribution ──────────────────────────────────────────────
        st.divider()
        st.subheader("Wind Speed Distribution @ 150 m")
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
                marker_color="rgba(144,191,216,0.6)",
                width=bin_w,
            )
        )
        fig_wb.add_trace(
            go.Bar(
                x=bc,
                y=h_corr * bin_w,
                name="GWA-corrected 150 m",
                marker_color="rgba(39,174,96,0.65)",
                width=bin_w,
            )
        )
        fig_wb.add_trace(
            go.Scatter(
                x=bc,
                y=pdf_gwa * bin_w,
                mode="lines",
                name="GWA Weibull target",
                line=dict(color="#c0392b", width=2.5, dash="dash"),
            )
        )
        fig_wb.update_layout(
            barmode="overlay",
            xaxis_title="Wind Speed (m/s)",
            yaxis_title="Probability",
            height=330,
            margin=dict(t=15, b=60, l=60, r=20),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_wb, use_container_width=True)

        # ── Daily time series ─────────────────────────────────────────────────
        st.divider()
        st.subheader("Daily Mean Wind Speed — Full Record")

        daily = df.resample("D").mean()[["ws_100m", "ws_150m_corrected"]]
        daily.columns = ["ERA5 raw 100 m", "GWA-corrected 150 m"]

        fig_ts = go.Figure()
        for col, colour in zip(daily.columns, ["#90bfd8", "#27ae60"]):
            fig_ts.add_trace(
                go.Scatter(
                    x=daily.index,
                    y=daily[col],
                    mode="lines",
                    name=col,
                    line=dict(color=colour, width=0.8),
                )
            )
        fig_ts.update_layout(
            xaxis_title="Date",
            yaxis_title="Wind Speed (m/s)",
            height=300,
            margin=dict(t=15, b=60, l=60, r=20),
            legend=dict(orientation="h", y=-0.3),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Download")

        if df_sub is not None:
            # Sub-hourly download
            dl = df_sub.copy()
            dl.index.name = f"datetime_{tz_display.replace('/', '_')}"
            dl.columns = ["gwa_corrected_150m_ms"]
            csv_bytes = dl.to_csv().encode()
            dl_label = f"Download {resolution} time series (CSV)"
            dl_caption = (
                f"Column: GWA-corrected 150 m  |  m/s  |  "
                f"{resolution} stochastic disaggregation  |  Timestamps: **{tz_display}**"
            )
            dl_fname = f"wind_150m_{resolution}_{lat:.4f}_{lon:.4f}_{_START_YEAR}_{_END_YEAR}.csv"
        else:
            # Hourly download (all columns)
            dl = df[["ws_100m", "ws_150m_raw", "ws_150m_corrected"]].copy()
            dl.index.name = f"datetime_{tz_display.replace('/', '_')}"
            dl.columns = [
                "era5_100m_ms",
                "era5_150m_height_extrap_ms",
                "gwa_corrected_150m_ms",
            ]
            csv_bytes = dl.to_csv().encode()
            dl_label = "Download hourly time series (CSV)"
            dl_caption = (
                f"Columns: ERA5 100 m raw · ERA5 150 m (height extrap.) · "
                f"GWA-corrected 150 m  |  All values in m/s  |  Timestamps: **{tz_display}**"
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
