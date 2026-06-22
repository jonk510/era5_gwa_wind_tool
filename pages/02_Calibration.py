"""
02_Calibration.py — Measurement-based calibration page (ERA5 × GWA Wind Tool)

Fits amplitude_scale and mean_multiplier against concurrent site measurements,
then applies corrections to the full long-term synthetic wind speed series.

Calibration uses ONLY the concurrent overlap period between model and measurements.
Corrections are assumed stationary (constant site bias) and applied to the full
long-term record.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize_scalar

st.set_page_config(
    page_title="Calibration — ERA5 × GWA",
    page_icon="🎯",
    layout="wide",
)

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

matplotlib.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
})


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers (no Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

def _skip_comments(raw: bytes) -> int:
    """Count leading '#' comment lines in raw CSV bytes."""
    n = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if line.startswith("#"):
            n += 1
        else:
            break
    return n


def _load_series(f, ws_col: str | None = None) -> tuple[pd.Series | None, str]:
    """
    Load a wind speed Series from a Streamlit UploadedFile.
    Handles '#' comment headers and auto-detects wind speed column.
    Returns (series_with_naive_index, column_name) or (None, "").
    Timezone info is stripped — caller is responsible for alignment.
    """
    try:
        raw  = f.read()
        skip = _skip_comments(raw)
        df   = pd.read_csv(io.BytesIO(raw), skiprows=skip, index_col=0, parse_dates=True)
    except Exception as e:
        st.error(f"Cannot read CSV: {e}")
        return None, ""

    if ws_col and ws_col in df.columns:
        col = ws_col
    else:
        ordered = (
            [c for c in df.columns if "gwa_corrected" in c.lower()]
            + [c for c in df.columns if c.lower().startswith("ws_")]
            + [c for c in df.columns if pd.api.types.is_float_dtype(df[c])]
        )
        if not ordered:
            st.error(f"No wind speed column found. Columns: {list(df.columns)}")
            return None, ""
        col = ordered[0]

    s = df[col].dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        st.error("Cannot parse a datetime index. Check the file format.")
        return None, col
    if s.index.tz is not None:
        s = s.tz_localize(None)
    return s.rename("wind_speed"), col


def _detect_model_tz(series: pd.Series) -> str:
    """
    Return a human-readable timezone label for the model series.
    Session state data is timezone-aware local time; CSV uploads are naive local time.
    """
    if series.index.tz is not None:
        return str(series.index.tz)
    # Try reading timezone from the index name (e.g. "datetime_UTC+08:00")
    name = str(series.index.name or "")
    if "UTC" in name:
        return name.split("datetime_")[-1] if "datetime_" in name else name
    return "local time (timezone unknown)"


def _find_overlap(
    model: pd.Series, meas: pd.Series, meas_shift_hours: float = 0.0
) -> tuple[pd.Series, pd.Series]:
    """
    Resample both to hourly means, optionally shift measured timestamps by
    meas_shift_hours, then align on the common timestamp index.

    meas_shift_hours: hours to ADD to measured timestamps so they align with
    the model's local time.  e.g. if model is UTC+8 local time and measured
    is UTC, set meas_shift_hours = +8.
    """
    m_h  = model.copy()
    if m_h.index.tz is not None:          # strip tz for naive comparison
        m_h = m_h.tz_localize(None)
    m_h  = m_h.resample("h").mean().dropna()

    ms_h = meas.resample("h").mean().dropna()
    if meas_shift_hours != 0.0:
        ms_h.index = ms_h.index + pd.Timedelta(hours=meas_shift_hours)

    idx = m_h.index.intersection(ms_h.index)
    return m_h.loc[idx], ms_h.loc[idx]


def _representativeness(model_ov: pd.Series, model_full: pd.Series) -> dict:
    """Coverage and representativeness metrics for the concurrent overlap period."""
    moy     = sorted(set(model_ov.index.month))
    missing = sorted(set(range(1, 13)) - set(moy))
    n_cal   = len(set(zip(model_ov.index.year, model_ov.index.month)))

    seas_map = {1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
                6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON",12:"DJF"}
    seasons  = sorted(set(seas_map[m] for m in moy))

    full_mean = float(model_full.mean())
    ov_mean   = float(model_ov.mean())
    rep_ratio = ov_mean / full_mean if full_mean > 0 else 1.0

    if len(model_ov) < 24 * 30:
        quality = "very_poor"
    elif n_cal < 6:
        quality = "poor"
    elif missing:
        quality = "moderate"
    else:
        quality = "good"

    return dict(
        n_hours=len(model_ov),
        n_cal_months=n_cal,
        months_of_year=moy,
        missing=missing,
        seasons=seasons,
        quality=quality,
        rep_ratio=rep_ratio,
        ov_mean=ov_mean,
        full_mean=full_mean,
        start=model_ov.index[0],
        end=model_ov.index[-1],
    )


def _diurnal_hourly_mean(ws: pd.Series) -> pd.Series:
    """For each timestep, return the long-term mean wind speed at that clock hour."""
    return ws.groupby(ws.index.hour).transform("mean")


def _apply_amp(ws: pd.Series, s: float) -> pd.Series:
    """Scale per-timestep deviation from the diurnal hourly mean by s.
    Preserves the long-term mean (mean of deviations ≈ 0)."""
    dm = _diurnal_hourly_mean(ws)
    return (dm + s * (ws - dm)).clip(lower=0.0)


def _rmse_diurnal(s: float, model: pd.Series, meas: pd.Series) -> float:
    """RMSE between 24-hour mean wind speed profiles at a given amplitude scale s."""
    scaled = _apply_amp(model, s)
    pm = scaled.groupby(scaled.index.hour).mean().reindex(range(24))
    pr = meas.groupby(meas.index.hour).mean().reindex(range(24))
    return float(np.sqrt(np.nanmean((pm.values - pr.values) ** 2)))


def _calibrate(model_ov: pd.Series, meas_ov: pd.Series) -> dict:
    """
    Two-step calibration on the concurrent overlap period:
      1. Optimise amplitude_scale (s) to minimise diurnal RMSE.
      2. Compute mean_multiplier (k) = measured_mean / corrected_model_mean.
    Both steps use the overlap period only.
    """
    res = minimize_scalar(
        _rmse_diurnal,
        bounds=(0.3, 4.0),
        method="bounded",
        args=(model_ov, meas_ov),
        options={"xatol": 0.005},
    )
    s      = float(res.x)
    rmse_b = _rmse_diurnal(1.0, model_ov, meas_ov)
    rmse_a = _rmse_diurnal(s,   model_ov, meas_ov)
    scaled = _apply_amp(model_ov, s)
    k      = float(meas_ov.mean() / scaled.mean()) if float(scaled.mean()) > 0 else 1.0
    return dict(
        amplitude_scale=s,
        mean_multiplier=k,
        rmse_before=rmse_b,
        rmse_after=rmse_a,
        mean_meas=float(meas_ov.mean()),
        mean_model_raw=float(model_ov.mean()),
        mean_model_corrected=float(scaled.mean() * k),
    )


def _apply_corrections(ws: pd.Series, s: float = 1.0, k: float = 1.0) -> pd.Series:
    """Apply amplitude scale then mean multiplier to a full wind speed series."""
    return (_apply_amp(ws, s) * k).clip(lower=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chart_diurnal(model_ov, meas_ov, s, k) -> plt.Figure:
    corrected_ov = _apply_corrections(model_ov, s, k)
    hrs = np.arange(24)

    def _prof(ser):
        return ser.groupby(ser.index.hour).mean().reindex(range(24)).values

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(hrs, _prof(meas_ov),      "k-",  lw=2.2, label="Measured",               zorder=5)
    ax.plot(hrs, _prof(model_ov),     "--",  lw=1.5, color="#94A3B8", label="Model (uncorrected)")
    ax.plot(hrs, _prof(corrected_ov),  "-",  lw=2.0, color="#4F46E5",
            label=f"Model corrected  (s={s:.2f}, k={k:.3f})")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.set_title("Diurnal profile — concurrent overlap period", fontweight="bold")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_monthly(model_full, corrected_full, meas_ov) -> plt.Figure:
    def _monthly(ser):
        mn = ser.resample("ME").mean()
        return mn.groupby(mn.index.month).mean().reindex(range(1, 13)).values

    meas_mn = meas_ov.resample("ME").mean()
    meas_by_m = meas_mn.groupby(meas_mn.index.month).mean().reindex(range(1, 13)).values

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    x = np.arange(1, 13)
    ax.plot(x, _monthly(model_full),     "--", color="#94A3B8", lw=1.5,
            label="Model (uncorrected, long-term)")
    ax.plot(x, _monthly(corrected_full),  "-", color="#4F46E5", lw=2.0,
            label="Model (corrected, long-term)")
    ax.scatter(x, meas_by_m, color="#16A34A", s=55, zorder=5, marker="D",
               label="Measured (overlap months only)")
    ax.set_xticks(x)
    ax.set_xticklabels(MONTH_NAMES, rotation=35, ha="right")
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.set_title("Monthly mean wind speed — full long-term record", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_rmse_curve(model_ov, meas_ov, s_opt) -> plt.Figure:
    scales = np.linspace(0.3, 3.5, 100)
    rmses  = [_rmse_diurnal(s, model_ov, meas_ov) for s in scales]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(scales, rmses, color="#0F172A", lw=1.5)
    ax.axvline(s_opt, color="#4F46E5", lw=1.5, ls="--", label=f"Optimal s = {s_opt:.2f}")
    ax.axvline(1.0,   color="#94A3B8", lw=1.0, ls=":",  label="Default s = 1.00")
    ax.set_xlabel("Amplitude scale (s)")
    ax.set_ylabel("Diurnal RMSE (m/s)")
    ax.set_title("Objective function — RMSE vs amplitude scale", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _chart_coverage(rep: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    vals   = [1 if m in rep["months_of_year"] else 0 for m in range(1, 13)]
    colors = ["#4F46E5" if v else "#E2E8F0" for v in vals]
    ax.bar(range(1, 13), vals, color=colors, width=0.7, edgecolor="white", lw=0.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES, rotation=35, ha="right")
    ax.set_yticks([])
    ax.set_ylim(0, 1.6)
    ax.set_title("Calendar months covered in concurrent overlap", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_csv(
    final_series: pd.Series,
    site_label: str,
    rep: dict,
    result: dict,
    s_use: float,
    k_use: float,
    apply_amp: bool,
    apply_mean: bool,
    original_mean: float,
    hub_h,
) -> bytes:
    missing_str = (
        ", ".join(MONTH_NAMES[m - 1] for m in rep["missing"])
        if rep["missing"] else "none"
    )
    lines = [
        "# ERA5 × GWA Wind Tool — Measurement-calibrated wind speed output",
        f"# Site: {site_label}",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "#",
        "# --- Calibration period ---",
        f"# Overlap start:              {rep['start'].strftime('%Y-%m-%d')}",
        f"# Overlap end:                {rep['end'].strftime('%Y-%m-%d')}",
        f"# Overlap duration:           {rep['n_cal_months']} calendar months "
        f"({rep['n_hours']:,} hourly records)",
        f"# Calendar coverage:          {rep['quality'].replace('_', ' ')}",
        f"# Missing months of year:     {missing_str}",
        f"# Concurrent model mean:      {rep['ov_mean']:.3f} m/s",
        f"# Concurrent measured mean:   {result['mean_meas']:.3f} m/s",
        f"# Concurrent/long-term ratio: {rep['rep_ratio']:.3f}  "
        "(>1 = overlap was a high-wind period; <1 = low-wind period)",
        "#",
        "# --- Long-term output ---",
        f"# Original model mean (long-term):  {original_mean:.3f} m/s",
        f"# Corrected model mean (long-term): {final_series.mean():.3f} m/s",
        "#",
        "# --- Corrections applied ---",
        f"# Amplitude scale (s): {s_use:.4f}  {'[applied]' if apply_amp else '[NOT applied — s=1.0 used]'}",
        f"# Mean multiplier (k): {k_use:.6f}  {'[applied]' if apply_mean else '[NOT applied — k=1.0 used]'}",
        f"# Diurnal RMSE before: {result['rmse_before']:.4f} m/s  (overlap period)",
        f"# Diurnal RMSE after:  {result['rmse_after']:.4f} m/s  (overlap period)",
        "#",
        "# --- Method ---",
        "# Step 1 (amplitude): ws_corr(t) = diurnal_mean(h) + s * (ws(t) - diurnal_mean(h))",
        "#   diurnal_mean(h) = long-term mean wind speed at clock hour h.",
        "#   s optimised to minimise RMSE of 24-hr mean diurnal profiles (overlap period).",
        "#   This step does not change the long-term mean wind speed.",
        "# Step 2 (mean):      ws_final(t) = ws_corr(t) * k",
        "#   k = mean(measured_overlap) / mean(ws_corr_overlap).",
        "#   This step adjusts the long-term mean to match measurements.",
        "# Both corrections derived from concurrent overlap only.",
        "# Assumed stationary (constant site bias) and applied to the full long-term record.",
    ]
    if rep["quality"] in ("poor", "very_poor"):
        lines.append(
            "# WARNING: Short overlap period — correction factors carry high uncertainty."
        )
    if abs(rep["rep_ratio"] - 1.0) > 0.10:
        pct = (rep["rep_ratio"] - 1.0) * 100
        lines.append(
            f"# WARNING: Concurrent model mean was {pct:+.1f}% vs long-term mean. "
            "Mean multiplier (k) may not be fully representative of long-term conditions."
        )

    buf = io.StringIO()
    for ln in lines:
        buf.write(ln + "\n")
    col_name = f"calibrated_ws_{hub_h}m_ms"
    final_series.round(2).to_frame(name=col_name).to_csv(buf)
    return buf.getvalue().encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ── Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.warn { background:#FFFBEB; border-left:3px solid #F59E0B; padding:8px 14px;
        border-radius:0 6px 6px 0; font-size:0.82rem; color:#92400E; margin:6px 0; }
.good { background:#ECFDF5; border-left:3px solid #10B981; padding:8px 14px;
        border-radius:0 6px 6px 0; font-size:0.82rem; color:#065F46; margin:6px 0; }
.info { background:#EFF6FF; border-left:3px solid #3B82F6; padding:8px 14px;
        border-radius:0 6px 6px 0; font-size:0.82rem; color:#1E40AF; margin:6px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding-bottom:1rem; margin-bottom:0.5rem; border-bottom:1px solid #E2E8F0;">
  <p style="font-size:0.7rem; font-weight:700; letter-spacing:0.1em;
            text-transform:uppercase; color:#94A3B8; margin:0 0 4px 0;">ERA5 × GWA Wind Tool</p>
  <h1 style="font-size:1.6rem; font-weight:700; color:#0F172A; margin:0;">
    Measurement Calibration</h1>
  <p style="font-size:0.88rem; color:#64748B; margin:4px 0 0 0;">
    Tune amplitude scale and mean speed to concurrent site measurements,
    then apply corrections to the full long-term synthetic output.
  </p>
</div>
""", unsafe_allow_html=True)

with st.expander("How calibration works — methodology"):
    st.markdown("""
This page derives two correction factors from the **concurrent overlap period** between
the synthetic model output and site measurements, then applies them to the **full
long-term synthetic record**. Not all sites will have measurements, and short records
may not be seasonally representative — both situations are handled with warnings.

---

**Factor 1 — Amplitude scale (s):** corrects the day/night diurnal swing without
changing the long-term mean.

> `ws_corr(t) = diurnal_mean(h) + s × (ws(t) − diurnal_mean(h))`

where `diurnal_mean(h)` is the long-term mean wind speed at clock hour h across
the full synthetic record. `s` is found by minimising the RMSE between the model
and measured 24-hour mean diurnal profiles over the overlap period.
`s > 1` → larger day/night swing; `s < 1` → flatter profile.
This step **does not change the long-term mean**.

**Factor 2 — Mean multiplier (k):** shifts the overall level up or down to match
the measured mean.

> `ws_final(t) = ws_corr(t) × k`

where `k = mean(measured_overlap) / mean(ws_corr_overlap)`.
`k > 1` → model underestimates mean; `k < 1` → model overestimates.
This step **does change the long-term mean**.

---

**Why concurrent overlap only?**

Short measurement records (< 12 months) may not capture all seasons, and the
measurement period itself may have been an unusually high or low wind year. This
page flags both risks:

- *Coverage warning* — if the overlap is missing months of year, diurnal or seasonal
  patterns for those months cannot be validated.
- *Representativeness warning* — if the concurrent model mean differs from the
  long-term model mean by more than 10%, the mean multiplier (k) may carry a
  wind-year bias component. In that case, consider applying only the amplitude scale.

**Timezone:** the model output is in **local time** (the tool converts ERA5 UTC to the
site's local timezone before all processing, so diurnal hour grouping is physically
correct). Your measured data must be in the same timezone, or you must apply the
"Shift measured timestamps" offset below to align them before overlap matching.

**Cross-site transfer:** the amplitude scale (s) reflects ERA5's systematic
underestimation of stability-driven shear variability and is moderately transferable
to nearby sites with similar terrain. The mean multiplier (k) is site-specific
(roughness, terrain) and should **not** be transferred to other locations.
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Model data
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 1 — Model data")

has_single = (
    st.session_state.get("site_df") is not None
    and "ws_150m_corrected" in getattr(st.session_state.get("site_df"), "columns", [])
)
has_batch = bool(st.session_state.get("batch_site_data"))

if has_single or has_batch:
    src = st.radio(
        "Model data source",
        ["Use data from this session", "Upload model output CSV"],
        horizontal=True,
        key="calib_src",
    )
else:
    src = "Upload model output CSV"
    st.markdown(
        '<div class="info">ℹ️ No model output found in this session. '
        "Run the main tool first, or upload a model output CSV below.</div>",
        unsafe_allow_html=True,
    )

model_full: pd.Series | None = None
model_label = "Site"
hub_h: int | str = "?"

if src == "Use data from this session":
    if has_batch:
        site_names = list(st.session_state["batch_site_data"].keys())
        sel = st.selectbox("Select site", site_names, key="calib_batch_site")
        sd  = st.session_state["batch_site_data"][sel]
        model_full  = sd["ws_corr"].dropna()
        model_label = sel
        try:
            hub_h = int(sd.get("meta", {}).get("hub_height",
                        st.session_state.get("aep_hub_height", "?")))
        except (TypeError, ValueError):
            hub_h = "?"
    else:
        model_full  = st.session_state["site_df"]["ws_150m_corrected"].dropna()
        model_label = st.session_state.get("site_name_input", "Site")
        try:
            hub_h = int(
                st.session_state.get("site_meta", {}).get(
                    "hub_height", st.session_state.get("aep_hub_height", "?")
                )
            )
        except (TypeError, ValueError):
            hub_h = "?"

    if model_full is not None:
        st.caption(
            f"**{model_label}** · {len(model_full):,} hourly records · "
            f"{model_full.index[0].date()} → {model_full.index[-1].date()} · "
            f"mean {model_full.mean():.2f} m/s · hub {hub_h} m"
        )
else:
    model_upload = st.file_uploader(
        "Upload model output CSV",
        type=["csv"],
        help="Tool-generated CSV (# comment headers OK). "
             "Wind speed column auto-detected (prefers gwa_corrected_* columns).",
        key="calib_model_file",
    )
    if model_upload:
        ws_hint = st.text_input(
            "Wind speed column (leave blank to auto-detect)", value="", key="calib_model_col"
        )
        model_full, detected = _load_series(model_upload, ws_hint or None)
        if model_full is not None:
            model_label = model_upload.name
            st.caption(
                f"Column **{detected}** · {len(model_full):,} records · "
                f"{model_full.index[0].date()} → {model_full.index[-1].date()} · "
                f"mean {model_full.mean():.2f} m/s"
            )

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Measured data
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 2 — Measured wind speed")

meas_upload = st.file_uploader(
    "Upload measured wind speed CSV",
    type=["csv"],
    help="Any CSV with a datetime index and a wind speed column in m/s. "
         "'#' comment header lines are skipped automatically. "
         "Hub-height measurements preferred.",
    key="calib_meas_file",
)

meas_full: pd.Series | None = None

if meas_upload:
    meas_hint = st.text_input(
        "Measured wind speed column (leave blank to auto-detect)", value="", key="calib_meas_col"
    )
    meas_full, meas_col = _load_series(meas_upload, meas_hint or None)
    if meas_full is not None:
        st.caption(
            f"Column **{meas_col}** · {len(meas_full):,} records · "
            f"{meas_full.index[0].date()} → {meas_full.index[-1].date()} · "
            f"mean {meas_full.mean():.2f} m/s"
        )

# ─────────────────────────────────────────────────────────────────────────────
# Gate: need both datasets to proceed
# ─────────────────────────────────────────────────────────────────────────────
if model_full is None or meas_full is None:
    st.markdown(
        '<div class="info">ℹ️ Provide both model data and measured data above to continue.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Timezone alignment
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Timezone alignment")

model_tz_label = _detect_model_tz(model_full)
st.markdown(
    f'<div class="info">ℹ️ Model timestamps are in <b>{model_tz_label}</b>. '
    "The tool converts ERA5 (UTC) to local time before processing, so the model output "
    "CSV and session state data are both in <b>local time</b>. "
    "Your measured data must use the same timezone for the overlap to align correctly.</div>",
    unsafe_allow_html=True,
)

meas_shift = st.number_input(
    "Shift measured timestamps by (hours)",
    min_value=-14.0,
    max_value=14.0,
    value=0.0,
    step=0.5,
    key="calib_tz_shift",
    help=(
        "Add this many hours to the measured timestamps before matching with the model. "
        "Example: if the model is UTC+8 local time and your logger records in UTC, "
        "set this to +8. If both are already in the same timezone, leave at 0."
    ),
)
if meas_shift != 0.0:
    st.markdown(
        f'<div class="warn">⚠️ Measured timestamps will be shifted by '
        f'<b>{meas_shift:+.1f} h</b> before overlap matching.</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Find concurrent overlap
# ─────────────────────────────────────────────────────────────────────────────
model_ov, meas_ov = _find_overlap(model_full, meas_full, meas_shift_hours=meas_shift)

if len(model_ov) < 24 * 7:
    st.error(
        f"Only {len(model_ov)} overlapping hourly records found "
        f"({len(model_ov)/24:.1f} days). Need at least 7 days. "
        "Check that both files cover the same period and use the same timezone convention "
        "(both should be UTC or both local time — not mixed)."
    )
    st.stop()

rep = _representativeness(model_ov, model_full)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Overlap analysis
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 3 — Concurrent overlap analysis")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Overlap hours",          f"{rep['n_hours']:,}")
c2.metric("Calendar months",        str(rep["n_cal_months"]))
c3.metric("Measured mean (overlap)", f"{float(meas_ov.mean()):.2f} m/s")
c4.metric("Model mean (overlap)",   f"{rep['ov_mean']:.2f} m/s")

fig_cov = _chart_coverage(rep)
st.pyplot(fig_cov)
plt.close(fig_cov)

# Coverage quality banner
_missing_names = ", ".join(MONTH_NAMES[m - 1] for m in rep["missing"]) if rep["missing"] else ""
QUAL_MSGS = {
    "very_poor": ("warn", "⚠️ <b>Very short overlap (&lt; 30 days).</b> "
                          "Calibration results have very high uncertainty."),
    "poor":      ("warn", "⚠️ <b>Short overlap (&lt; 6 calendar months).</b> "
                          "Seasonal cycle may be poorly represented."),
    "moderate":  ("warn", f"⚠️ <b>Partial coverage.</b> Missing months of year: "
                          f"{_missing_names}. These seasons are unvalidated."),
    "good":      ("good", "✓ <b>Good coverage.</b> All 12 calendar months of year represented."),
}
cls, msg = QUAL_MSGS[rep["quality"]]
st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

# Representativeness check
rep_pct = abs(rep["rep_ratio"] - 1.0) * 100
if rep_pct > 10:
    direction = "above" if rep["rep_ratio"] > 1.0 else "below"
    st.markdown(
        f'<div class="warn">⚠️ <b>Concurrent period model mean ({rep["ov_mean"]:.2f} m/s) '
        f'is {rep_pct:.0f}% {direction} the long-term model mean ({rep["full_mean"]:.2f} m/s).</b> '
        f"The measurement period may not be wind-climatologically representative of long-term conditions. "
        f"The mean multiplier (k) may carry a wind-year bias. "
        f"Consider applying only the amplitude scale correction.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="good">✓ Concurrent model mean is within {rep_pct:.1f}% of the '
        f"long-term model mean — measurement period is reasonably representative.</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Calibration
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 4 — Calibration results")

# Stable cache key: changing data invalidates the result automatically
calib_key = (
    f"_calib_{model_label}_{len(model_ov)}"
    f"_{int(meas_ov.mean()*1000)}_{int(model_ov.mean()*1000)}"
)
if calib_key not in st.session_state:
    with st.spinner("Optimising amplitude scale…"):
        st.session_state[calib_key] = _calibrate(model_ov, meas_ov)

result = st.session_state[calib_key]
s_opt  = result["amplitude_scale"]
k_opt  = result["mean_multiplier"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Amplitude scale (s)", f"{s_opt:.3f}",
    help="s > 1 → larger day/night swing; s < 1 → flatter profile. Does not change long-term mean.",
)
m2.metric(
    "Mean multiplier (k)", f"{k_opt:.4f}",
    help="k > 1 → model underestimates mean; k < 1 → overestimates. Changes long-term mean.",
)
m3.metric("Diurnal RMSE before", f"{result['rmse_before']:.3f} m/s")
m4.metric(
    "Diurnal RMSE after", f"{result['rmse_after']:.3f} m/s",
    delta=f"{result['rmse_after'] - result['rmse_before']:+.3f} m/s",
    delta_color="inverse",
)
m5.metric(
    "Final corrected mean vs measured",
    f"{result['mean_model_corrected']:.2f} vs {result['mean_meas']:.2f} m/s",
)

corrected_full = _apply_corrections(model_full, s_opt, k_opt)

col_l, col_r = st.columns(2)
with col_l:
    fig_d = _chart_diurnal(model_ov, meas_ov, s_opt, k_opt)
    st.pyplot(fig_d)
    plt.close(fig_d)
with col_r:
    fig_m = _chart_monthly(model_full, corrected_full, meas_ov)
    st.pyplot(fig_m)
    plt.close(fig_m)

with st.expander("Show optimisation curve (RMSE vs amplitude scale)"):
    fig_r = _chart_rmse_curve(model_ov, meas_ov, s_opt)
    st.pyplot(fig_r)
    plt.close(fig_r)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Apply and download
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 5 — Apply corrections and download")

st.markdown(
    "Select which corrections to apply to the **full long-term synthetic series**. "
    "Both factors are derived from the concurrent overlap period only and assumed "
    "stationary (constant site-specific bias)."
)

ca, cb = st.columns(2)
with ca:
    apply_amp = st.checkbox(
        f"Apply amplitude scale  (s = {s_opt:.3f})",
        value=True,
        help="Adjusts day/night swing. Does not change long-term mean.",
        key="calib_apply_amp",
    )
with cb:
    # Default the mean multiplier to OFF if representativeness ratio is poor
    default_k = abs(rep["rep_ratio"] - 1.0) <= 0.10
    apply_mean = st.checkbox(
        f"Apply mean multiplier  (k = {k_opt:.4f})",
        value=default_k,
        help="Scales all wind speeds to match measured mean. Changes long-term mean. "
             "Disabled by default if concurrent period is not representative of long-term conditions.",
        key="calib_apply_mean",
    )

s_use = s_opt if apply_amp  else 1.0
k_use = k_opt if apply_mean else 1.0
final_series = _apply_corrections(model_full, s_use, k_use)

p1, p2, p3 = st.columns(3)
p1.metric("Original model mean",     f"{model_full.mean():.2f} m/s")
p2.metric(
    "Final corrected mean",
    f"{final_series.mean():.2f} m/s",
    delta=f"{final_series.mean() - model_full.mean():+.2f} m/s",
)
p3.metric("Measured mean (overlap)", f"{float(meas_ov.mean()):.2f} m/s")

csv_bytes = _build_csv(
    final_series, model_label, rep, result,
    s_use, k_use, apply_amp, apply_mean,
    float(model_full.mean()), hub_h,
)
fname = (
    f"calibrated_{model_label.replace(' ', '_').replace('/', '_')}"
    f"_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
)
st.download_button(
    "⬇ Download calibrated wind speed CSV",
    data=csv_bytes,
    file_name=fname,
    mime="text/csv",
)

# Persist calibrated parameters in session state for cross-page reference
st.session_state["calib_amplitude_scale"] = s_use
st.session_state["calib_mean_multiplier"] = k_use
st.session_state["calib_s_applied"]       = apply_amp
st.session_state["calib_k_applied"]       = apply_mean
st.session_state["calib_site_label"]      = model_label
st.session_state["calib_rep_quality"]     = rep["quality"]

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Cross-site guidance
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 6 — Cross-site guidance")

other_sites = []
if has_batch:
    other_sites = [
        nm for nm in st.session_state["batch_site_data"].keys() if nm != model_label
    ]

st.markdown(f"""
For nearby sites **without** concurrent measurements, the amplitude scale calibrated
here may be a useful starting point:

| Parameter | Calibrated value | Transferability |
|-----------|-----------------|----------------|
| **Amplitude scale (s)** | **{s_opt:.2f}** | Moderately transferable to nearby sites with similar terrain, fetch, and stability regime. Enter in *Advanced settings → Diurnal amplitude scale* in the main tool. |
| **Mean multiplier (k)** | **{k_opt:.4f}** | Site-specific (roughness + terrain bias). **Do not transfer** to other sites without separate validation. |

Transferability of the amplitude scale depends on:
- Terrain similarity (coastal vs inland, flat vs complex)
- Similar prevailing wind direction and upstream fetch
- Measurement record quality ({rep["quality"].replace("_", " ")} coverage here)
- How representative the overlap period was of long-term conditions
  (concurrent/long-term model mean ratio: {rep["rep_ratio"]:.2f})
""")

if other_sites:
    st.info(
        f"Your current session contains {len(other_sites)} other batch site(s) without measurements: "
        + (", ".join(other_sites[:5]) + (f" and {len(other_sites)-5} more" if len(other_sites) > 5 else ""))
        + f". Consider re-running those sites with amplitude_scale = {s_opt:.2f} in the main tool's "
          "Advanced settings to apply regional calibration."
    )
