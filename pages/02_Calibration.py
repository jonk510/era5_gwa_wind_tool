"""
02_Calibration.py — Measurement-based calibration page (ERA5 × GWA Wind Tool)

Fits amplitude_scale and mean_multiplier against concurrent site measurements,
then applies corrections to the full long-term synthetic wind speed series.

Calibration uses ONLY the concurrent overlap period between model and measurements.
Corrections are assumed stationary (constant site bias) and applied to the full
long-term record.
"""

import io
import re
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


def _is_windpro(raw: bytes) -> bool:
    """Return True if the bytes look like a WindPRO time-series export."""
    first = raw.decode("utf-8", errors="replace").split("\n", 1)[0]
    return "windpro" in first.lower()


def _load_windpro_series(raw: bytes) -> tuple[pd.Series | None, str]:
    """
    Parse a WindPRO time-series CSV export.

    Header layout (6 lines):
      1: "WindPRO time series export version 1,,,,,"
      2: "<height> - <label>,,,,,"
      3: "<hub height>,,,,,"
      4: ",Time stamp,MeanWindSpeedUID,..."
      5: "Disabled,Time stamp (UTC+08:00) Perth,Mean wind speed,..."
      6: ",d/mm/yyyy h:mm AMPM,m/s,..."
    Data: <disabled_flag>,<timestamp>,<ws_ms>,<dir_deg>,<TI>,<shear>,
    """
    try:
        lines = raw.decode("utf-8", errors="replace").splitlines()

        # Extract timezone from line 5 (0-indexed line 4)
        tz_label = ""
        if len(lines) >= 5:
            m = re.search(r"UTC[+\-]\d{2}:\d{2}", lines[4])
            if m:
                tz_label = m.group(0)   # e.g. "UTC+08:00"

        df = pd.read_csv(io.BytesIO(raw), skiprows=6, header=None)
        # Cols: 0=disabled flag, 1=timestamp, 2=wind speed, 3=direction, ...
        df[1] = pd.to_datetime(df[1], format="%d/%m/%Y %H:%M", dayfirst=True)
        df = df.set_index(1)
        df.index.name = "datetime"

        ws = pd.to_numeric(df[2], errors="coerce").dropna()
        ws.name = "wind_speed"

        col_label = f"Mean wind speed [{tz_label}]" if tz_label else "Mean wind speed"
        return ws, col_label
    except Exception as e:
        st.error(f"Failed to parse WindPRO export: {e}")
        return None, ""


def _load_windpro_meteo_series(raw: bytes) -> tuple[pd.Series | None, str]:
    """
    Parse a WindPRO Meteo Data Export (.txt, tab-separated).

    Header: free-form metadata lines until a row whose first field is "TimeStamp",
    followed by a units row, then data rows.
    Wind speed column detected by "MeanWindSpeed" in the column name.
    Rows where SampleStatus != 0 are dropped.
    """
    try:
        text  = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()

        # Extract timezone from metadata block
        tz_label = ""
        for line in lines:
            m = re.search(r"UTC[+\-]\d{2}:\d{2}", line)
            if m:
                tz_label = m.group(0)
                break

        # Find the header row (first field == "TimeStamp")
        header_idx = None
        for i, line in enumerate(lines):
            if line.split("\t")[0].strip() == "TimeStamp":
                header_idx = i
                break

        if header_idx is None:
            st.error("Cannot find data header row in WindPRO Meteo export.")
            return None, ""

        # Build TSV: header + data (skip the units row at header_idx + 1)
        data_text = "\n".join([lines[header_idx]] + lines[header_idx + 2:])
        df = pd.read_csv(io.StringIO(data_text), sep="\t", low_memory=False)

        # Locate wind speed column (contains "MeanWindSpeed")
        ws_col = next((c for c in df.columns if "MeanWindSpeed" in c), None)
        if ws_col is None:
            st.error(f"No MeanWindSpeed column found. Columns: {list(df.columns[:6])}")
            return None, ""

        # Drop rows flagged as bad by SampleStatus
        ss_col = next((c for c in df.columns if c.startswith("SampleStatus")), None)
        if ss_col:
            bad = pd.to_numeric(df[ss_col], errors="coerce").fillna(0) != 0
            df  = df[~bad]

        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="%Y-%m-%d %H:%M")
        df = df.set_index("TimeStamp")
        df.index.name = "datetime"

        ws = pd.to_numeric(df[ws_col], errors="coerce").dropna()
        ws.name = "wind_speed"

        # Extract hub height from column name (e.g. "_125.0m_")
        height_str = ""
        hm = re.search(r"_(\d+\.?\d*)m_", ws_col)
        if hm:
            height_str = f" {hm.group(1)}m"

        col_label = f"Mean wind speed{height_str} [{tz_label}]" if tz_label else f"Mean wind speed{height_str}"
        return ws, col_label
    except Exception as e:
        st.error(f"Failed to parse WindPRO Meteo export: {e}")
        return None, ""


def _load_direction_from_raw(raw: bytes) -> pd.Series | None:
    """
    Try to extract a wind direction series from already-read file bytes.
    Handles all three formats (WindPRO Meteo TXT, WindPRO time-series CSV, standard CSV).
    Returns None if no direction column can be found.
    """
    try:
        first_line = raw.decode("utf-8", errors="replace").split("\n", 1)[0].lower()
        if "windpro" in first_line:
            if "meteo data export" in first_line:
                # TXT: direction is the DirectionUID column
                text = raw.decode("utf-8", errors="replace")
                lines = text.splitlines()
                hi = next(i for i, l in enumerate(lines) if l.split("\t")[0].strip() == "TimeStamp")
                data_text = "\n".join([lines[hi]] + lines[hi + 2:])
                df = pd.read_csv(io.StringIO(data_text), sep="\t", low_memory=False)
                dir_col = next((c for c in df.columns if "DirectionUID" in c), None)
                if dir_col is None:
                    return None
                ss_col = next((c for c in df.columns if c.startswith("SampleStatus")), None)
                if ss_col:
                    df = df[pd.to_numeric(df[ss_col], errors="coerce").fillna(0) == 0]
                df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="%Y-%m-%d %H:%M")
                wd = pd.to_numeric(df.set_index("TimeStamp")[dir_col], errors="coerce").dropna()
            else:
                # CSV: direction is column 3
                df = pd.read_csv(io.BytesIO(raw), skiprows=6, header=None)
                df[1] = pd.to_datetime(df[1], format="%d/%m/%Y %H:%M", dayfirst=True)
                df = df.set_index(1)
                wd = pd.to_numeric(df[3], errors="coerce").dropna()
        else:
            skip = _skip_comments(raw)
            df = pd.read_csv(io.BytesIO(raw), skiprows=skip, index_col=0, parse_dates=True)
            dir_col = next(
                (c for c in df.columns
                 if "wd_" in c.lower() or "direction" in c.lower() or c.lower().endswith("_deg")),
                None,
            )
            if dir_col is None:
                return None
            wd = pd.to_numeric(df[dir_col], errors="coerce").dropna()

        wd.name = "wind_direction"
        if wd.index.tz is not None:
            wd = wd.tz_localize(None)
        return wd
    except Exception:
        return None


def _load_series(f, ws_col: str | None = None) -> tuple[pd.Series | None, str]:
    """
    Load a wind speed Series from a Streamlit UploadedFile.
    Handles '#' comment headers and auto-detects wind speed column.
    Returns (series_with_naive_index, column_name) or (None, "").
    Timezone info is stripped — caller is responsible for alignment.
    """
    try:
        raw = f.read()
        if _is_windpro(raw):
            first = raw.decode("utf-8", errors="replace").split("\n", 1)[0].lower()
            if "meteo data export" in first:
                st.info("WindPRO Meteo Data Export detected — parsing automatically.")
                return _load_windpro_meteo_series(raw)
            st.info("WindPRO time-series export detected — parsing automatically.")
            return _load_windpro_series(raw)
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


def _scan_tz_offsets(model: pd.Series, meas: pd.Series) -> pd.Series:
    """
    Sweep measured timestamp shift from -14 to +14 h in 0.5 h steps.
    Returns a Series indexed by offset (hours) containing hourly Pearson r.
    The offset with the highest r is the suggested timezone alignment.
    """
    m_h = model.copy()
    if m_h.index.tz is not None:
        m_h = m_h.tz_localize(None)
    m_h  = m_h.resample("h").mean().dropna()
    ms_h = meas.resample("h").mean().dropna()

    offsets = np.arange(-14.0, 14.5, 0.5)
    results = {}
    for offset in offsets:
        shifted = ms_h.copy()
        shifted.index = shifted.index + pd.Timedelta(hours=float(offset))
        idx = m_h.index.intersection(shifted.index)
        if len(idx) < 24 * 14:          # need at least 2 weeks
            results[float(offset)] = np.nan
        else:
            results[float(offset)] = float(m_h.loc[idx].corr(shifted.loc[idx]))
    return pd.Series(results, name="r")


def _chart_tz_scan(r_series: pd.Series, best_offset: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    valid = r_series.dropna()
    ax.plot(valid.index, valid.values, color="#0F172A", lw=1.5, zorder=3)
    ax.axvline(best_offset, color="#4F46E5", lw=1.8, ls="--", zorder=4,
               label=f"Best: {best_offset:+.1f} h  (r = {r_series.loc[best_offset]:.3f})")
    ax.axvline(0.0, color="#94A3B8", lw=1.0, ls=":", zorder=2, label="No shift (0 h)")
    ax.set_xlabel("Shift applied to measured timestamps (hours)")
    ax.set_ylabel("Pearson r (hourly)")
    ax.set_title("Correlation vs timezone offset — auto scan", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


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
    """Scale per-timestep deviation from the long-term mean by s.
    Preserves the long-term mean; changes the diurnal profile shape."""
    M = float(ws.mean())
    return (M + s * (ws - M)).clip(lower=0.0)


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

    # Pearson r: hourly timestep-by-timestep correlation on the overlap period
    r = float(model_ov.corr(meas_ov))

    return dict(
        amplitude_scale=s,
        mean_multiplier=k,
        rmse_before=rmse_b,
        rmse_after=rmse_a,
        mean_meas=float(meas_ov.mean()),
        mean_model_raw=float(model_ov.mean()),
        mean_model_corrected=float(scaled.mean() * k),
        r=r,
        r2=r ** 2,
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
# Parameter file builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_params_file(
    site_label: str,
    rep: dict,
    result: dict,
    s_use: float,
    k_use: float,
    apply_amp: bool,
    apply_mean: bool,
    meas_shift: float,
    original_mean: float,
    corrected_mean: float,
    hub_h,
) -> bytes:
    """Build a human-readable plain-text parameter file for the calibration results."""
    missing_str = (
        ", ".join(MONTH_NAMES[m - 1] for m in rep["missing"]) if rep["missing"] else "none"
    )
    W = 65  # line width for separator
    lines = [
        "=" * W,
        "  ERA5 × GWA Wind Tool — Calibration Parameters",
        "=" * W,
        f"  Site:        {site_label}",
        f"  Hub height:  {hub_h} m",
        f"  Generated:   {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "-" * W,
        "  CALIBRATION PERIOD",
        "-" * W,
        f"  Overlap:         {rep['start'].strftime('%Y-%m-%d')} to {rep['end'].strftime('%Y-%m-%d')}",
        f"  Duration:        {rep['n_cal_months']} calendar months  ({rep['n_hours']:,} hourly records)",
        f"  Coverage:        {rep['quality'].replace('_', ' ')}",
        f"  Missing months:  {missing_str}",
        f"  Measured mean (overlap):      {result['mean_meas']:.3f} m/s",
        f"  Model mean (overlap):         {rep['ov_mean']:.3f} m/s",
        f"  Model mean (long-term):       {rep['full_mean']:.3f} m/s",
        f"  Concurrent / long-term ratio: {rep['rep_ratio']:.3f}",
        f"  Timestamp shift applied:      {meas_shift:+.1f} h  (to measured data)",
        "",
        "-" * W,
        "  QUALITY METRICS  (concurrent overlap period)",
        "-" * W,
        f"  Hourly Pearson r:      {result['r']:.4f}",
        f"  R² (variance expl.):   {result['r2']:.4f}",
        f"  Diurnal RMSE before:   {result['rmse_before']:.4f} m/s",
        f"  Diurnal RMSE after:    {result['rmse_after']:.4f} m/s",
        f"  RMSE improvement:      {(result['rmse_before']-result['rmse_after'])/result['rmse_before']*100:.1f}%",
        "",
        "-" * W,
        "  CORRECTION FACTORS",
        "-" * W,
        f"  Amplitude scale  (s):   {result['amplitude_scale']:.4f}   {'[APPLIED]' if apply_amp else '[derived only — not applied]'}",
        f"  Mean multiplier  (k):   {result['mean_multiplier']:.6f}   {'[APPLIED]' if apply_mean else '[derived only — not applied]'}",
        "",
        f"  Long-term mean (original):   {original_mean:.3f} m/s",
        f"  Long-term mean (corrected):  {corrected_mean:.3f} m/s",
        "",
        "-" * W,
        "  HOW TO USE THESE PARAMETERS",
        "-" * W,
        "",
        "  AMPLITUDE SCALE (s):",
        f"    Value: {result['amplitude_scale']:.4f}",
        "    Where: Main tool > Advanced settings > Diurnal amplitude scale (s)",
        "    Effect: Scales each timestep's deviation from the long-term mean M.",
        "            Formula: ws_corr = M + s × (ws − M)",
        "            s < 1 → flatter profile (raises low hours, lowers high hours).",
        "            s > 1 → more pronounced day/night swing.",
        "            Does not change long-term mean.",
        "    Cross-site: Moderately transferable to nearby sites with similar",
        "            terrain and stability regime.",
        "",
        "  MEAN MULTIPLIER (k):",
        f"    Value: {result['mean_multiplier']:.6f}",
        "    Where: Main tool > Advanced settings > Mean wind speed multiplier (k)",
        "    Effect: Multiplies all wind speeds after diurnal correction.",
        "            k > 1 → model underestimated mean; k < 1 → overestimated.",
        "            Changes long-term mean.",
        "    Cross-site: Site-specific. DO NOT transfer to other sites.",
        "",
        "  REPRODUCIBILITY:",
        "    Both corrections are applied post-Weibull in the main tool using",
        "    the identical formula to this Calibration page. Entering s and k",
        "    from this file into the main tool's Advanced settings and",
        "    re-running will produce output IDENTICAL to the calibrated CSV.",
        "",
        "=" * W,
    ]
    return "\n".join(lines).encode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CSV builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_csv(
    final_series: pd.Series,
    dir_series: pd.Series | None,
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
        f"# Hourly correlation:  r = {result['r']:.4f},  R² = {result['r2']:.4f}  (overlap period)",
        "#",
        "# --- Method ---",
        "# Step 1 (amplitude): ws_corr(t) = M + s * (ws(t) - M)",
        "#   M = long-term mean wind speed of the synthetic record.",
        "#   s optimised to minimise RMSE of 24-hr mean diurnal profiles (overlap period).",
        "#   s < 1 flattens the diurnal cycle (raises low hours, lowers high hours).",
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
    col_ws  = f"calibrated_ws_{hub_h}m_ms"
    out_df  = final_series.round(2).to_frame(name=col_ws)
    if dir_series is not None:
        wd = dir_series.resample("h").mean().reindex(final_series.index)
        out_df["wind_direction_deg"] = wd.round(0).astype("Int64")
    out_df.to_csv(buf)
    return buf.getvalue().encode("utf-8")


def _build_csv_clean(
    final_series: pd.Series,
    dir_series: pd.Series | None,
    hub_h,
) -> bytes:
    """
    Plain CSV with no # comment lines — opens directly in Excel and can be
    imported into WindPRO via the Meteo Object import wizard.
    Timestamps: YYYY-MM-DD HH:MM  Wind speed: m/s (2 dp)  Direction: integer degrees
    """
    col_ws = f"wind_speed_{hub_h}m_ms"
    out_df = final_series.round(2).to_frame(name=col_ws)
    if dir_series is not None:
        wd = dir_series.resample("h").mean().reindex(final_series.index)
        out_df["wind_direction_deg"] = wd.round(0).astype("Int64")
    out_df.index = out_df.index.strftime("%Y-%m-%d %H:%M")
    out_df.index.name = "datetime"
    return out_df.to_csv().encode("utf-8")


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

> `ws_corr(t) = M + s × (ws(t) − M)`

where `M` is the long-term mean wind speed of the synthetic record. `s` is found
by minimising the RMSE between the corrected model and measured 24-hour mean
diurnal profiles over the overlap period.
`s < 1` → flatter profile (raises low-wind hours, lowers high-wind hours); `s > 1` → more pronounced swing.
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
model_dir:  pd.Series | None = None
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
        # batch_site_data does not store direction — model_dir stays None
    else:
        _sdf        = st.session_state["site_df"]
        model_full  = _sdf["ws_150m_corrected"].dropna()
        model_label = st.session_state.get("site_name_input", "Site")
        try:
            hub_h = int(
                st.session_state.get("site_meta", {}).get(
                    "hub_height", st.session_state.get("aep_hub_height", "?")
                )
            )
        except (TypeError, ValueError):
            hub_h = "?"
        if "wd_100m" in _sdf.columns:
            _wd = _sdf["wd_100m"].dropna()
            if _wd.index.tz is not None:
                _wd = _wd.tz_localize(None)
            model_dir = _wd

    if model_full is not None:
        st.caption(
            f"**{model_label}** · {len(model_full):,} hourly records · "
            f"{model_full.index[0].date()} → {model_full.index[-1].date()} · "
            f"mean {model_full.mean():.2f} m/s · hub {hub_h} m"
            + (" · direction included" if model_dir is not None else "")
        )
else:
    model_upload = st.file_uploader(
        "Upload model output CSV",
        type=["csv"],
        help="Tool-generated CSV (# comment headers OK). "
             "Wind speed and direction columns auto-detected.",
        key="calib_model_file",
    )
    if model_upload:
        ws_hint = st.text_input(
            "Wind speed column (leave blank to auto-detect)", value="", key="calib_model_col"
        )
        _raw_model  = model_upload.read()
        model_full, detected = _load_series(io.BytesIO(_raw_model), ws_hint or None)
        model_dir   = _load_direction_from_raw(_raw_model)
        if model_full is not None:
            model_label = model_upload.name
            st.caption(
                f"Column **{detected}** · {len(model_full):,} records · "
                f"{model_full.index[0].date()} → {model_full.index[-1].date()} · "
                f"mean {model_full.mean():.2f} m/s"
                + (f" · direction: {len(model_dir):,} records" if model_dir is not None else " · no direction column found")
            )

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Measured data
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 2 — Measured wind speed")

meas_upload = st.file_uploader(
    "Upload measured wind speed CSV or WindPRO export",
    type=["csv", "txt"],
    help="Accepts: (1) any CSV with datetime index + wind speed column in m/s; "
         "(2) WindPRO time-series export CSV (auto-detected); "
         "(3) WindPRO Meteo Data Export .txt (auto-detected). "
         "'#' comment header lines are skipped automatically.",
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

# Run offset scan (cached per dataset pair)
scan_key = f"_tz_scan_{model_label}_{len(model_full)}_{int(model_full.mean()*100)}_{len(meas_full)}_{int(meas_full.mean()*100)}"
if scan_key not in st.session_state:
    with st.spinner("Scanning timezone offsets (−14 to +14 h)…"):
        st.session_state[scan_key] = _scan_tz_offsets(model_full, meas_full)

r_scan      = st.session_state[scan_key]
best_offset = float(r_scan.idxmax())
best_r      = float(r_scan.max())
r_at_zero   = float(r_scan.get(0.0, np.nan))

# Show scan chart
fig_scan = _chart_tz_scan(r_scan, best_offset)
st.pyplot(fig_scan)
plt.close(fig_scan)

# Interpret the result
if abs(best_offset) < 0.25:
    st.markdown(
        f'<div class="good">✓ Peak correlation at <b>0 h shift</b> (r = {best_r:.3f}) — '
        "timezones appear aligned. No adjustment needed.</div>",
        unsafe_allow_html=True,
    )
elif best_offset % 1.0 != 0.0:
    # Non-integer hour — could be genuine ERA5 timing lag, not just timezone
    st.markdown(
        f'<div class="warn">⚠️ Suggested shift: <b>{best_offset:+.1f} h</b> (r = {best_r:.3f} vs '
        f'r = {r_at_zero:.3f} at 0 h). The non-integer offset may reflect a genuine '
        "timing lag in ERA5 mesoscale events rather than a timezone issue — consider "
        "rounding to the nearest whole hour for timezone correction only.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="warn">⚠️ Suggested shift: <b>{best_offset:+.1f} h</b> (r = {best_r:.3f} vs '
        f'r = {r_at_zero:.3f} at 0 h). Enter this in the field below to align the timestamps.</div>',
        unsafe_allow_html=True,
    )

# Pre-populate shift input with suggested value on first load
_shift_init_key = f"_shift_init_{scan_key}"
if _shift_init_key not in st.session_state:
    st.session_state["calib_tz_shift"] = best_offset
    st.session_state[_shift_init_key]  = True

meas_shift = st.number_input(
    "Shift measured timestamps by (hours)",
    min_value=-14.0,
    max_value=14.0,
    value=st.session_state.get("calib_tz_shift", best_offset),
    step=0.5,
    key="calib_tz_shift",
    help=(
        "Add this many hours to measured timestamps before matching with the model. "
        "Pre-filled with the scan suggestion above. "
        "Example: if model is UTC+8 and logger records UTC, set to +8."
    ),
)
if meas_shift != 0.0:
    st.markdown(
        f'<div class="warn">Measured timestamps shifted by <b>{meas_shift:+.1f} h</b> '
        "before overlap matching.</div>",
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

m1, m2, m3, m4, m5, m6 = st.columns(6)
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
    "Corrected mean vs measured",
    f"{result['mean_model_corrected']:.2f} vs {result['mean_meas']:.2f} m/s",
)
m6.metric(
    "Correlation (r)",
    f"{result['r']:.3f}",
    help=(
        "Pearson r between hourly model and measured wind speeds over the concurrent "
        "overlap period. Measures how well ERA5 captures weather event timing. "
        f"R² = {result['r2']:.3f} (proportion of variance explained)."
    ),
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

slug      = model_label.replace(" ", "_").replace("/", "_")
now_str   = datetime.now().strftime("%Y%m%d_%H%M")

csv_bytes = _build_csv(
    final_series, model_dir, model_label, rep, result,
    s_use, k_use, apply_amp, apply_mean,
    float(model_full.mean()), hub_h,
)
clean_csv_bytes = _build_csv_clean(final_series, model_dir, hub_h)
params_bytes = _build_params_file(
    model_label, rep, result,
    s_use, k_use, apply_amp, apply_mean,
    meas_shift,
    float(model_full.mean()), float(final_series.mean()), hub_h,
)

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button(
        "⬇ Download calibrated CSV (full metadata)",
        data=csv_bytes,
        file_name=f"calibrated_{slug}_{now_str}.csv",
        mime="text/csv",
        help="Includes # comment header with calibration metadata. "
             "Open in a text editor or Python/pandas (skiprows auto-handled).",
    )
with dl2:
    st.download_button(
        "⬇ Download clean CSV (Excel / WindPRO)",
        data=clean_csv_bytes,
        file_name=f"calibrated_clean_{slug}_{now_str}.csv",
        mime="text/csv",
        help="No comment lines — opens directly in Excel. "
             "Import into WindPRO via Meteo Object → Import Wizard, "
             "mapping 'datetime' → timestamp, 'wind_speed_*m_ms' → mean wind speed, "
             "'wind_direction_deg' → wind direction.",
    )
with dl3:
    st.download_button(
        "⬇ Download calibration parameters (.txt)",
        data=params_bytes,
        file_name=f"calib_params_{slug}_{now_str}.txt",
        mime="text/plain",
        help="Human-readable record of all calibration factors, quality metrics, "
             "and guidance on where to enter each parameter in the main tool.",
    )

st.markdown(
    '<div class="good">✓ <b>Fully reproducible:</b> Both corrections are applied post-Weibull '
    "using the same formula in the main tool and this page. Enter s and k from the parameter "
    "file into <b>Advanced settings</b> in the main tool and re-run — you will get output "
    "identical to the calibrated CSV above.</div>",
    unsafe_allow_html=True,
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
| **Amplitude scale (s)** | **{s_opt:.2f}** | Moderately transferable to nearby sites with similar terrain, fetch, and stability regime. Enter in *Advanced settings → Diurnal amplitude scale (s)* in the main tool. |
| **Mean multiplier (k)** | **{k_opt:.4f}** | Site-specific (roughness + terrain bias). **Do not transfer** to other sites. Enter in *Advanced settings → Mean wind speed multiplier (k)* only for this site. |

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
