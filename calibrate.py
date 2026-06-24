"""
calibrate.py — Fit amplitude_scale (and optionally mean-α clamp bounds) to
site measurements by minimising RMSE of the hourly diurnal wind speed profile.

Usage
-----
python calibrate.py --measured  path/to/measured.csv  \
                    --modelled  path/to/model_output.csv \
                    [--ws-col   COLUMN_NAME]  \
                    [--fit-alpha-bounds]      \
                    [--plot]

Measured CSV
    Any CSV with a datetime index and a wind speed column (m/s).
    Supported formats:
      • Tool output CSV (has '#' comment header lines — skipped automatically)
      • Plain CSV with a datetime column and a wind speed column
      • WindPRO time-series export CSV (6-line header, auto-detected)
      • WindPRO Meteo Data Export TXT (tab-separated, auto-detected)

Modelled CSV
    The tool's hourly output CSV for the same site (gwa_corrected_ws_*m_ms column).
    The tool must have been run to produce this before calibrating.

The script:
  1. Aligns measured and modelled to a common hourly index.
  2. Computes the 24-hour mean profile for both.
  3. Optimises amplitude_scale to minimise RMSE between those profiles.
  4. (Optional) then fits mean-α clamp bounds for mean speed alignment.
  5. Reports the result and (with --plot) saves a comparison PNG.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize

# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv_skip_comments(path: Path) -> pd.DataFrame:
    """Read a CSV that may have '#'-prefixed comment lines at the top."""
    with open(path, "r", encoding="utf-8") as f:
        skip = sum(1 for line in f if line.startswith("#"))
    return pd.read_csv(path, skiprows=skip, index_col=0, parse_dates=True)


def _is_windpro(path: Path) -> bool:
    """Return True if the file looks like any WindPRO export."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    return "windpro" in first.lower()


def _load_windpro_series(path: Path) -> pd.Series:
    """
    Parse a WindPRO time-series CSV export (6-line header, timestamp in col 1,
    wind speed in col 2, date format d/mm/yyyy H:MM).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    tz_label = ""
    if len(lines) >= 5:
        m = re.search(r"UTC[+\-]\d{2}:\d{2}", lines[4])
        if m:
            tz_label = m.group(0)

    df = pd.read_csv(path, skiprows=6, header=None)
    df[1] = pd.to_datetime(df[1], format="%d/%m/%Y %H:%M", dayfirst=True)
    df = df.set_index(1)
    df.index.name = "datetime"

    ws = pd.to_numeric(df[2], errors="coerce").dropna()
    if tz_label:
        print(f"  WindPRO time-series export — timestamps are {tz_label}")
    return ws.rename("wind_speed")


def _load_windpro_meteo_series(path: Path) -> pd.Series:
    """
    Parse a WindPRO Meteo Data Export (.txt, tab-separated).
    Finds the header row by detecting 'TimeStamp' as the first field,
    skips the units row, and extracts the MeanWindSpeed column.
    Drops rows where SampleStatus != 0.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    tz_label = ""
    for line in lines:
        m = re.search(r"UTC[+\-]\d{2}:\d{2}", line)
        if m:
            tz_label = m.group(0)
            break

    header_idx = None
    for i, line in enumerate(lines):
        if line.split("\t")[0].strip() == "TimeStamp":
            header_idx = i
            break

    if header_idx is None:
        sys.exit(f"Cannot find TimeStamp header row in WindPRO Meteo export: {path}")

    import io as _io
    data_text = "".join([lines[header_idx]] + lines[header_idx + 2:])
    df = pd.read_csv(_io.StringIO(data_text), sep="\t", low_memory=False)

    ws_col = next((c for c in df.columns if "MeanWindSpeed" in c), None)
    if ws_col is None:
        sys.exit(f"No MeanWindSpeed column in {path}. Columns: {list(df.columns[:5])}")

    ss_col = next((c for c in df.columns if c.startswith("SampleStatus")), None)
    if ss_col:
        bad = pd.to_numeric(df[ss_col], errors="coerce").fillna(0) != 0
        df  = df[~bad]

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], format="%Y-%m-%d %H:%M")
    df = df.set_index("TimeStamp")

    ws = pd.to_numeric(df[ws_col], errors="coerce").dropna()
    if tz_label:
        print(f"  WindPRO Meteo Data Export — timestamps are {tz_label}")
    return ws.rename("wind_speed")


def _load_series(path: Path, ws_col: str | None) -> pd.Series:
    """Load a wind speed Series from a CSV, auto-detecting the speed column."""
    if _is_windpro(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            first = f.readline().lower()
        if "meteo data export" in first:
            s = _load_windpro_meteo_series(path)
            print(f"  Auto-selected 'MeanWindSpeed' column from {path.name} (WindPRO Meteo format)")
        else:
            s = _load_windpro_series(path)
            print(f"  Auto-selected 'Mean wind speed' column from {path.name} (WindPRO time-series format)")
        if s.index.tz is not None:
            s = s.tz_localize(None)
        return s

    df = _read_csv_skip_comments(path)

    if ws_col and ws_col in df.columns:
        col = ws_col
    else:
        # Auto-detect: prefer gwa_corrected_* or measured hub-height column
        candidates = [c for c in df.columns if "gwa_corrected" in c.lower() or "ws_" in c.lower()]
        if not candidates:
            candidates = [c for c in df.columns if df[c].dtype in (float, np.float64)]
        if not candidates:
            sys.exit(f"Cannot find a wind speed column in {path}. Use --ws-col to specify it.")
        col = candidates[0]
        print(f"  Auto-selected column '{col}' from {path.name}")

    s = df[col].dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        sys.exit(f"Could not parse datetime index in {path}. Check the file format.")

    # Strip timezone for alignment
    if s.index.tz is not None:
        s = s.tz_localize(None)

    return s.rename(col)


def diurnal_profile(s: pd.Series) -> pd.Series:
    """24-hour mean wind speed profile (index 0–23)."""
    return s.groupby(s.index.hour).mean()


def align_hourly(measured: pd.Series, modelled: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Resample both to hourly means and align on common index."""
    m_h = measured.resample("h").mean().dropna()
    mod_h = modelled.resample("h").mean().dropna()
    common = m_h.index.intersection(mod_h.index)
    if len(common) < 24 * 30:
        print(f"  Warning: only {len(common)} common hourly records — at least 30 days recommended.")
    return m_h.loc[common], mod_h.loc[common]


# ── Objective functions ───────────────────────────────────────────────────────

def _scale_modelled(mod_h: pd.Series, meas_h: pd.Series, amplitude_scale: float) -> pd.Series:
    """
    Apply amplitude_scale to the modelled series.

        ws_scaled(t) = M + scale × (ws(t) − M)

    where M is the long-term mean. Preserves the mean; scales deviations from it,
    which changes the 24-hour mean diurnal profile shape.
    """
    M = float(mod_h.mean())
    return (M + amplitude_scale * (mod_h - M)).clip(lower=0)


def rmse_diurnal(amplitude_scale: float, mod_h: pd.Series, meas_h: pd.Series) -> float:
    """RMSE between 24-hour mean profiles at a given amplitude_scale."""
    scaled = _scale_modelled(mod_h, meas_h, amplitude_scale)
    prof_mod = diurnal_profile(scaled)
    prof_meas = diurnal_profile(meas_h)
    common_h = prof_mod.index.intersection(prof_meas.index)
    err = prof_mod.loc[common_h].values - prof_meas.loc[common_h].values
    return float(np.sqrt(np.mean(err ** 2)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Calibrate ERA5×GWA tool amplitude_scale to measurements")
    parser.add_argument("--measured",  required=True,  type=Path, help="Measured wind speed CSV")
    parser.add_argument("--modelled",  required=True,  type=Path, help="Tool output CSV (hourly)")
    parser.add_argument("--ws-col",    default=None,   help="Wind speed column name (auto-detected if omitted)")
    parser.add_argument("--plot",      action="store_true", help="Save comparison plot as calibration_result.png")
    args = parser.parse_args()

    print(f"\nLoading measured:  {args.measured}")
    meas = _load_series(args.measured, args.ws_col)
    print(f"  {len(meas)} records, {meas.index[0].date()} → {meas.index[-1].date()}")

    print(f"Loading modelled:  {args.modelled}")
    mod  = _load_series(args.modelled, ws_col=None)
    print(f"  {len(mod)} records, {mod.index[0].date()} → {mod.index[-1].date()}")

    print("\nAligning to common hourly index…")
    meas_h, mod_h = align_hourly(meas, mod)
    print(f"  {len(meas_h)} common hourly records ({len(meas_h)/8760:.1f} yr)")

    # ── Step 1: Fit amplitude_scale ───────────────────────────────────────────
    print("\nOptimising amplitude_scale…")
    result = minimize_scalar(
        rmse_diurnal,
        bounds=(0.3, 4.0),
        method="bounded",
        args=(mod_h, meas_h),
        options={"xatol": 0.005},
    )
    best_scale = float(result.x)
    rmse_before = rmse_diurnal(1.0, mod_h, meas_h)
    rmse_after  = rmse_diurnal(best_scale, mod_h, meas_h)

    print(f"\n  Optimal amplitude_scale : {best_scale:.3f}")
    print(f"  Diurnal RMSE  before    : {rmse_before:.4f} m/s  (scale = 1.00)")
    print(f"  Diurnal RMSE  after     : {rmse_after:.4f} m/s  (scale = {best_scale:.3f})")
    print(f"  Improvement             : {(rmse_before - rmse_after) / rmse_before * 100:.1f}%")

    # ── Step 2: Mean speed bias ───────────────────────────────────────────────
    scaled_h = _scale_modelled(mod_h, meas_h, best_scale)
    mean_meas  = float(meas_h.mean())
    mean_mod   = float(mod_h.mean())
    mean_scaled = float(scaled_h.mean())
    mean_bias_before = mean_mod - mean_meas
    mean_bias_after  = mean_scaled - mean_meas

    print(f"\n  Mean wind speed — measured : {mean_meas:.3f} m/s")
    print(f"  Mean wind speed — model    : {mean_mod:.3f} m/s  (bias {mean_bias_before:+.3f} m/s)")
    print(f"  Mean wind speed — scaled   : {mean_scaled:.3f} m/s  (bias {mean_bias_after:+.3f} m/s)")
    print("  Note: mean bias is corrected by the GWA Weibull step, not amplitude_scale.")
    print("        A residual mean bias usually means the roughness class or GWA node is off.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("RESULT — enter these values in the Advanced settings panel:")
    print(f"  Diurnal amplitude scale : {best_scale:.2f}")
    print("─" * 60)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.ticker as mticker

            prof_meas   = diurnal_profile(meas_h)
            prof_before = diurnal_profile(mod_h)
            prof_after  = diurnal_profile(scaled_h)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            fig.suptitle(
                f"ERA5×GWA Calibration  —  amplitude_scale = {best_scale:.2f}",
                fontsize=12, fontweight="bold",
            )

            # Left: diurnal profile comparison
            ax = axes[0]
            ax.plot(prof_meas.index,  prof_meas.values,   "k-",  lw=2.0, label="Measured")
            ax.plot(prof_before.index, prof_before.values, "--",  lw=1.5, color="#94A3B8", label="Model (scale=1.00)")
            ax.plot(prof_after.index,  prof_after.values,  "-",   lw=2.0, color="#4F46E5", label=f"Model (scale={best_scale:.2f})")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Mean wind speed (m/s)")
            ax.set_title("Diurnal profile")
            ax.xaxis.set_major_locator(mticker.MultipleLocator(3))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Right: RMSE vs amplitude_scale sweep
            ax2 = axes[1]
            scales = np.linspace(0.3, 3.5, 120)
            rmses  = [rmse_diurnal(s, mod_h, meas_h) for s in scales]
            ax2.plot(scales, rmses, color="#0F172A", lw=1.5)
            ax2.axvline(best_scale, color="#4F46E5", lw=1.5, linestyle="--",
                        label=f"Optimal = {best_scale:.2f}")
            ax2.axvline(1.0, color="#94A3B8", lw=1.0, linestyle=":",
                        label="Default = 1.00")
            ax2.set_xlabel("amplitude_scale")
            ax2.set_ylabel("Diurnal RMSE (m/s)")
            ax2.set_title("Objective function")
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            out = Path("calibration_result.png")
            fig.savefig(out, dpi=150)
            print(f"\nPlot saved → {out.resolve()}")
        except ImportError:
            print("\nmatplotlib not available — skipping plot. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
