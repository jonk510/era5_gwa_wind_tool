"""
PDF QA report generator for ERA5 × GWA Wind Resource Assessment batch results.

Typography and layout derived from GHD Shadow Flicker Assessment reference document:
  Font:          Arial (Bold, Regular, Italic, Bold Italic)
  H1:            14 pt bold black, thin grey rule below
  H2:            11 pt bold black
  Body:          9 pt regular black (#000000)
  Table header:  9 pt bold white on black (#000000) background
  Table body:    9 pt regular black, white rows, grey horizontal rules (#A1A1A1)
  Caption:       8 pt bold italic, dark grey (#404040)
  Page header:   7.5 pt regular, dark grey, thin rule below
  Page footer:   6.5 pt italic, dark grey, thin rule above
"""

import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "axes.labelsize":    8,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "figure.facecolor":  "white",
})
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from scipy.stats import weibull_min

try:
    import contextily as ctx
    from pyproj import Transformer as _ProjT
    _HAS_CTX = True
except ImportError:
    _HAS_CTX = False

# ── Design tokens ─────────────────────────────────────────────────────────────
# Extracted from GHD Shadow Flicker Assessment reference PDF
C_BLACK    = "#000000"   # table header background, headings
C_BODY     = "#000000"   # body text
C_BORDER   = "#A1A1A1"   # table/rule borders
C_CAPTION  = "#404040"   # figure/table captions
C_WHITE    = "#FFFFFF"   # table header text
C_LIGHT    = "#F5F5F5"   # very light grey — cover metadata band
C_NAVY     = "#1B3A6B"   # chart accent (dark professional blue)
C_SLATE    = "#64748B"   # secondary chart colour
C_AMBER    = "#B45309"   # chart highlight
C_RED      = "#B91C1C"   # chart warning/reference line

A4         = (8.27, 11.69)   # A4 portrait inches
ML         = 0.095           # left margin (figure fraction, ≈ 20 mm)
MR         = 0.905           # right margin
AVAIL      = MR - ML         # available width fraction

WIND_CMAP  = LinearSegmentedColormap.from_list(
    "wind", ["#DC2626", "#FBBF24", "#16A34A"], N=256
)


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _fig():
    return plt.figure(figsize=A4, facecolor="white")


def _rule(fig, y: float, color=C_BORDER, x0=None, width=None):
    """Draw a horizontal rule across the content area."""
    x0    = x0    if x0    is not None else ML
    width = width if width is not None else AVAIL
    ax = fig.add_axes([x0, y, width, 0.0008])
    ax.set_facecolor(color)
    ax.axis("off")


def _page_chrome(fig, section: str = "", page_label: str = ""):
    """Running header, thin rule above and below it, and footer rule + disclaimer."""
    # ── Header ──────────────────────────────────────────────────────────────
    _rule(fig, 0.957)
    hdr = "ERA5 × GWA Wind Tool"
    if section:
        hdr = f"ERA5 × GWA Wind Tool  |  {section}"
    fig.text(ML,  0.966, hdr,          fontsize=7.5, color=C_CAPTION, va="bottom")
    fig.text(MR,  0.966, page_label,   fontsize=7.5, color=C_CAPTION, va="bottom",
             ha="right")
    _rule(fig, 0.953)

    # ── Footer ──────────────────────────────────────────────────────────────
    _rule(fig, 0.043)
    fig.text(
        ML, 0.038,
        "INDICATIVE ONLY — Results synthesised from ERA5 reanalysis (Open-Meteo) "
        "and Global Wind Atlas. Not a bankable wind resource assessment. "
        "Not a substitute for on-site measurement.",
        fontsize=6.5, color=C_CAPTION, va="top", style="italic",
    )


def _header_band(fig, y_top: float, height: float, title: str, subtitle: str = ""):
    """Black section-header band (GHD reference style)."""
    ax = fig.add_axes([0, y_top - height, 1, height])
    ax.set_facecolor(C_BLACK)
    ax.text(ML, 0.62, title,
            transform=ax.transAxes, color=C_WHITE,
            fontsize=14, fontweight="bold", va="center")
    if subtitle:
        ax.text(ML, 0.22, subtitle,
                transform=ax.transAxes, color="#CCCCCC",
                fontsize=8.5, va="center", style="italic")
    ax.axis("off")
    return ax


def _section_title(fig, y: float, text: str, level: int = 1) -> float:
    """Numbered section heading with optional thin rule below (H1) or plain (H2)."""
    if level == 1:
        fig.text(ML, y, text, fontsize=13, color=C_BLACK, fontweight="bold", va="top")
        y -= 0.024
        _rule(fig, y + 0.004)
        return y - 0.010
    else:
        fig.text(ML, y, text, fontsize=11, color=C_BLACK, fontweight="bold", va="top")
        return y - 0.022


def _table_caption(fig, y: float, label: str, caption: str) -> float:
    """'Table X  Caption text' in 8 pt bold italic dark grey (GHD style)."""
    full = f"{label}  {caption}"
    fig.text(ML, y, full, fontsize=8, color=C_CAPTION, va="top",
             fontweight="bold", style="italic")
    return y - 0.022


def _para(fig, y: float, text: str, width: int = 92, size: float = 9.0) -> float:
    """Render wrapped paragraph; returns updated y."""
    for raw_line in text.split("\n"):
        for line in textwrap.wrap(raw_line, width) or [""]:
            if y < 0.06:
                break
            fig.text(ML, y, line, fontsize=size, color=C_BODY, va="top")
            y -= 0.016
        y -= 0.004
    return y


def _bullet(fig, y: float, text: str, width: int = 89, size: float = 9.0) -> float:
    lines = textwrap.wrap(text, width)
    if not lines:
        return y
    fig.text(ML, y, "•  " + lines[0], fontsize=size, color=C_BODY, va="top")
    y -= 0.016
    for cont in lines[1:]:
        fig.text(ML + 0.018, y, cont, fontsize=size, color=C_BODY, va="top")
        y -= 0.016
    return y


# ── Per-site chart helpers ────────────────────────────────────────────────────

def _chart_style(ax):
    """Common chart style: remove top/right spines, light grid."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.5, color=C_BORDER)
    ax.tick_params(labelsize=7.5)


def _plot_weibull(ax, ws_corr: pd.Series, ws_raw: pd.Series, meta: dict):
    ws_c = ws_corr.dropna().values
    ws_r = ws_raw.dropna().values
    ws_c = ws_c[ws_c > 0.1]
    ws_r = ws_r[ws_r > 0.1]
    v = np.linspace(0, max(ws_c.max(), ws_r.max()) * 1.1, 300)

    def _fit_and_plot(ws, color, ls, label_prefix):
        k, _, A = weibull_min.fit(ws, floc=0)
        pdf = weibull_min.pdf(v, k, 0, A)
        ax.hist(ws, bins=40, density=True, alpha=0.20, color=color, edgecolor="none")
        ax.plot(v, pdf, color=color, lw=1.8, ls=ls,
                label=f"{label_prefix}  A={A:.2f}, k={k:.2f}")
        return A, k

    _fit_and_plot(ws_r, C_SLATE, "--", "ERA5 extrapolated")
    A_c, k_c = _fit_and_plot(ws_c, C_NAVY, "-", "GWA-corrected")

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Wind Speed Distribution", pad=4)
    ax.legend(framealpha=0.85)
    ax.set_xlim(left=0)
    _chart_style(ax)
    ax.text(0.97, 0.97, f"A={A_c:.2f} m/s\nk={k_c:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right", color=C_NAVY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec=C_BORDER, lw=0.5))


def _plot_monthly_wind(ax, ws_corr: pd.Series):
    monthly  = ws_corr.resample("ME").mean()
    mon_mean = monthly.groupby(monthly.index.month).mean()
    months   = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    x    = np.arange(1, 13)
    vals = [mon_mean.get(m, np.nan) for m in x]
    bars = ax.bar(x, vals, color=C_NAVY, alpha=0.85, width=0.7,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(ws_corr.mean(), color=C_AMBER, lw=1.2, ls="--",
               label=f"Annual mean {ws_corr.mean():.2f} m/s")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=30)
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.set_title("Monthly Mean Wind Speed", pad=4)
    ax.legend(framealpha=0.85)
    _chart_style(ax)
    for bar in bars:
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=6.5, color=C_BODY)


def _plot_diurnal(ax, ws_corr: pd.Series):
    diurnal = ws_corr.groupby(ws_corr.index.hour).mean()
    hours   = np.arange(24)
    vals    = [diurnal.get(h, np.nan) for h in hours]
    ax.plot(hours, vals, color=C_NAVY, lw=2, marker="o", ms=3)
    ax.fill_between(hours, vals, alpha=0.10, color=C_NAVY)
    ax.axhline(np.nanmean(vals), color=C_AMBER, lw=1, ls="--", label="Daily mean")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], rotation=30)
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_title("Diurnal Wind Profile", pad=4)
    ax.legend(framealpha=0.85)
    _chart_style(ax)


def _plot_monthly_aep(ax, aep: dict):
    gross  = aep["gross_mw"].resample("ME").sum() / 1000
    net    = aep["net_mw"].resample("ME").sum()   / 1000
    mon_g  = gross.groupby(gross.index.month).mean()
    mon_n  = net.groupby(net.index.month).mean()
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]
    x      = np.arange(1, 13)
    ax.bar(x, [mon_g.get(m, 0) for m in x], color=C_SLATE,  alpha=0.55,
           width=0.7, label="Gross AEP")
    ax.bar(x, [mon_n.get(m, 0) for m in x], color=C_NAVY,   alpha=0.85,
           width=0.7, label="Net AEP")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=30)
    ax.set_ylabel("Mean monthly energy (GWh)")
    ax.set_title("Monthly AEP (mean annual)", pad=4)
    ax.legend(framealpha=0.85)
    _chart_style(ax)


def _plot_power_curve(ax, pc_df: pd.DataFrame, wtg: str, nameplate_mw: float,
                      ws_corr: pd.Series):
    ws_pc   = pc_df.index.values
    kw_pc   = pc_df[wtg].values
    rated_kw = kw_pc.max()
    mw_pc   = kw_pc / rated_kw * nameplate_mw

    ax2 = ax.twinx()
    ax2.hist(ws_corr.dropna().values, bins=40, color=C_SLATE, alpha=0.15,
             density=True, label="Wind freq.", zorder=0)
    ax2.set_ylabel("Prob. density", color=C_SLATE)
    ax2.tick_params(colors=C_SLATE)
    ax2.spines["right"].set_color(C_SLATE)

    ax.plot(ws_pc, mw_pc, color=C_NAVY, lw=2, zorder=5, label=wtg)
    ax.fill_between(ws_pc, mw_pc, alpha=0.10, color=C_NAVY, zorder=4)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power (MW)", color=C_NAVY)
    ax.tick_params(axis="y", colors=C_NAVY)
    ax.spines["left"].set_color(C_NAVY)
    ax.set_title("Power Curve", pad=4)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.25, lw=0.5, color=C_BORDER, zorder=1)
    lines,  labs  = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labs + labs2, framealpha=0.85)


def _plot_wake_matrix(ax, wake_df: pd.DataFrame):
    caps = wake_df.columns.values
    ws   = wake_df.index.values
    data = wake_df.values
    vmax = data.max() if data.max() > 0 else 1.0
    im   = ax.pcolormesh(caps, ws, data, cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Wake loss (%)", pad=0.01)

    # Numeric labels in each cell — white text on dark cells, black on light
    for i, w in enumerate(ws):
        for j, c in enumerate(caps):
            val = data[i, j]
            txt_col = "white" if val > vmax * 0.55 else "#222222"
            ax.text(c, w, f"{val:.1f}%",
                    ha="center", va="center",
                    fontsize=6.5, color=txt_col, fontweight="bold")

    ax.set_xlabel("Park nameplate capacity (MW)")
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title("Wake Loss Matrix", pad=4)
    ax.tick_params(labelsize=7.5)


# ── Table rendering ───────────────────────────────────────────────────────────

def _render_table(fig, rows, col_specs, y_start: float, y_min: float = 0.08):
    """
    Render a data table in GHD style: black header, grey horizontal rules, 9 pt Arial.
    Returns the final y position after the last row.

    col_specs: list of (key, label, rel_width)
    """
    keys   = [c[0] for c in col_specs]
    labels = [c[1] for c in col_specs]
    ws_rel = [c[2] for c in col_specs]
    total  = sum(ws_rel)
    col_w  = [w / total * AVAIL for w in ws_rel]

    LINE_H  = 0.027
    HDR_H   = 0.034

    y = y_start

    # ── Black header row ──────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([ML, y - HDR_H, AVAIL, HDR_H])
    ax_hdr.set_facecolor(C_BLACK)
    ax_hdr.axis("off")
    x_frac = 0.0
    for lbl, cw in zip(labels, col_w):
        for li, line in enumerate(lbl.split("\n")):
            ax_hdr.text(
                x_frac / AVAIL + 0.008,
                0.70 - li * 0.32,
                line,
                transform=ax_hdr.transAxes,
                fontsize=9, color=C_WHITE, fontweight="bold", va="top",
            )
        x_frac += cw
    y -= HDR_H + 0.002

    # ── Data rows ─────────────────────────────────────────────────────────────
    overflow_rows = []
    for ri, row in enumerate(rows):
        if y < y_min + LINE_H:
            overflow_rows = rows[ri:]
            break

        ax_row = fig.add_axes([ML, y - LINE_H, AVAIL, LINE_H])
        ax_row.set_facecolor("white")
        ax_row.axis("off")

        x_frac = 0.0
        for key, cw in zip(keys, col_w):
            raw = row.get(key)
            txt = _fmt_cell(raw, key)
            is_name = key == "site_name"
            ax_row.text(
                x_frac / AVAIL + 0.008, 0.52, txt,
                transform=ax_row.transAxes,
                fontsize=9, va="center", color=C_BODY,
                fontweight="bold" if is_name else "normal",
                clip_on=True,
            )
            x_frac += cw

        y -= LINE_H
        # Thin grey horizontal rule below each row
        _rule(fig, y, color=C_BORDER)
        y -= 0.001

    # Outer border: thin rule above the header
    _rule(fig, y_start, color=C_BORDER)

    return y, overflow_rows


def _fmt_cell(val, key):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if key in ("latitude", "longitude"):
        return f"{float(val):.4f}"
    if key in ("era5_mean_100m_ms", "gwa_mean_100m_ms",
               "gwa_mean_hub_ms", "gwa_corrected_mean_hub_ms"):
        return f"{float(val):.2f}"
    if key == "wind_shear_alpha":
        return f"{float(val):.3f}"
    if key == "mean_air_density_kg_m3":
        return f"{float(val):.4f}"
    if key in ("gross_aep_mwh_yr", "net_aep_mwh_yr"):
        v = float(val)
        return f"{v/1000:.1f} GWh" if v >= 1000 else f"{v:.0f} MWh"
    if key in ("nameplate_mw", "mean_wake_loss_pct", "capacity_factor_pct"):
        return f"{float(val):.1f}"
    if key in ("hub_height_m", "elevation_m_asl"):
        try:
            return str(int(val))
        except Exception:
            return str(val)
    return str(val) if val is not None else "-"


# ── Page builders ─────────────────────────────────────────────────────────────

def _page_cover(pdf, summary_rows, start_year, end_year, report_dt, page_num):
    fig = _fig()
    _page_chrome(fig, "Cover", f"{page_num}")

    # ── Title band ────────────────────────────────────────────────────────────
    ax_hdr = fig.add_axes([0, 0.80, 1, 0.155])
    ax_hdr.set_facecolor(C_BLACK)
    ax_hdr.text(ML, 0.72, "ERA5 × GWA Wind Resource Assessment",
                transform=ax_hdr.transAxes, ha="left", va="center",
                color=C_WHITE, fontsize=18, fontweight="bold")
    ax_hdr.text(ML, 0.30, "QA & Review Report",
                transform=ax_hdr.transAxes, ha="left", va="center",
                color="#CCCCCC", fontsize=12, style="italic")
    ax_hdr.axis("off")

    # Thin accent rule below band
    _rule(fig, 0.796, color=C_BORDER)

    # ── Metadata block ────────────────────────────────────────────────────────
    n       = len(summary_rows)
    has_aep = any("gross_aep_mwh_yr" in r for r in summary_rows)
    lats    = [r["latitude"]  for r in summary_rows]
    lons    = [r["longitude"] for r in summary_rows]

    meta_lines = [
        ("Sites assessed",     str(n)),
        ("ERA5 period",        f"{start_year} - {end_year}"),
        ("Report generated",   report_dt),
        ("Latitude range",     f"{min(lats):.3f} - {max(lats):.3f} deg"),
        ("Longitude range",    f"{min(lons):.3f} - {max(lons):.3f} deg"),
        ("AEP calculations",   "Yes" if has_aep else "No"),
        ("Data source",        "ERA5 (Open-Meteo) + Global Wind Atlas (DTU)"),
    ]

    ax_meta = fig.add_axes([0, 0.63, 1, 0.145])
    ax_meta.set_facecolor(C_LIGHT)
    ax_meta.axis("off")
    for i, (lbl, val) in enumerate(meta_lines):
        col = i % 2
        row = i // 2
        x_lbl = 0.06 + col * 0.48
        x_val = x_lbl + 0.16
        y_pos = 0.88 - row * 0.22
        ax_meta.text(x_lbl, y_pos, lbl, transform=ax_meta.transAxes,
                     fontsize=8.5, color=C_CAPTION, va="top", fontweight="bold")
        ax_meta.text(x_val, y_pos, val, transform=ax_meta.transAxes,
                     fontsize=8.5, color=C_BODY, va="top")
    _rule(fig, 0.630)

    # ── Site list (bullet names — full data table follows as Wind Resource Summary page)
    y = 0.595
    fig.text(ML, y, "Sites assessed in this report:", fontsize=9,
             color=C_CAPTION, va="top", fontweight="bold")
    y -= 0.022
    for r in summary_rows:
        sn  = r.get("site_name", "")
        lat = r.get("latitude", "")
        lon = r.get("longitude", "")
        ws  = r.get("gwa_corrected_mean_hub_ms")
        ws_txt = f"  —  {ws:.2f} m/s" if ws is not None else ""
        fig.text(ML + 0.012, y,
                 f"•  {sn}  ({lat:.4f}°N, {lon:.4f}°E){ws_txt}",
                 fontsize=8.5, color=C_BODY, va="top")
        y -= 0.018
        if y < 0.10:
            break

    fig.text(ML, y - 0.010,
             "Full wind resource and AEP data tables are provided on the following pages.",
             fontsize=8, color=C_CAPTION, va="top", style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_map(pdf, summary_rows, page_num):
    fig = _fig()
    _page_chrome(fig, "Site Location Map", f"{page_num}")
    _header_band(fig, 0.952, 0.055,
                 "Site Location Map",
                 "GWA-corrected mean wind speed at hub height")

    ws_vals = np.array([r.get("gwa_corrected_mean_hub_ms", np.nan)
                        for r in summary_rows], dtype=float)
    lats    = np.array([r["latitude"]  for r in summary_rows], dtype=float)
    lons    = np.array([r["longitude"] for r in summary_rows], dtype=float)

    ax_map = fig.add_axes([0.05, 0.09, 0.90, 0.80])

    if _HAS_CTX:
        try:
            tf = _ProjT.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = tf.transform(lons, lats)
            x_pad = max((xs.max() - xs.min()) * 0.35, 30_000)
            y_pad = max((ys.max() - ys.min()) * 0.35, 30_000)
            ax_map.set_xlim(xs.min() - x_pad, xs.max() + x_pad)
            ax_map.set_ylim(ys.min() - y_pad, ys.max() + y_pad)
            ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery,
                            zoom="auto", attribution_size=5)
            norm   = Normalize(vmin=np.nanmin(ws_vals), vmax=np.nanmax(ws_vals))
            sc     = ax_map.scatter(xs, ys, c=ws_vals, cmap=WIND_CMAP,
                                    vmin=np.nanmin(ws_vals), vmax=np.nanmax(ws_vals),
                                    s=120, edgecolors="white", linewidths=1.2, zorder=5)
            for i, r in enumerate(summary_rows):
                ax_map.annotate(r.get("site_name", ""),
                                (xs[i], ys[i]),
                                textcoords="offset points", xytext=(6, 6),
                                fontsize=6, color="white", fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.15",
                                          fc="#00000088", ec="none"),
                                zorder=6)
            cbar = plt.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.01)
            cbar.set_label("GWA-corrected mean wind speed (m/s)")
            cbar.ax.tick_params(labelsize=7)
            ax_map.set_axis_off()
        except Exception:
            _map_fallback(ax_map, lats, lons, ws_vals, summary_rows)
    else:
        _map_fallback(ax_map, lats, lons, ws_vals, summary_rows)

    fig.text(0.5, 0.055, "Source: ESRI World Imagery © Esri, DigitalGlobe, GeoEye, Earthstar Geographics",
             ha="center", fontsize=6, color=C_CAPTION, style="italic")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _map_fallback(ax, lats, lons, ws_vals, summary_rows):
    ax.set_facecolor("#D4E6F1")
    sc = ax.scatter(lons, lats, c=ws_vals, cmap=WIND_CMAP, s=120,
                    edgecolors="white", linewidths=1.2, zorder=5)
    for i, r in enumerate(summary_rows):
        ax.annotate(r.get("site_name", ""), (lons[i], lats[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, alpha=0.4, color=C_BORDER)
    plt.colorbar(sc, ax=ax, label="GWA-corrected mean (m/s)", fraction=0.025)
    ax.set_title("Install contextily for satellite imagery", fontsize=8,
                 color=C_CAPTION, style="italic")


def _page_summary_table(pdf, summary_rows, page_counter):
    wind_cols = [
        ("site_name",                 "Site",           14),
        ("latitude",                  "Lat",             6),
        ("longitude",                 "Lon",             7),
        ("elevation_m_asl",           "Elev\n(m)",       5),
        ("hub_height_m",              "Hub\n(m)",        5),
        ("era5_mean_100m_ms",         "ERA5\n100m",      6),
        ("gwa_mean_100m_ms",          "GWA\n100m",       6),
        ("gwa_mean_hub_ms",           "GWA\nHub",        6),
        ("gwa_corrected_mean_hub_ms", "Corr.\nMean",     6),
        ("wind_shear_alpha",          "Shear\nalpha",    5),
        ("mean_air_density_kg_m3",    "Air dens.\n(kg/m3)", 7),
    ]
    _table_page(pdf, summary_rows, wind_cols,
                "Wind Resource Summary",
                "ERA5 100 m, GWA-calibrated hub-height mean, shear and air density",
                page_counter)
    page_counter[0] += 1

    has_aep = any("gross_aep_mwh_yr" in r for r in summary_rows)
    if has_aep:
        aep_cols = [
            ("site_name",              "Site",              16),
            ("turbine_type",           "WTG Model",         18),
            ("nameplate_mw",           "NP\n(MW)",           6),
            ("gross_aep_mwh_yr",       "Gross AEP\n(MWh/yr)",10),
            ("net_aep_mwh_yr",         "Net AEP\n(MWh/yr)", 10),
            ("mean_wake_loss_pct",     "Wake\n(%)",           5),
            ("capacity_factor_pct",    "CF\n(%)",             5),
            ("mean_air_density_kg_m3", "Air dens.\n(kg/m3)",  7),
        ]
        _table_page(pdf, summary_rows, aep_cols,
                    "AEP Results Summary",
                    "Gross/net annual energy, wake losses, capacity factor and air density",
                    page_counter)
        page_counter[0] += 1


def _table_page(pdf, rows, col_specs, title, subtitle, page_counter):
    """Render a data table onto one (or more) A4 page(s) in GHD style."""
    pg     = page_counter[0]
    fig    = _fig()
    _page_chrome(fig, title, str(pg))
    _header_band(fig, 0.952, 0.055, title, subtitle)

    y   = 0.875
    remaining = rows

    while remaining:
        y, remaining = _render_table(fig, remaining, col_specs, y, y_min=0.065)

        fig.text(ML, 0.052,
                 "All wind speeds in m/s  |  AEP values are mean annual  |  "
                 "CF = net AEP / (nameplate x 8760 h)",
                 fontsize=7, color=C_CAPTION, va="top", style="italic")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if remaining:
            pg += 1
            page_counter[0] = pg
            fig  = _fig()
            _page_chrome(fig, title + " (continued)", str(pg))
            _header_band(fig, 0.952, 0.055, title, subtitle + " (continued)")
            y = 0.875

    page_counter[0] = pg


def _page_site(pdf, summary_row, site_data, pc_df, page_num):
    """One page per site: key stats table + 4 charts."""
    sname    = summary_row.get("site_name", "Unknown")
    lat      = summary_row.get("latitude", 0)
    lon      = summary_row.get("longitude", 0)
    elev     = summary_row.get("elevation_m_asl", "-")
    hub      = summary_row.get("hub_height_m", "-")
    ws_corr  = site_data["ws_corr"].dropna()
    ws_raw   = site_data["ws_raw"].dropna()
    meta     = site_data["meta"]
    aep      = site_data.get("aep")
    wtg      = site_data.get("wtg", "")
    nameplate = site_data.get("nameplate_mw")

    fig = _fig()
    _page_chrome(fig, sname, str(page_num))

    # Header band: occupies 0.877–0.952 (below page chrome at 0.953)
    _header_band(fig, 0.952, 0.075, sname,
                 f"{lat:.4f}° N  {lon:.4f}° E  •  "
                 f"Elevation {elev} m ASL  •  Hub height {hub} m AGL")

    # ── Key stats panel — sits below header band (0.877), no overlap ────────
    stats = [
        ("ERA5 100 m mean",   f"{meta.get('mean_era5_100', 0):.2f} m/s"),
        ("GWA hub mean",      f"{meta.get('mean_gwa_150', 0):.2f} m/s"),
        ("Corrected mean",    f"{meta.get('mean_corrected', 0):.2f} m/s"),
        ("Shear alpha",       f"{meta.get('alpha_mean', 0):.3f}"),
        ("Air density",       f"{meta.get('mean_air_density', 1.225):.4f} kg/m3"),
    ]
    if aep:
        stats += [
            ("WTG",           str(wtg) if wtg else "-"),
            ("Nameplate",     f"{nameplate:.1f} MW" if nameplate else "-"),
            ("Gross AEP",     f"{aep.get('gross_aep_mwh', 0)/1000:.2f} GWh/yr"),
            ("Net AEP",       f"{aep.get('net_aep_mwh', 0)/1000:.2f} GWh/yr"),
            ("Wake loss",     f"{aep.get('mean_wake_pct', 0):.1f} %"),
            ("CF",            f"{aep.get('capacity_factor', 0)*100:.1f} %"),
        ]

    # Stats panel: 0.815–0.872 — clear gap below header (bottom=0.877)
    ax_stats = fig.add_axes([ML, 0.815, AVAIL, 0.057])
    ax_stats.set_facecolor(C_LIGHT)
    ax_stats.axis("off")
    n_cols = min(len(stats), 6)
    col_w  = 1.0 / n_cols
    for i, (lbl, val) in enumerate(stats[:n_cols * 2]):
        col = i % n_cols
        row = i // n_cols
        x   = 0.008 + col * col_w
        y   = 0.78  - row * 0.44
        ax_stats.text(x, y, lbl, transform=ax_stats.transAxes,
                      fontsize=7.5, color=C_CAPTION, va="top", fontweight="bold")
        ax_stats.text(x, y - 0.26, val, transform=ax_stats.transAxes,
                      fontsize=8.5, color=C_BODY, va="top")
    _rule(fig, 0.813)

    # ── 4 charts ──────────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.07, right=0.96,
                           top=0.805, bottom=0.07,
                           wspace=0.32, hspace=0.50)

    ax1 = fig.add_subplot(gs[0, 0])
    _plot_weibull(ax1, ws_corr, ws_raw, meta)

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_monthly_wind(ax2, ws_corr)

    ax3 = fig.add_subplot(gs[1, 0])
    _plot_diurnal(ax3, ws_corr)

    ax4 = fig.add_subplot(gs[1, 1])
    if aep and aep.get("gross_mw") is not None:
        _plot_monthly_aep(ax4, aep)
    elif pc_df is not None and wtg and wtg in pc_df.columns and nameplate:
        _plot_power_curve(ax4, pc_df, wtg, nameplate, ws_corr)
    elif site_data.get("air_density") is not None:
        rho     = site_data["air_density"].dropna()
        mon_rho = rho.groupby(rho.index.month).mean()
        months  = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
        x = np.arange(1, 13)
        ax4.bar(x, [mon_rho.get(m, np.nan) for m in x],
                color=C_SLATE, alpha=0.80, edgecolor="white", lw=0.5)
        ax4.axhline(1.225, color=C_RED, lw=1.2, ls="--",
                    label="1.225 kg/m3 (standard)")
        ax4.set_xticks(x)
        ax4.set_xticklabels(months, rotation=30)
        ax4.set_ylabel("Air density (kg/m3)")
        ax4.set_title("Monthly Mean Air Density", pad=4)
        ax4.legend(framealpha=0.85)
        _chart_style(ax4)
    else:
        ax4.text(0.5, 0.5, "No AEP data", transform=ax4.transAxes,
                 ha="center", va="center", fontsize=9, color=C_CAPTION)
        ax4.axis("off")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_wake_matrix(pdf, wake_df: pd.DataFrame, page_num):
    fig = _fig()
    _page_chrome(fig, "Wake Loss Matrix", str(page_num))
    _header_band(fig, 0.952, 0.055,
                 "Wake Loss Matrix",
                 "Percentage self-wake loss by wind speed and total park nameplate capacity")

    ax = fig.add_axes([0.10, 0.20, 0.78, 0.63])
    _plot_wake_matrix(ax, wake_df)

    note = (
        "The wake loss matrix provides a 2-D lookup of wind-farm self-wake losses "
        "as a function of wind speed (m/s) and total park nameplate capacity (MW). "
        "Losses are linearly interpolated between table nodes. Values at nameplate <= 8 MW "
        "are 0% (single turbine - no self-wake). The matrix is applied per timestep: "
        "for each hour, the actual wind speed and park nameplate are used to look up "
        "the wake loss fraction, which is then applied to reduce gross power output. "
        "This is an indicative simplification; actual wake losses depend on park geometry, "
        "turbine spacing, prevailing wind direction, and atmospheric stability."
    )
    _para(fig, 0.17, note, size=8.5)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pages_methodology(pdf, has_subhourly=False, has_density=True, page_counter=None):
    """Render methodology pages in GHD report style."""
    if page_counter is None:
        page_counter = [1]

    sections = [
        (1, "About This Assessment", ""),
        (0, "", (
            "This report presents the results of a synthesised wind resource assessment "
            "using two complementary data sources: ERA5 global atmospheric reanalysis for "
            "temporal variability, and the Global Wind Atlas (GWA) for spatial accuracy. "
            "The methodology combines the strengths of both datasets to produce a long-term "
            "hourly wind speed time series at hub height that is both temporally realistic "
            "and locally calibrated.\n"
            "\n"
            "Results are INDICATIVE ONLY. They are not a bankable wind resource assessment "
            "and should not be used as a substitute for on-site measurement campaigns."
        )),
        (1, "1.  Data Sources", ""),
        (2, "1.1  ERA5 Reanalysis (Temporal Backbone)", (
            "ERA5 is the ECMWF global atmospheric reanalysis covering 1940 to near-present "
            "at approximately 28 km horizontal resolution and 1-hour timesteps. It is accessed "
            "via the Open-Meteo archive API, returning wind speed at 10 m and 100 m above "
            "ground level, 2 m air temperature, and 10 m wind gusts. ERA5 captures the full "
            "temporal structure of the wind climate - inter-annual variability, seasonal cycles, "
            "storm events, and diurnal patterns - but its coarse resolution misses local terrain "
            "channelling, coastal effects, and sub-kilometre roughness changes."
        )),
        (2, "1.2  Global Wind Atlas (Spatial Calibration)", (
            "The Global Wind Atlas (GWA) is produced by the Technical University of Denmark "
            "(DTU) using WAsP mesoscale modelling driven by ERA5, downscaled to a 250 m grid. "
            "At each grid point, GWA provides Weibull scale (A) and shape (k) parameters at "
            "multiple heights (50, 100, 150, 200 m) for 12 wind direction sectors. These "
            "parameters encode local terrain, roughness, and coastal effects at much finer "
            "resolution than ERA5. GWA has no temporal dimension of its own."
        )),
        (1, "2.  Processing Pipeline", ""),
        (2, "2.1  Height Extrapolation (100 m to Hub Height)", (
            "ERA5 provides wind at 100 m. To reach the specified hub height, a power-law "
            "extrapolation is applied with a diurnal shear exponent alpha:\n"
            "\n"
            "    V_hub(t) = V_100(t) x (h_hub / 100)^alpha(h)\n"
            "\n"
            "The shear exponent is not a single constant - it varies by hour of day to "
            "capture the diurnal stability cycle. The magnitude is anchored to GWA by "
            "computing the ratio of GWA mean wind speeds at 100 m and hub height. "
            "The diurnal shape is derived from the ERA5 10 m / 100 m wind ratio; the "
            "24-hour profile is normalised so its mean equals the GWA-calibrated alpha_mean."
        )),
        (2, "2.2  Weibull Quantile Transform (GWA Bias Correction)", (
            "After height extrapolation the ERA5-derived wind speed distribution at hub "
            "height still differs from GWA due to ERA5's spatial resolution bias. A Weibull "
            "quantile transform re-shapes the ERA5 distribution to match GWA's "
            "locally-calibrated Weibull:\n"
            "\n"
            "    V*(t) = A_GWA x (V_hub(t) / A_ERA5)^(k_ERA5 / k_GWA)\n"
            "\n"
            "The transform is rank-preserving: the hour-by-hour sequence, storm timing, "
            "and seasonal patterns are all unchanged - only the speed distribution is "
            "reshaped to match GWA. Roughness class selection uses OpenStreetMap within "
            "500 m of the site to avoid the large positive bias (~25%) from inadvertently "
            "using sea-surface roughness for a land site."
        )),
        (1, "3.  Air Density Correction (IEC Method)" if has_density else "", ""),
        (0, "", (
            "Standard power curves are published at ISO sea-level conditions "
            "(rho0 = 1.225 kg/m3). Sites at elevation or in warm climates have lower "
            "actual air density, reducing the kinetic energy available to the rotor. "
            "The IEC method corrects for this by computing an equivalent wind speed at "
            "standard density before the power curve lookup:\n"
            "\n"
            "    V_eq = V_actual x (rho_site / 1.225)^(1/3)\n"
            "\n"
            "Air density at hub height is computed per timestep from ERA5 2 m temperature, "
            "site elevation (from Open-Meteo), and standard ISA lapse rate (-6.5 K/km):\n"
            "\n"
            "    T_hub = T_2m + 273.15 - 0.0065 x (h_hub - 2)   [K]\n"
            "    P_hub = 101325 x (1 - 2.2558e-5 x (h_elev + h_hub))^5.2559   [Pa]\n"
            "    rho = P_hub / (287.05 x T_hub)   [kg/m3]\n"
            "\n"
            "The density correction affects AEP (through the equivalent wind speed and "
            "power curve lookup) but not the wind speed time series itself."
        ) if has_density else "Air density correction was not applied in this assessment."),
        (1, "4.  Wake Loss Model", ""),
        (0, "", (
            "Farm-level wake losses are estimated using a 2-D lookup matrix of percentage "
            "wake loss as a function of wind speed (m/s) and total park nameplate capacity "
            "(MW). The matrix is applied per timestep using bilinear interpolation:\n"
            "\n"
            "    wake_loss%(V, NP) = interpolate(wake_matrix, [V, NP])\n"
            "    net_power = gross_power x (1 - wake_loss% / 100)\n"
            "\n"
            "Losses are zero for nameplate <= 8 MW (single turbine - no self-wake), "
            "and increase with farm size. The nameplate used for wake lookup must be the "
            "TOTAL PARK CAPACITY in MW, not the individual turbine rating. See the Wake "
            "Loss Matrix page for the full lookup table."
        )),
    ]

    if has_subhourly:
        sections += [
            (1, "5.  Sub-hourly Disaggregation", ""),
            (0, "", (
                "When 30-min or 10-min output is selected, hourly ERA5+GWA values are "
                "stochastically disaggregated using an AR(1) process (Ornstein-Uhlenbeck). "
                "Per-hour turbulence intensity is estimated from the ERA5 gust factor at "
                "10 m, scaled to hub height using a power-law profile (exponent 0.11). "
                "Each hourly block is mean-corrected so that sub-hourly values average "
                "exactly to the ERA5+GWA hourly value. Sub-hourly output is clearly "
                "labelled as synthetic and should not be used for fatigue-load analysis."
            )),
        ]

    sections += [
        (1, "6.  Limitations", ""),
        (0, "", (
            "Not a measured record. Output is synthesised from modelled data. "
            "Uncertainty is greater than for a site-specific mast measurement campaign.\n"
            "\n"
            "ERA5 grid spacing approximately 28 km. Local terrain effects within this "
            "radius are captured only partially - GWA improves spatial accuracy but cannot "
            "fully replace high-resolution mesoscale modelling or on-site measurement.\n"
            "\n"
            "GWA is a long-term climatology. Year-to-year variation in the output comes "
            "entirely from ERA5.\n"
            "\n"
            "Diurnal shear uses the 10-100 m ERA5 layer as a proxy for the 100 m - hub "
            "height layer. At hub heights well above 100 m the actual diurnal shear profile "
            "may differ.\n"
            "\n"
            "Wake model is simplified. The 2-D matrix approach does not account for "
            "directional variation in wake losses, park layout geometry, or atmospheric "
            "stability effects on wake recovery.\n"
            "\n"
            "Onshore use only. GWA terrain modelling and the roughness class selection are "
            "not designed for offshore environments.\n"
            "\n"
            "Results are indicative and should inform - not replace - site-specific "
            "measurement campaigns and bankable wind resource assessments."
        )),
    ]

    pg  = page_counter[0]
    fig = _fig()
    _page_chrome(fig, "Methodology", str(pg))
    _header_band(fig, 0.952, 0.055, "Methodology", "How results are derived")
    y = 0.873

    for (level, title, content) in sections:
        if y < 0.10:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            pg += 1
            page_counter[0] = pg
            fig = _fig()
            _page_chrome(fig, "Methodology (continued)", str(pg))
            _header_band(fig, 0.952, 0.055,
                         "Methodology (continued)", "How results are derived")
            y = 0.873

        if title:
            y = _section_title(fig, y, title, level=level)
            y -= 0.004
        if content:
            y = _para(fig, y, content, size=9.0)
            y -= 0.006

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_pdf_report(
    summary_rows: list[dict],
    site_data: dict,
    wake_df,
    pc_df,
    start_year: int,
    end_year: int,
) -> bytes:
    """
    Generate a multi-page PDF QA report styled after GHD reference document.

    Args:
        summary_rows: list of per-site summary dicts (from batch processing)
        site_data:    dict keyed by site_name with keys:
                        ws_corr, ws_raw, ws_100m, air_density, meta, aep,
                        wtg, nameplate_mw, lat, lon, elevation
        wake_df:      wake loss DataFrame (or None)
        pc_df:        power curve DataFrame (or None)
        start_year / end_year: ERA5 period
    Returns:
        PDF bytes
    """
    report_dt    = datetime.now().strftime("%d %B %Y  %H:%M")
    has_density  = any("mean_air_density_kg_m3" in r for r in summary_rows)
    has_subhourly = False   # batch mode uses hourly pipeline

    # Shared page counter so page numbers run across all pages
    pc = [1]

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        d = pdf.infodict()
        d["Title"]        = "ERA5 x GWA Wind Resource Assessment Report"
        d["Author"]       = "ERA5 x GWA Wind Tool"
        d["Subject"]      = f"Batch assessment - {len(summary_rows)} sites, {start_year}-{end_year}"
        d["CreationDate"] = datetime.now()

        _page_cover(pdf, summary_rows, start_year, end_year, report_dt, pc[0]); pc[0] += 1
        _page_map(pdf, summary_rows, pc[0]); pc[0] += 1
        _page_summary_table(pdf, summary_rows, pc)

        for row in summary_rows:
            sname = row.get("site_name", "")
            if sname in site_data:
                pc[0] += 1
                _page_site(pdf, row, site_data[sname], pc_df=pc_df, page_num=pc[0])

        if wake_df is not None:
            pc[0] += 1
            _page_wake_matrix(pdf, wake_df, pc[0])

        pc[0] += 1
        _pages_methodology(pdf, has_subhourly=has_subhourly,
                           has_density=has_density, page_counter=pc)

    return buf.getvalue()
