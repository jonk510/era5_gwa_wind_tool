"""
PDF Wind Resource Assessment Report Generator
ERA5 × Global Wind Atlas Wind Tool

Report structure (engineering standard):
  Cover Page
  Executive Summary
  1. Introduction
  2. Methodology
  3. Site Location
  4. Wind Resource Results  (per-site pages)
  5. AEP Results           (if turbine data supplied)
  6. Conclusions and Limitations
  Appendix A – Wake Loss Matrix
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, Normalize
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
C_BLACK    = "#000000"
C_BODY     = "#1A1A1A"
C_BORDER   = "#AAAAAA"
C_CAPTION  = "#404040"
C_WHITE    = "#FFFFFF"
C_LIGHT    = "#F4F4F4"
C_MIDGREY  = "#888888"
C_NAVY     = "#1B3A6B"
C_SLATE    = "#4A6080"
C_AMBER    = "#B45309"
C_RED      = "#B91C1C"
C_TEAL     = "#0F766E"

A4         = (8.27, 11.69)
ML         = 0.095
MR         = 0.905
AVAIL      = MR - ML

WIND_CMAP  = LinearSegmentedColormap.from_list(
    "wind", ["#DC2626", "#FBBF24", "#16A34A"], N=256
)


# ── Primitive helpers ─────────────────────────────────────────────────────────

def _fig():
    return plt.figure(figsize=A4, facecolor="white")


def _rule(fig, y, color=C_BORDER, x0=None, width=None, lw=0.0008):
    x0    = x0    if x0    is not None else 0.0
    width = width if width is not None else 1.0
    ax = fig.add_axes([x0, y, width, lw])
    ax.set_facecolor(color)
    ax.axis("off")


def _content_rule(fig, y):
    """Thin rule spanning the content column only."""
    _rule(fig, y, color=C_BORDER, x0=ML, width=AVAIL)


def _page_chrome(fig, section="", page_label=""):
    """Consistent running header + footer on every page."""
    # Header
    _rule(fig, 0.958, color=C_MIDGREY)
    left  = "ERA5 × GWA Wind Resource Assessment"
    right = section
    fig.text(ML,  0.964, left,  fontsize=7,   color=C_MIDGREY, va="bottom")
    fig.text(MR,  0.964, right, fontsize=7,   color=C_MIDGREY, va="bottom", ha="right")
    _rule(fig, 0.954, color=C_MIDGREY)

    # Footer
    _rule(fig, 0.042, color=C_MIDGREY)
    fig.text(ML, 0.037,
             "INDICATIVE ONLY — Results synthesised from ERA5 reanalysis and Global Wind Atlas. "
             "Not a bankable wind resource assessment. Not a substitute for site measurement.",
             fontsize=6.5, color=C_MIDGREY, va="top", style="italic")
    fig.text(MR, 0.037, f"Page {page_label}", fontsize=7, color=C_MIDGREY,
             va="top", ha="right")


def _section_band(fig, y_top, height, title, subtitle="", number=""):
    """Dark grey section-title band (not full black — softer for body pages)."""
    ax = fig.add_axes([0, y_top - height, 1, height])
    ax.set_facecolor("#1F2937")
    ax.axis("off")
    tx = ML
    t  = f"{number}  {title}" if number else title
    ax.text(tx, 0.65, t, transform=ax.transAxes,
            color=C_WHITE, fontsize=13, fontweight="bold", va="center")
    if subtitle:
        ax.text(tx, 0.22, subtitle, transform=ax.transAxes,
                color="#CBD5E1", fontsize=8.5, va="center", style="italic")
    return y_top - height


def _h1(fig, y, text, number=""):
    """Level-1 section heading with rule below."""
    label = f"{number}  {text}" if number else text
    fig.text(ML, y, label, fontsize=13, color=C_BLACK, fontweight="bold", va="top")
    y -= 0.024
    _content_rule(fig, y + 0.005)
    return y - 0.012


def _h2(fig, y, text):
    fig.text(ML, y, text, fontsize=10.5, color=C_NAVY, fontweight="bold", va="top")
    return y - 0.020


def _para(fig, y, text, width=94, size=9.0, indent=0):
    """Wrap and render a paragraph. Returns new y."""
    x = ML + indent
    for raw in text.split("\n"):
        for line in textwrap.wrap(raw, width - int(indent * 130)) or [""]:
            if y < 0.055:
                return y
            fig.text(x, y, line, fontsize=size, color=C_BODY, va="top")
            y -= 0.0155
        y -= 0.005
    return y


def _bullet(fig, y, text, width=90):
    lines = textwrap.wrap(text, width)
    if not lines:
        return y
    fig.text(ML + 0.010, y, "•", fontsize=9, color=C_BODY, va="top")
    fig.text(ML + 0.022, y, lines[0], fontsize=9, color=C_BODY, va="top")
    y -= 0.0155
    for cont in lines[1:]:
        fig.text(ML + 0.022, y, cont, fontsize=9, color=C_BODY, va="top")
        y -= 0.0155
    return y


def _eq(fig, y, equation, note=""):
    """Display a centred equation block."""
    fig.text(0.5, y, equation, fontsize=9.5, color=C_NAVY,
             va="top", ha="center", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF", ec="#BFDBFE", lw=0.8))
    y -= 0.032
    if note:
        fig.text(ML + 0.015, y, f"where  {note}", fontsize=8, color=C_MIDGREY, va="top")
        y -= 0.018
    return y - 0.006


# ── Table renderer ────────────────────────────────────────────────────────────

def _render_table(fig, rows, col_specs, y_start, caption="", y_min=0.08):
    """
    Render a styled data table.
    col_specs: list of (key, header_label, relative_width)
    Returns (final_y, overflow_rows).
    """
    keys    = [c[0] for c in col_specs]
    labels  = [c[1] for c in col_specs]
    rel_w   = [c[2] for c in col_specs]
    total   = sum(rel_w)
    col_w   = [w / total * AVAIL for w in rel_w]

    HDR_H   = 0.032
    ROW_H   = 0.026

    y = y_start

    if caption:
        fig.text(ML, y, caption, fontsize=8, color=C_CAPTION,
                 va="top", fontweight="bold", style="italic")
        y -= 0.020

    # Header row
    ax_h = fig.add_axes([ML, y - HDR_H, AVAIL, HDR_H])
    ax_h.set_facecolor("#1F2937")
    ax_h.axis("off")
    xf = 0.0
    for lbl, cw in zip(labels, col_w):
        for li, line in enumerate(lbl.split("\n")):
            ax_h.text(xf / AVAIL + 0.008, 0.72 - li * 0.36, line,
                      transform=ax_h.transAxes,
                      fontsize=8, color=C_WHITE, fontweight="bold", va="top")
        xf += cw
    y -= HDR_H + 0.001

    # Rule above header
    _rule(fig, y_start if not caption else y_start - 0.022,
          color="#1F2937", x0=ML, width=AVAIL, lw=HDR_H)

    overflow = []
    for ri, row in enumerate(rows):
        if y < y_min + ROW_H:
            overflow = rows[ri:]
            break
        bg = C_WHITE if ri % 2 == 0 else "#F8F9FA"
        ax_r = fig.add_axes([ML, y - ROW_H, AVAIL, ROW_H])
        ax_r.set_facecolor(bg)
        ax_r.axis("off")
        xf = 0.0
        for key, cw in zip(keys, col_w):
            txt = _fmt(row.get(key), key)
            bold = key == "site_name"
            ax_r.text(xf / AVAIL + 0.008, 0.5, txt,
                      transform=ax_r.transAxes, fontsize=8,
                      color=C_BODY, va="center",
                      fontweight="bold" if bold else "normal", clip_on=True)
            xf += cw
        y -= ROW_H
        _rule(fig, y, color=C_BORDER, x0=ML, width=AVAIL)
        y -= 0.001

    _rule(fig, y_start if not caption else y_start - 0.020,
          color=C_BORDER, x0=ML, width=AVAIL)

    return y, overflow


def _fmt(val, key):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "–"
    if key in ("latitude", "longitude"):
        return f"{float(val):.4f}"
    if key in ("era5_mean_100m_ms", "gwa_mean_100m_ms",
               "gwa_mean_hub_ms", "gwa_corrected_mean_hub_ms"):
        return f"{float(val):.2f} m/s"
    if key == "wind_shear_alpha":
        return f"{float(val):.3f}"
    if key == "mean_air_density_kg_m3":
        return f"{float(val):.4f} kg/m³"
    if key in ("gross_aep_mwh_yr", "net_aep_mwh_yr"):
        v = float(val)
        return f"{v/1000:.2f} GWh/yr" if v >= 1000 else f"{v:.0f} MWh/yr"
    if key == "nameplate_mw":
        return f"{float(val):.1f} MW"
    if key in ("mean_wake_loss_pct", "capacity_factor_pct"):
        return f"{float(val):.1f} %"
    if key in ("hub_height_m", "elevation_m_asl"):
        try:
            return f"{int(val)} m"
        except Exception:
            return str(val)
    return str(val)


# ── Chart helpers (all include site name in title) ────────────────────────────

def _chart_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.22, lw=0.5, color="#DDDDDD")
    ax.tick_params(labelsize=7.5)


def _plot_weibull(ax, ws_corr, ws_raw, meta, sname=""):
    ws_c = ws_corr.dropna().values
    ws_r = ws_raw.dropna().values
    ws_c = ws_c[ws_c > 0.1]; ws_r = ws_r[ws_r > 0.1]
    v = np.linspace(0, max(ws_c.max(), ws_r.max()) * 1.12, 300)

    def _fit_plot(ws, color, ls, prefix):
        k, _, A = weibull_min.fit(ws, floc=0)
        ax.hist(ws, bins=40, density=True, alpha=0.18, color=color, edgecolor="none")
        ax.plot(v, weibull_min.pdf(v, k, 0, A), color=color, lw=1.8, ls=ls,
                label=f"{prefix}  A={A:.2f}, k={k:.2f}")
        return A, k

    _fit_plot(ws_r, C_SLATE, "--", "ERA5 extrap.")
    A_c, k_c = _fit_plot(ws_c, C_NAVY, "-",  "GWA-corrected")

    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability density")
    title = "Wind Speed Distribution" if not sname else f"{sname} — Wind Speed Distribution"
    ax.set_title(title, pad=4, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(left=0)
    ax.text(0.97, 0.97, f"A = {A_c:.2f} m/s\nk = {k_c:.2f}",
            transform=ax.transAxes, fontsize=7.5, va="top", ha="right",
            color=C_NAVY,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85,
                      ec=C_BORDER, lw=0.5))
    _chart_style(ax)


def _plot_monthly_wind(ax, ws_corr, sname=""):
    monthly = ws_corr.resample("ME").mean()
    mon_m   = monthly.groupby(monthly.index.month).mean()
    MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    x    = np.arange(1, 13)
    vals = [mon_m.get(m, np.nan) for m in x]
    bars = ax.bar(x, vals, color=C_NAVY, alpha=0.82, width=0.7, edgecolor="white", lw=0.4)
    ax.axhline(ws_corr.mean(), color=C_AMBER, lw=1.3, ls="--",
               label=f"Annual mean  {ws_corr.mean():.2f} m/s")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, rotation=35, ha="right")
    ax.set_ylabel("Mean wind speed (m/s)")
    title = "Monthly Mean Wind Speed" if not sname else f"{sname} — Monthly Mean Wind Speed"
    ax.set_title(title, pad=4, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)
    for b in bars:
        h = b.get_height()
        if not np.isnan(h):
            ax.text(b.get_x() + b.get_width()/2, h + 0.04,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=6.5)
    _chart_style(ax)


def _plot_diurnal(ax, ws_corr, sname=""):
    diurnal = ws_corr.groupby(ws_corr.index.hour).mean()
    hrs  = np.arange(24)
    vals = [diurnal.get(h, np.nan) for h in hrs]
    ax.plot(hrs, vals, color=C_NAVY, lw=2, marker="o", ms=3)
    ax.fill_between(hrs, vals, alpha=0.10, color=C_NAVY)
    ax.axhline(np.nanmean(vals), color=C_AMBER, lw=1, ls="--", label="Daily mean")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)], rotation=35, ha="right")
    ax.set_ylabel("Mean wind speed (m/s)")
    ax.set_xlabel("Hour of day (local time)")
    title = "Diurnal Wind Profile" if not sname else f"{sname} — Diurnal Profile"
    ax.set_title(title, pad=4, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)
    _chart_style(ax)


def _plot_monthly_aep(ax, aep, sname=""):
    gross  = aep["gross_mw"].resample("ME").sum() / 1000
    net    = aep["net_mw"].resample("ME").sum()   / 1000
    mon_g  = gross.groupby(gross.index.month).mean()
    mon_n  = net.groupby(net.index.month).mean()
    MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(1, 13)
    ax.bar(x, [mon_g.get(m, 0) for m in x], color=C_SLATE, alpha=0.55,
           width=0.7, label="Gross AEP", edgecolor="white", lw=0.4)
    ax.bar(x, [mon_n.get(m, 0) for m in x], color=C_NAVY, alpha=0.85,
           width=0.7, label="Net AEP",   edgecolor="white", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, rotation=35, ha="right")
    ax.set_ylabel("Mean monthly energy (GWh)")
    title = "Monthly AEP (mean annual)" if not sname else f"{sname} — Monthly AEP"
    ax.set_title(title, pad=4, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)
    _chart_style(ax)


def _plot_power_curve(ax, pc_df, wtg, nameplate_mw, ws_corr, sname=""):
    ws_pc   = pc_df.index.values
    kw_pc   = pc_df[wtg].values
    rated   = kw_pc.max()
    mw_pc   = kw_pc / rated * nameplate_mw
    ax2 = ax.twinx()
    ax2.hist(ws_corr.dropna().values, bins=40, color=C_SLATE, alpha=0.13,
             density=True, zorder=0)
    ax2.set_ylabel("Probability density", color=C_SLATE, fontsize=7.5)
    ax2.tick_params(colors=C_SLATE, labelsize=7)
    ax2.spines[["top", "right"]].set_color(C_SLATE)
    ax.plot(ws_pc, mw_pc, color=C_NAVY, lw=2, zorder=5, label=wtg)
    ax.fill_between(ws_pc, mw_pc, alpha=0.10, color=C_NAVY, zorder=4)
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Power (MW)", color=C_NAVY)
    ax.tick_params(axis="y", colors=C_NAVY)
    ax.spines["left"].set_color(C_NAVY)
    ax.spines["top"].set_visible(False)
    ax.grid(True, alpha=0.22, lw=0.5, color="#DDDDDD", zorder=1)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    title = "Power Curve" if not sname else f"{sname} — Power Curve"
    ax.set_title(title, pad=4, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)


def _plot_air_density(ax, air_density, sname=""):
    rho     = air_density.dropna()
    mon_rho = rho.groupby(rho.index.month).mean()
    MONTHS  = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(1, 13)
    ax.bar(x, [mon_rho.get(m, np.nan) for m in x],
           color=C_TEAL, alpha=0.82, width=0.7, edgecolor="white", lw=0.4)
    ax.axhline(1.225, color=C_RED, lw=1.2, ls="--", label="1.225 kg/m³ (ISA std)")
    ax.axhline(float(rho.mean()), color=C_AMBER, lw=1.2, ls=":",
               label=f"Site mean  {rho.mean():.4f} kg/m³")
    ax.set_xticks(x); ax.set_xticklabels(MONTHS, rotation=35, ha="right")
    ax.set_ylabel("Air density (kg/m³)")
    title = "Monthly Air Density" if not sname else f"{sname} — Monthly Air Density"
    ax.set_title(title, pad=4, fontsize=8.5, fontweight="bold")
    ax.legend(fontsize=7)
    _chart_style(ax)


# ── Page builders ─────────────────────────────────────────────────────────────

def _page_cover(pdf, summary_rows, start_year, end_year, report_dt, page_num):
    fig = _fig()

    # Full-width black title strip
    ax_t = fig.add_axes([0, 0.77, 1, 0.23])
    ax_t.set_facecolor("#0F172A")
    ax_t.axis("off")
    ax_t.text(0.5, 0.78, "Wind Resource Assessment Report",
              transform=ax_t.transAxes, ha="center", va="center",
              color=C_WHITE, fontsize=20, fontweight="bold")
    ax_t.text(0.5, 0.52, "ERA5 Reanalysis  ×  Global Wind Atlas",
              transform=ax_t.transAxes, ha="center", va="center",
              color="#94A3B8", fontsize=12)
    ax_t.text(0.5, 0.28, "Synthetic Wind Data and Energy Production Estimate",
              transform=ax_t.transAxes, ha="center", va="center",
              color="#CBD5E1", fontsize=10, style="italic")

    # Accent rule
    _rule(fig, 0.768, color="#3B82F6", lw=0.005)

    # Metadata grid
    n       = len(summary_rows)
    has_aep = any("gross_aep_mwh_yr" in r for r in summary_rows)
    lats    = [r["latitude"]  for r in summary_rows]
    lons    = [r["longitude"] for r in summary_rows]

    meta = [
        ("Number of sites",    str(n)),
        ("ERA5 period",        f"{start_year} – {end_year}"),
        ("Date of report",     report_dt),
        ("Latitude range",     f"{min(lats):.3f}° – {max(lats):.3f}°"),
        ("Longitude range",    f"{min(lons):.3f}° – {max(lons):.3f}°"),
        ("AEP calculated",     "Yes" if has_aep else "No"),
        ("ERA5 data source",   "Open-Meteo Archive API"),
        ("GWA data source",    "Global Wind Atlas v3 (DTU)"),
    ]

    ax_m = fig.add_axes([0, 0.58, 1, 0.168])
    ax_m.set_facecolor(C_LIGHT)
    ax_m.axis("off")
    for i, (lbl, val) in enumerate(meta):
        col = i % 2
        row = i // 2
        xL  = 0.06 + col * 0.48
        xV  = xL + 0.18
        yp  = 0.90 - row * 0.23
        ax_m.text(xL, yp, lbl, transform=ax_m.transAxes,
                  fontsize=8.5, color=C_CAPTION, va="top", fontweight="bold")
        ax_m.text(xV, yp, val, transform=ax_m.transAxes,
                  fontsize=8.5, color=C_BODY, va="top")

    _rule(fig, 0.578, color=C_BORDER)

    # Site list
    y = 0.555
    fig.text(ML, y, "Sites included in this assessment:",
             fontsize=9.5, color=C_BLACK, fontweight="bold", va="top")
    y -= 0.028
    for r in summary_rows:
        sn  = r.get("site_name", "")
        la  = r.get("latitude", "")
        lo  = r.get("longitude", "")
        ws  = r.get("gwa_corrected_mean_hub_ms")
        ws_t = f"   GWA-corrected mean:  {ws:.2f} m/s" if ws is not None else ""
        fig.text(ML + 0.012, y,
                 f"•   {sn}   ({float(la):.4f}°N, {float(lo):.4f}°E){ws_t}",
                 fontsize=9, color=C_BODY, va="top")
        y -= 0.020
        if y < 0.12:
            break

    _rule(fig, 0.10, color=C_BORDER)
    fig.text(0.5, 0.075,
             "INDICATIVE ONLY — Results are synthesised from ERA5 reanalysis and Global Wind Atlas. "
             "This report does not constitute a bankable wind resource assessment.",
             fontsize=7.5, color=C_MIDGREY, ha="center", va="top", style="italic")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _page_executive_summary(pdf, summary_rows, start_year, end_year, page_num):
    fig = _fig()
    _page_chrome(fig, "Executive Summary", str(page_num))
    y = _section_band(fig, 0.952, 0.065,
                      "Executive Summary",
                      f"Key findings — {len(summary_rows)} sites, ERA5 {start_year}–{end_year}")

    y -= 0.018
    y = _h1(fig, y, "Purpose and Scope")
    y -= 0.006
    y = _para(fig, y,
        "This report presents the results of a synthesised wind resource and indicative energy "
        "production assessment for the sites listed below. The analysis combines ERA5 global "
        "atmospheric reanalysis data with spatially calibrated wind statistics from the Global "
        "Wind Atlas (GWA) to produce long-term hourly wind speed time series at the specified "
        "hub heights. Where turbine data were provided, indicative Annual Energy Production (AEP) "
        "estimates including simplified wake losses have been calculated."
    )

    y -= 0.010
    y = _h1(fig, y, "Key Results")
    y -= 0.006

    # Summary table — wind stats
    wind_cols = [
        ("site_name",                "Site",           14),
        ("hub_height_m",             "Hub\n(m)",        5),
        ("elevation_m_asl",          "Elev\n(m ASL)",   6),
        ("gwa_corrected_mean_hub_ms","Corrected\nMean", 9),
        ("wind_shear_alpha",         "Shear\nAlpha",    6),
        ("mean_air_density_kg_m3",   "Air Density\n(kg/m³)", 8),
    ]
    has_aep = any("gross_aep_mwh_yr" in r for r in summary_rows)
    if has_aep:
        wind_cols += [
            ("turbine_type",         "Turbine",        12),
            ("nameplate_mw",         "Nameplate",       7),
            ("net_aep_mwh_yr",       "Net AEP\n(MWh/yr)", 9),
            ("capacity_factor_pct",  "CF\n(%)",          5),
        ]

    y, _ = _render_table(fig, summary_rows, wind_cols, y,
                         caption="Table 1  Summary of key wind resource and AEP metrics")

    y -= 0.015
    y = _h2(fig, y, "Interpretation Notes")
    y -= 0.005
    notes = [
        "GWA-corrected mean wind speed accounts for local terrain, roughness and coastal effects "
        "via the Global Wind Atlas 250 m grid. ERA5 alone typically over-estimates wind speeds "
        "at land sites due to its coarse (~28 km) grid resolution.",
        "Wind shear alpha (α) describes the rate of increase in mean wind speed with height "
        "above ground, fitted between ERA5 100 m and the hub height using a power-law profile. "
        "Typical values: 0.10–0.15 (open flat terrain), 0.20–0.30 (complex terrain).",
        "Air density below 1.225 kg/m³ (ISA standard) reduces the kinetic energy available "
        "to the rotor. AEP estimates incorporate the IEC 61400-12-1 equivalent wind speed "
        "correction for site air density.",
        "AEP values are gross-to-net after applying simplified self-wake losses from a 2-D "
        "lookup matrix. Actual losses depend on park layout, turbine spacing, and prevailing "
        "wind direction — site-specific micrositing analysis is recommended.",
    ]
    for note in notes:
        y = _bullet(fig, y, note)
        y -= 0.004

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _pages_introduction(pdf, start_year, end_year, page_counter):
    pg  = page_counter[0]
    fig = _fig()
    _page_chrome(fig, "1. Introduction", str(pg))
    y   = _section_band(fig, 0.952, 0.065,
                        "1.  Introduction",
                        "Background, objectives and scope of assessment")

    y -= 0.018
    y = _h1(fig, y, "Background", number="1.1")
    y -= 0.005
    y = _para(fig, y,
        "Wind resource assessment is a critical step in the development of wind energy projects. "
        "Accurate knowledge of the long-term wind climate at a site — including mean wind speed, "
        "wind speed distribution, seasonal variability, and wind shear — is essential for "
        "estimating energy production, selecting appropriate turbines, and assessing project "
        "financial viability."
    )
    y = _para(fig, y,
        "Traditional wind resource assessments rely on on-site measurement campaigns using "
        "meteorological masts or remote sensing instruments (LiDAR, SoDAR) over a minimum "
        "period of one year, correlated with long-term reference data to produce a "
        "Measure–Correlate–Predict (MCP) estimate. Such campaigns are costly and time-consuming."
    )
    y = _para(fig, y,
        "This assessment uses a desktop synthesis methodology combining two globally available "
        "data sources — ERA5 reanalysis (temporal variability) and the Global Wind Atlas "
        "(spatial accuracy) — to produce a long-term synthetic wind speed time series without "
        "the need for site measurement. The approach is suitable for early-stage screening, "
        "feasibility studies, and indicative energy estimates."
    )

    y -= 0.010
    y = _h1(fig, y, "Objectives", number="1.2")
    y -= 0.005
    objectives = [
        f"Derive a long-term ({start_year}–{end_year}) synthetic hourly wind speed time series "
        "at hub height for each site, calibrated to the Global Wind Atlas 250 m grid.",
        "Characterise the wind climate at each site, including Weibull distribution parameters, "
        "monthly variability, diurnal profile, wind shear, and air density.",
        "Calculate indicative Annual Energy Production (AEP) estimates where turbine "
        "specifications were provided, including simplified wake loss correction.",
        "Identify key uncertainties and limitations associated with the synthetic approach.",
    ]
    for obj in objectives:
        y = _bullet(fig, y, obj)
        y -= 0.004

    y -= 0.010
    y = _h1(fig, y, "Scope and Limitations", number="1.3")
    y -= 0.005
    y = _para(fig, y,
        "This assessment is indicative only. Results are derived from modelled data sources "
        "and have not been validated against on-site measurements. The methodology is "
        "appropriate for early-stage screening and feasibility studies but should not be used "
        "as the sole basis for investment decisions or project financing. A bankable wind "
        "resource assessment requires a minimum 12-month on-site measurement campaign "
        "following IEC 61400-12-1 and MEASNET procedures."
    )
    y = _para(fig, y,
        "The ERA5 × GWA synthesis is designed for onshore sites. The roughness class selection "
        "and GWA terrain modelling assumptions are not appropriate for offshore environments "
        "or complex coastal locations where sea–land transitions dominate the flow."
    )

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    page_counter[0] = pg + 1


def _pages_methodology(pdf, has_density, has_aep, has_subhourly, page_counter):
    def _new_page(section_title, subtitle=""):
        nonlocal fig, y
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        page_counter[0] += 1
        fig = _fig()
        _page_chrome(fig, "2. Methodology", str(page_counter[0]))
        y = _section_band(fig, 0.952, 0.065, section_title, subtitle)
        y -= 0.018

    fig = _fig()
    _page_chrome(fig, "2. Methodology", str(page_counter[0]))
    y = _section_band(fig, 0.952, 0.065,
                      "2.  Methodology",
                      "Data sources, processing pipeline, and calculation methods")

    y -= 0.018
    y = _h1(fig, y, "Overview", number="2.1")
    y -= 0.005
    y = _para(fig, y,
        "The synthesis methodology combines two complementary global datasets: ERA5 reanalysis "
        "provides the temporal variability of the wind climate (inter-annual variation, seasonal "
        "cycles, storm events, diurnal patterns), while the Global Wind Atlas provides spatially "
        "accurate long-term mean wind statistics calibrated to local terrain at 250 m resolution. "
        "The five-step processing pipeline is summarised below."
    )

    # Pipeline steps summary
    steps = [
        ("Step 1 – ERA5 Data Retrieval",
         "Hourly wind speed at 100 m, 10 m, temperature at 2 m, and gusts are retrieved from "
         "the Open-Meteo ERA5 archive API for the assessment period."),
        ("Step 2 – GWA Statistics Retrieval",
         "Weibull A and k parameters at multiple heights (50, 100, 150, 200 m) are retrieved "
         "from the Global Wind Atlas v3 API for the nearest 250 m grid cell."),
        ("Step 3 – Height Extrapolation",
         "ERA5 100 m wind speeds are extrapolated to hub height using a power-law profile "
         "with a diurnal shear exponent anchored to GWA statistics."),
        ("Step 4 – Weibull Bias Correction",
         "The ERA5-derived distribution at hub height is reshaped to match the GWA Weibull "
         "using a rank-preserving quantile transform."),
        ("Step 5 – AEP and Wake Calculation",
         "Gross power output is calculated per timestep using the power curve with IEC "
         "air-density correction. Net AEP is derived after applying wake losses."),
    ]
    y -= 0.005
    for title, desc in steps:
        if y < 0.18:
            _new_page("2.  Methodology (continued)")
        fig.text(ML, y, title, fontsize=9, color=C_NAVY, fontweight="bold", va="top")
        y -= 0.016
        y = _para(fig, y, desc, indent=0.015)
        y -= 0.004

    # 2.2 ERA5
    _new_page("2.  Methodology — Data Sources", "ERA5 Reanalysis and Global Wind Atlas")
    y = _h1(fig, y, "ERA5 Reanalysis Data", number="2.2")
    y -= 0.005
    y = _para(fig, y,
        "ERA5 is the fifth-generation global atmospheric reanalysis produced by the European "
        "Centre for Medium-Range Weather Forecasts (ECMWF). It covers the period from 1940 to "
        "near-present at approximately 28 km horizontal grid spacing and 1-hour timesteps, "
        "with 137 model levels from the surface to 80 km altitude."
    )
    y = _para(fig, y,
        "Variables retrieved for this assessment via the Open-Meteo ERA5 archive API:"
    )
    era5_vars = [
        "Wind speed at 100 m above ground level (m/s) — primary wind input",
        "Wind speed at 10 m above ground level (m/s) — stability fallback and turbulence proxy",
        "Wind gust at 10 m (m/s) — used to estimate sub-hourly turbulence intensity",
        "Wind direction at 100 m (°) — used for directional roughness fetch and CSV output",
        "Air temperature at 2 m (°C) — used to calculate site air density",
        "Boundary layer height (m) — primary atmospheric stability proxy for per-timestep shear",
    ]
    for v in era5_vars:
        y = _bullet(fig, y, v); y -= 0.003

    y -= 0.010
    y = _para(fig, y,
        "ERA5 accurately captures the large-scale temporal structure of the wind climate — "
        "inter-annual variability, seasonal cycles, synoptic weather systems, and diurnal "
        "patterns. However, its coarse horizontal resolution means it cannot resolve local "
        "terrain channelling, coastal effects, or roughness changes at scales below ~50 km. "
        "ERA5 wind speeds at land sites are typically biased high by 15–30% compared to "
        "measurements (Olauson 2018, Gruber et al. 2022) due to the effective smoothing of "
        "terrain roughness at the model grid scale."
    )

    y -= 0.010
    y = _h1(fig, y, "Global Wind Atlas", number="2.3")
    y -= 0.005
    y = _para(fig, y,
        "The Global Wind Atlas (GWA) v3 is produced by the Technical University of Denmark "
        "(DTU) using the mesoscale atmospheric model WRF driven by ERA5, downscaled to 250 m "
        "horizontal resolution using the linearised wind flow model WAsP. The GWA provides "
        "long-term Weibull wind speed distribution parameters (scale parameter A in m/s and "
        "shape parameter k, dimensionless) for each 250 m grid cell at heights of 50, 100, "
        "150, and 200 m above ground level, for 12 compass sectors and as omni-directional "
        "averages."
    )
    y = _para(fig, y,
        "The GWA has no temporal dimension — it represents a long-term climatological average "
        "incorporating local terrain effects, surface roughness, and coastal influences at "
        "finer spatial resolution than ERA5. It cannot capture year-to-year variability or "
        "individual weather events. The combination of ERA5 (temporal) and GWA (spatial) is "
        "designed to leverage the strengths of both datasets."
    )
    y = _para(fig, y,
        "Surface roughness class selection is a critical parameter in GWA. This tool "
        "selects the roughness class using a directional upwind fetch approach: (1) the "
        "ERA5 100 m vector-mean wind direction is computed to identify the prevailing inflow "
        "direction; (2) OpenStreetMap landuse and natural tags are queried within a 120-degree "
        "sector polygon upwind of the site at a radius of 100 x hub_height (bounded 5–15 km); "
        "(3) the dominant land cover determines roughness: water = 0.0003 m, bare ground = "
        "0.025 m, general land = 0.1 m. The closest GWA roughness class is selected. This "
        "directional fetch targets the actual upwind exposure rather than averaging over all "
        "directions. Using the sea-surface class for a land site would overestimate wind "
        "speeds by 20–30%."
    )

    # 2.4 Height extrapolation
    _new_page("2.  Methodology — Processing Steps", "Height extrapolation and bias correction")
    y = _h1(fig, y, "Wind Speed Height Extrapolation", number="2.4")
    y -= 0.005
    y = _para(fig, y,
        "ERA5 provides wind speed at 100 m above ground level. A power-law profile is used "
        "to extrapolate to the specified hub height:"
    )
    y = _eq(fig, y,
            "V_hub(t)  =  V₁₀₀(t)  ×  (h_hub / 100)^α(h)",
            "α(h) = diurnal-varying shear exponent [-], "
            "h_hub = hub height [m], V₁₀₀ = ERA5 100 m wind speed [m/s]")

    y = _para(fig, y,
        "The shear exponent alpha is not a single constant. It varies per timestep to "
        "capture individual atmospheric stability episodes. The long-term mean alpha is "
        "anchored to GWA statistics:"
    )
    y = _eq(fig, y,
            "alpha_mean  =  ln(V_GWA_hub / V_GWA_100)  /  ln(h_hub / 100)",
            "calibrates mean shear to the locally-resolved GWA wind profile")

    y = _para(fig, y,
        "Per-timestep shear variation is then computed from the ERA5 boundary layer height "
        "(BLH) as the primary stability signal. The stability index SI(t) = h_hub / BLH(t) "
        "captures the physical mechanism: when BLH is below hub height (SI > 1) the rotor "
        "is above the stable nocturnal boundary layer (low-level jet regime) and shear is "
        "strong; when BLH greatly exceeds hub height (SI << 1) the atmosphere is well-mixed "
        "and shear is low. The per-timestep shear exponent is:"
    )
    y = _eq(fig, y,
            "alpha(t)  =  alpha_mean  +  s * sigma_alpha * SI_norm(t)   [clipped]",
            "s = amplitude scale (default 1.0); sigma_alpha = std of ERA5 instantaneous "
            "10-100m shear; SI_norm = zero-mean unit-std normalised stability index")

    y = _para(fig, y,
        "Unlike an hour-of-day grouping, this per-timestep approach captures episode-specific "
        "stability: the same clock hour in winter can produce very different shear depending "
        "on whether BLH is 50 m (calm clear-sky, NLLJ regime, high alpha) or 800 m (deep "
        "frontal mixing, low alpha). If BLH is unavailable from the API, the tool falls back "
        "to deriving the diurnal alpha profile from the ERA5 10 m / 100 m wind speed ratio "
        "grouped by hour of day, normalised to alpha_mean."
    )

    y -= 0.010
    y = _h1(fig, y, "Weibull Distribution Bias Correction", number="2.5")
    y -= 0.005
    y = _para(fig, y,
        "After height extrapolation, the ERA5-derived wind speed distribution at hub height "
        "still differs from the GWA distribution due to ERA5's spatial resolution bias and "
        "terrain smoothing. A Weibull quantile transform is applied to reshape the ERA5 "
        "distribution to match the GWA Weibull parameters while preserving the rank order "
        "(and therefore the temporal structure) of the ERA5 time series:"
    )
    y = _eq(fig, y,
            "V*(t)  =  A_GWA  ×  (V_hub(t) / A_ERA5)^(k_ERA5 / k_GWA)",
            "A = Weibull scale parameter [m/s], k = Weibull shape parameter [-]")

    y = _para(fig, y,
        "This transform is rank-preserving: the temporal sequence, storm timing, seasonal "
        "patterns, and diurnal cycles are all unchanged. Only the wind speed distribution "
        "is reshaped to match the GWA climatology. The result is a time series that has "
        "the temporal richness of ERA5 and the spatial accuracy of GWA."
    )

    if has_density or has_aep:
        _new_page("2.  Methodology — AEP Calculation", "Air density, power curve, wake losses")
        y = _h1(fig, y, "Air Density Correction", number="2.6")
        y -= 0.005
        y = _para(fig, y,
            "Air density at hub height varies with elevation, temperature and humidity. "
            "Standard power curves are defined at ISO sea-level air density of 1.225 kg/m³ "
            "(15°C, 101.325 kPa). Sites at elevation or in warm climates have lower air "
            "density, and therefore lower kinetic energy flux through the rotor for the same "
            "wind speed. This assessment calculates air density per timestep from ERA5 2 m "
            "air temperature and site elevation using the International Standard Atmosphere "
            "(ISA) temperature lapse rate and the standard barometric pressure formula:"
        )
        y = _eq(fig, y,
                "T_hub  =  T₂ₘ  +  273.15  −  0.0065 × (h_hub − 2)    [K]",
                "T₂ₘ = ERA5 2 m temperature [°C], h_hub = hub height above ground [m]")
        y = _eq(fig, y,
                "P_hub  =  101325 × (1 − 2.2558×10⁻⁵ × (h_elev + h_hub))^5.2559    [Pa]",
                "h_elev = site elevation above mean sea level [m]")
        y = _eq(fig, y,
                "ρ_site  =  P_hub / (287.05 × T_hub)    [kg/m³]",
                "287.05 J/(kg·K) = specific gas constant for dry air")

        y = _para(fig, y,
            "The IEC 61400-12-1 standard method is then applied to convert actual wind speed "
            "to an equivalent wind speed at standard air density before the power curve lookup:"
        )
        y = _eq(fig, y,
                "V_eq  =  V_actual  ×  (ρ_site / 1.225)^(1/3)",
                "The (1/3) exponent arises from the cube-root relationship between "
                "kinetic power flux and air density (P ∝ ρ V³)")
        y = _para(fig, y,
            "The corrected power curve lookup uses V_eq rather than V_actual. At a site with "
            "ρ = 1.17 kg/m³ (e.g. 400 m elevation, warm climate), a 10.0 m/s wind speed "
            "becomes V_eq = 9.87 m/s — resulting in slightly lower power output than the "
            "standard curve would predict at sea level. The equivalent wind speed column in "
            "the AEP output CSV reflects this per-timestep correction."
        )

        y -= 0.010
        y = _h1(fig, y, "AEP Calculation", number="2.7")
        y -= 0.005
        y = _para(fig, y,
            "Annual Energy Production is calculated from the synthetic wind speed time series "
            "using the supplied turbine power curve. For each timestep, the gross power output "
            "is determined by interpolating the power curve at V_eq (density-corrected wind "
            "speed). The gross AEP is the mean annual sum of per-timestep power outputs:"
        )
        y = _eq(fig, y,
                "Gross AEP  =  Σ P_gross(V_eq(t)) × Δt  /  N_years    [MWh/yr]",
                "Δt = timestep duration [h], N_years = record length [yr]")

        y -= 0.010
        y = _h1(fig, y, "Wake Loss Model", number="2.8")
        y -= 0.005
        y = _para(fig, y,
            "Farm self-wake losses are estimated using a 2-D lookup matrix of percentage wake "
            "loss as a function of wind speed and total park nameplate capacity. The matrix is "
            "applied per timestep by bilinear interpolation. The net power is:"
        )
        y = _eq(fig, y,
                "P_net(t)  =  P_gross(t)  ×  (1 − wake_loss%(V, NP) / 100)",
                "NP = total park nameplate capacity [MW], V = actual wind speed [m/s]")
        y = _para(fig, y,
            "This is a simplified parametric model. It does not account for directionality "
            "of wake losses, park geometry, or atmospheric stability effects on wake recovery. "
            "The nameplate capacity used should be the TOTAL PARK capacity in MW — not the "
            "individual turbine rating — to obtain meaningful wake loss estimates. Wake losses "
            "are set to zero for nameplate ≤ 8 MW (single turbine, no self-wake)."
        )

    if has_subhourly:
        y -= 0.010
        y = _h1(fig, y, "Sub-hourly Disaggregation", number="2.9")
        y -= 0.005
        y = _para(fig, y,
            "When 30-minute or 10-minute output is requested, the hourly ERA5+GWA values are "
            "stochastically disaggregated using an Ornstein–Uhlenbeck (AR(1)) process. The "
            "per-hour turbulence intensity is estimated from the ERA5 gust factor at 10 m, "
            "scaled to hub height. Each hourly block of sub-hourly values is mean-corrected so "
            "that sub-hourly values average exactly to the ERA5+GWA hourly value. Sub-hourly "
            "output is clearly labelled as synthetic and should not be used for fatigue-load "
            "analysis or grid stability studies."
        )

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _page_map(pdf, summary_rows, page_num):
    fig = _fig()
    _page_chrome(fig, "3. Site Location", str(page_num))
    y = _section_band(fig, 0.952, 0.065,
                      "3.  Site Location",
                      "Geographic overview — GWA-corrected mean wind speed at hub height")

    ws_vals = np.array([r.get("gwa_corrected_mean_hub_ms", np.nan)
                        for r in summary_rows], dtype=float)
    lats    = np.array([r["latitude"]  for r in summary_rows], dtype=float)
    lons    = np.array([r["longitude"] for r in summary_rows], dtype=float)

    ax_map = fig.add_axes([0.05, 0.10, 0.90, 0.76])

    if _HAS_CTX and len(summary_rows) > 0:
        try:
            tf = _ProjT.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = tf.transform(lons, lats)
            pad_x  = max((xs.max() - xs.min()) * 0.40, 35_000)
            pad_y  = max((ys.max() - ys.min()) * 0.40, 35_000)
            ax_map.set_xlim(xs.min()-pad_x, xs.max()+pad_x)
            ax_map.set_ylim(ys.min()-pad_y, ys.max()+pad_y)
            ctx.add_basemap(ax_map, source=ctx.providers.Esri.WorldImagery,
                            zoom="auto", attribution_size=5)
            sc = ax_map.scatter(xs, ys, c=ws_vals, cmap=WIND_CMAP,
                                vmin=np.nanmin(ws_vals), vmax=np.nanmax(ws_vals),
                                s=130, edgecolors="white", linewidths=1.4, zorder=5)
            for i, r in enumerate(summary_rows):
                ax_map.annotate(
                    r.get("site_name", ""),
                    (xs[i], ys[i]),
                    textcoords="offset points", xytext=(8, 7),
                    fontsize=6.5, color="white", fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="#00000099", ec="none"))
            cbar = plt.colorbar(sc, ax=ax_map, fraction=0.025, pad=0.01)
            cbar.set_label("GWA-corrected mean wind speed (m/s)", fontsize=8)
            cbar.ax.tick_params(labelsize=7)
            ax_map.set_axis_off()
        except Exception:
            _map_fallback(ax_map, lats, lons, ws_vals, summary_rows)
    else:
        _map_fallback(ax_map, lats, lons, ws_vals, summary_rows)

    fig.text(0.5, 0.065,
             "Figure 1  Site locations coloured by GWA-corrected mean wind speed. "
             "Background: ESRI World Imagery © Esri, DigitalGlobe, GeoEye, Earthstar Geographics.",
             ha="center", fontsize=7, color=C_MIDGREY, style="italic")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _map_fallback(ax, lats, lons, ws_vals, summary_rows):
    ax.set_facecolor("#D4E6F1")
    sc = ax.scatter(lons, lats, c=ws_vals, cmap=WIND_CMAP, s=130,
                    edgecolors="white", linewidths=1.4, zorder=5)
    for i, r in enumerate(summary_rows):
        ax.annotate(r.get("site_name", ""), (lons[i], lats[i]),
                    textcoords="offset points", xytext=(6, 5), fontsize=7.5)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.grid(True, alpha=0.4, color=C_BORDER)
    plt.colorbar(sc, ax=ax, label="GWA-corrected mean (m/s)", fraction=0.025)


def _page_site(pdf, summary_row, site_data, pc_df, section_num, page_num):
    """Two-column layout: stats panel top, 4 charts below. Site name prominent throughout."""
    sname    = summary_row.get("site_name", "Unknown")
    lat      = summary_row.get("latitude", 0)
    lon      = summary_row.get("longitude", 0)
    elev     = summary_row.get("elevation_m_asl", "–")
    hub      = summary_row.get("hub_height_m", "–")
    ws_corr  = site_data["ws_corr"].dropna()
    ws_raw   = site_data["ws_raw"].dropna()
    meta     = site_data["meta"]
    aep      = site_data.get("aep")
    wtg      = site_data.get("wtg", "")
    nameplate = site_data.get("nameplate_mw")
    air_d    = site_data.get("air_density")

    fig = _fig()
    _page_chrome(fig, f"4. Results — {sname}", str(page_num))

    # Section header band — prominently shows site name
    subtitle = (f"{float(lat):.4f}°N  {float(lon):.4f}°E   •   "
                f"Elevation {elev} m ASL   •   Hub height {hub} m AGL")
    y = _section_band(fig, 0.952, 0.075, sname, subtitle, number=section_num)

    # ── Key stats bar ─────────────────────────────────────────────────────────
    # sits 6 pt below the header band (no overlap)
    y -= 0.008
    stats = [
        ("ERA5 100 m mean",  f"{meta.get('mean_era5_100', 0):.2f} m/s"),
        ("GWA hub mean",     f"{meta.get('mean_gwa_150', 0):.2f} m/s"),
        ("Corrected mean",   f"{meta.get('mean_corrected', 0):.2f} m/s"),
        ("Shear  α",         f"{meta.get('alpha_mean', 0):.3f}"),
        ("Air density",      f"{meta.get('mean_air_density', 1.225):.4f} kg/m³"),
    ]
    if aep:
        stats += [
            ("Turbine",      str(wtg) if wtg else "–"),
            ("Nameplate",    f"{nameplate:.1f} MW" if nameplate else "–"),
            ("Gross AEP",    f"{aep.get('gross_aep_mwh', 0)/1000:.2f} GWh/yr"),
            ("Net AEP",      f"{aep.get('net_aep_mwh', 0)/1000:.2f} GWh/yr"),
            ("Wake loss",    f"{aep.get('mean_wake_pct', 0):.1f} %"),
            ("Cap. factor",  f"{aep.get('capacity_factor', 0)*100:.1f} %"),
        ]

    n_cols = min(len(stats), 6)
    PANEL_H = 0.060
    ax_s = fig.add_axes([ML, y - PANEL_H, AVAIL, PANEL_H])
    ax_s.set_facecolor("#EFF6FF")
    ax_s.axis("off")
    col_w = 1.0 / n_cols
    for i, (lbl, val) in enumerate(stats[:n_cols * 2]):
        col_i = i % n_cols
        row_i = i // n_cols
        x = 0.008 + col_i * col_w
        yy = 0.80 - row_i * 0.46
        ax_s.text(x, yy,      lbl, transform=ax_s.transAxes,
                  fontsize=7.5, color=C_CAPTION, va="top", fontweight="bold")
        ax_s.text(x, yy-0.28, val, transform=ax_s.transAxes,
                  fontsize=8.5, color=C_NAVY,  va="top")

    _rule(fig, y - PANEL_H - 0.002, color=C_BORDER, x0=ML, width=AVAIL)
    y -= PANEL_H + 0.010

    # ── 2×2 chart grid ────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           left=0.075, right=0.955,
                           top=y - 0.005, bottom=0.075,
                           wspace=0.34, hspace=0.52)

    ax1 = fig.add_subplot(gs[0, 0])
    _plot_weibull(ax1, ws_corr, ws_raw, meta, sname=sname)

    ax2 = fig.add_subplot(gs[0, 1])
    _plot_monthly_wind(ax2, ws_corr, sname=sname)

    ax3 = fig.add_subplot(gs[1, 0])
    _plot_diurnal(ax3, ws_corr, sname=sname)

    ax4 = fig.add_subplot(gs[1, 1])
    if aep and aep.get("gross_mw") is not None:
        _plot_monthly_aep(ax4, aep, sname=sname)
    elif pc_df is not None and wtg and wtg in pc_df.columns and nameplate:
        _plot_power_curve(ax4, pc_df, wtg, nameplate, ws_corr, sname=sname)
    elif air_d is not None:
        _plot_air_density(ax4, air_d, sname=sname)
    else:
        ax4.text(0.5, 0.5, "No AEP or power curve data",
                 transform=ax4.transAxes, ha="center", va="center",
                 fontsize=9, color=C_MIDGREY)
        ax4.axis("off")

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _pages_results_tables(pdf, summary_rows, page_counter):
    def _new(title, subtitle=""):
        nonlocal fig, y
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        page_counter[0] += 1
        fig = _fig()
        _page_chrome(fig, "4. Results — Summary Tables", str(page_counter[0]))
        y = _section_band(fig, 0.952, 0.065, title, subtitle)
        y -= 0.018

    fig = _fig()
    _page_chrome(fig, "4. Results — Summary Tables", str(page_counter[0]))
    y = _section_band(fig, 0.952, 0.065,
                      "4.  Wind Resource Summary",
                      "ERA5, GWA, and corrected mean wind speeds at hub height")
    y -= 0.018

    wind_cols = [
        ("site_name",                "Site",             14),
        ("latitude",                 "Lat (°)",           6),
        ("longitude",                "Lon (°)",           7),
        ("elevation_m_asl",          "Elev\n(m ASL)",     6),
        ("hub_height_m",             "Hub\n(m)",          5),
        ("era5_mean_100m_ms",        "ERA5\n100 m",       7),
        ("gwa_mean_100m_ms",         "GWA\n100 m",        7),
        ("gwa_mean_hub_ms",          "GWA\nHub",          7),
        ("gwa_corrected_mean_hub_ms","Corr.\nMean",       7),
        ("wind_shear_alpha",         "Shear\nα",          5),
        ("mean_air_density_kg_m3",   "Air\nDensity",      7),
    ]
    y, rem = _render_table(fig, summary_rows, wind_cols, y,
                           caption="Table 2  Wind resource summary — all sites")
    while rem:
        _new("4.  Wind Resource Summary (continued)")
        y, rem = _render_table(fig, rem, wind_cols, y)

    has_aep = any("gross_aep_mwh_yr" in r for r in summary_rows)
    if has_aep:
        _new("4.  AEP Results Summary",
             "Gross / net annual energy, wake losses and capacity factor")
        aep_cols = [
            ("site_name",            "Site",             14),
            ("turbine_type",         "Turbine Model",    16),
            ("nameplate_mw",         "Nameplate\n(MW)",   7),
            ("gross_aep_mwh_yr",     "Gross AEP\n(MWh/yr)", 10),
            ("net_aep_mwh_yr",       "Net AEP\n(MWh/yr)", 10),
            ("mean_wake_loss_pct",   "Wake\nLoss (%)",    6),
            ("capacity_factor_pct",  "Cap.\nFactor (%)",  6),
            ("mean_air_density_kg_m3","Air Density\n(kg/m³)", 8),
        ]
        y, rem = _render_table(fig, summary_rows, aep_cols, y,
                               caption="Table 3  AEP results summary — all sites")
        while rem:
            _new("4.  AEP Results Summary (continued)")
            y, rem = _render_table(fig, rem, aep_cols, y)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _pages_conclusions(pdf, summary_rows, start_year, end_year, page_counter):
    pg  = page_counter[0]
    fig = _fig()
    _page_chrome(fig, "5. Conclusions and Limitations", str(pg))
    y   = _section_band(fig, 0.952, 0.065,
                        "5.  Conclusions and Limitations",
                        "Summary of findings and applicable uncertainty statements")

    y -= 0.018
    y = _h1(fig, y, "Conclusions", number="5.1")
    y -= 0.005

    n   = len(summary_rows)
    has_aep = any("gross_aep_mwh_yr" in r for r in summary_rows)
    ws_vals = [r.get("gwa_corrected_mean_hub_ms") for r in summary_rows
               if r.get("gwa_corrected_mean_hub_ms") is not None]
    ws_min, ws_max = (min(ws_vals), max(ws_vals)) if ws_vals else (0, 0)

    y = _para(fig, y,
        f"A synthetic wind resource assessment has been completed for {n} site(s) using "
        f"ERA5 reanalysis data for the period {start_year}–{end_year}, calibrated to the "
        f"Global Wind Atlas v3. The following conclusions are drawn:"
    )
    concl = [
        f"GWA-corrected mean wind speeds at hub height range from {ws_min:.2f} to "
        f"{ws_max:.2f} m/s across the assessed sites. The correction method has been shown "
        "in validation studies to reduce ERA5 land-site bias from typically 20–30% to "
        "within 5–10% of measurements.",
    ]
    if has_aep:
        aep_vals = [r.get("net_aep_mwh_yr") for r in summary_rows
                    if r.get("net_aep_mwh_yr") is not None]
        if aep_vals:
            concl.append(
                f"Indicative net AEP estimates range from "
                f"{min(aep_vals)/1000:.2f} to {max(aep_vals)/1000:.2f} GWh/yr. "
                "These are gross-to-net after simplified wake loss correction and "
                "IEC 61400-12-1 air density adjustment."
            )
    concl += [
        "Wind shear and air density both vary significantly across the assessed sites. "
        "Sites at higher elevation or in warmer climates will experience lower air density "
        "and correspondingly lower energy yield per unit swept area.",
        "The synthetic methodology is appropriate for early-stage screening and feasibility "
        "assessment. It is not a substitute for a full bankable wind resource assessment "
        "based on on-site measurement.",
    ]
    for c in concl:
        y = _bullet(fig, y, c); y -= 0.006

    y -= 0.012
    y = _h1(fig, y, "Limitations and Uncertainty", number="5.2")
    y -= 0.005

    limitations = [
        ("ERA5 spatial resolution",
         "ERA5 has ~28 km grid spacing. Terrain channelling, coastal jets, valley "
         "drainage winds, and roughness transitions at sub-28 km scales are not resolved. "
         "In complex terrain the actual wind climate may differ substantially from the "
         "ERA5+GWA synthesis."),
        ("GWA climatological average",
         "GWA represents a long-term average wind climate based on WRF/WAsP modelling "
         "driven by ERA5. It does not capture inter-annual variability; all inter-annual "
         "variation in this assessment comes from ERA5."),
        ("Wake loss simplification",
         "The 2-D wake loss matrix is a parametric model that does not account for "
         "park layout, turbine spacing, wind direction, or stability effects. "
         "Actual wake losses may differ by several percentage points from the estimate."),
        ("Absence of site measurements",
         "Results have not been validated against on-site measurements. Without "
         "measurement data, the uncertainty in long-term mean wind speed is typically "
         "±10–15% at P90 confidence level."),
        ("Power curve applicability",
         "Turbine power curves supplied by manufacturers are measured in controlled "
         "conditions. Site-specific turbulence intensity, wind shear profile, and "
         "inflow angle may cause the actual performance to differ from the curve."),
    ]
    for title, desc in limitations:
        if y < 0.14:
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
            pg += 1
            page_counter[0] = pg
            fig = _fig()
            _page_chrome(fig, "5. Conclusions and Limitations (continued)", str(pg))
            y = 0.915
        fig.text(ML, y, title, fontsize=9, color=C_NAVY, fontweight="bold", va="top")
        y -= 0.016
        y = _para(fig, y, desc, indent=0.015)
        y -= 0.006

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
    page_counter[0] = pg


def _page_wake_matrix(pdf, wake_df, page_num):
    fig = _fig()
    _page_chrome(fig, "Appendix A — Wake Loss Matrix", str(page_num))
    y = _section_band(fig, 0.952, 0.065,
                      "Appendix A — Wake Loss Matrix",
                      "Percentage self-wake loss by wind speed and total park nameplate capacity")

    ax = fig.add_axes([0.09, 0.20, 0.78, 0.61])
    caps = wake_df.columns.values.astype(float)
    ws   = wake_df.index.values.astype(float)
    data = wake_df.values
    vmax = data.max() if data.max() > 0 else 1.0

    # Log-scale x-axis so widely-spread capacity columns have even visual spacing.
    # Map each cap value to a uniform integer position, then relabel with real MW.
    n_caps   = len(caps)
    n_ws     = len(ws)
    x_pos    = np.arange(n_caps)          # 0, 1, 2, … n_caps-1
    y_pos    = np.arange(n_ws)            # 0, 1, 2, … n_ws-1

    # Build cell-edge arrays (one longer than data in each dimension)
    x_edges  = np.concatenate([[-0.5], (x_pos[:-1] + x_pos[1:]) / 2, [x_pos[-1] + 0.5]])
    y_edges  = np.concatenate([[-0.5], (y_pos[:-1] + y_pos[1:]) / 2, [y_pos[-1] + 0.5]])

    im = ax.pcolormesh(x_edges, y_edges, data, cmap="YlOrRd", vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Wake loss (%)", pad=0.015)

    # Numeric labels at each cell centre
    for i in range(n_ws):
        for j in range(n_caps):
            val = data[i, j]
            tc  = "white" if val > vmax * 0.55 else "#222222"
            ax.text(x_pos[j], y_pos[i], f"{val:.1f}%",
                    ha="center", va="center", fontsize=7, color=tc, fontweight="bold")

    # Tick marks at cell centres, labelled with real values
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{c:.0f}" for c in caps], rotation=35, ha="right", fontsize=7.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{w:.0f}" for w in ws], fontsize=7.5)
    ax.set_xlim(-0.5, n_caps - 0.5)
    ax.set_ylim(-0.5, n_ws  - 0.5)

    ax.set_xlabel("Park nameplate capacity (MW)")
    ax.set_ylabel("Wind speed (m/s)")
    ax.set_title("Wake Loss Lookup Matrix  —  bilinear interpolation applied per timestep",
                 pad=6, fontsize=9)
    ax.tick_params(labelsize=7.5)

    fig.text(ML, 0.175,
             "Figure A1  Wake loss matrix. Applied per timestep: for each hour, "
             "wind speed and total park nameplate are used to interpolate the loss "
             "fraction. Values ≤ 0% at low nameplate reflect single-turbine assumption "
             "(no self-wake). Total park nameplate must be entered, not individual turbine rating.",
             fontsize=7.5, color=C_CAPTION, va="top", style="italic",
             wrap=True)

    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


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
    Generate a professional multi-page PDF wind resource assessment report.

    Args:
        summary_rows : list of per-site summary dicts
        site_data    : dict keyed by site_name; each value has:
                         ws_corr, ws_raw, meta, air_density (opt), aep (opt),
                         wtg (opt), nameplate_mw (opt)
        wake_df      : wake loss DataFrame (or None)
        pc_df        : power curve DataFrame (or None)
        start_year / end_year : ERA5 period
    Returns:
        PDF bytes
    """
    report_dt    = datetime.now().strftime("%d %B %Y  %H:%M")
    has_aep      = any("gross_aep_mwh_yr" in r for r in summary_rows)
    has_density  = any("mean_air_density_kg_m3" in r for r in summary_rows)
    has_subhourly = False

    pc = [1]  # shared page counter

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        d = pdf.infodict()
        d["Title"]        = "ERA5 x GWA Wind Resource Assessment Report"
        d["Author"]       = "ERA5 x GWA Wind Tool"
        d["Subject"]      = f"{len(summary_rows)} sites, {start_year}–{end_year}"
        d["CreationDate"] = datetime.now()

        # 1 — Cover
        _page_cover(pdf, summary_rows, start_year, end_year, report_dt, pc[0])

        # 2 — Executive Summary
        pc[0] += 1
        _page_executive_summary(pdf, summary_rows, start_year, end_year, pc[0])

        # 3 — Introduction
        pc[0] += 1
        _pages_introduction(pdf, start_year, end_year, pc)

        # 4 — Methodology
        pc[0] += 1
        _pages_methodology(pdf, has_density, has_aep, has_subhourly, pc)

        # 5 — Site location map
        pc[0] += 1
        _page_map(pdf, summary_rows, pc[0])

        # 6 — Per-site results
        for si, row in enumerate(summary_rows):
            sname = row.get("site_name", "")
            if sname in site_data:
                pc[0] += 1
                _page_site(pdf, row, site_data[sname], pc_df=pc_df,
                           section_num=f"4.{si+1}", page_num=pc[0])

        # 7 — Summary tables
        pc[0] += 1
        _pages_results_tables(pdf, summary_rows, pc)

        # 8 — Conclusions
        pc[0] += 1
        _pages_conclusions(pdf, summary_rows, start_year, end_year, pc)

        # Appendix A — Wake matrix
        if wake_df is not None:
            pc[0] += 1
            _page_wake_matrix(pdf, wake_df, pc[0])

    return buf.getvalue()
