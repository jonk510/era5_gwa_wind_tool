# ERA5 × GWA Wind Resource Synthesis Tool

A free, web-based wind resource assessment tool built in Python/Streamlit. Fuses two globally available datasets — **ERA5 reanalysis** for temporal variability and the **Global Wind Atlas (GWA)** for local spatial accuracy — to produce a long-term synthetic wind speed time series at hub height for any onshore location on Earth. No account, no API key, no cost.

Suitable for early-stage screening, feasibility studies, and indicative energy estimates. Not a substitute for a bankable wind resource assessment based on on-site measurement.

---

## Data Sources

### ERA5 Reanalysis — temporal backbone

[ERA5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) is the ECMWF global atmospheric reanalysis, covering 1940 to near-present at ~28 km horizontal resolution and 1-hour timesteps. Accessed via the [Open-Meteo archive API](https://open-meteo.com), which returns wind speed at **10 m** and **100 m** above ground level, wind direction at 100 m, 2 m air temperature, and 10 m gust speed for the grid node nearest to the input coordinates.

ERA5 captures the full temporal structure of the wind climate — inter-annual variability, seasonal cycles, storm events, diurnal patterns, and calm periods — but its coarse resolution means it cannot represent local terrain channelling, coastal effects, or roughness changes at the sub-kilometre scale. ERA5 wind speeds are therefore systematically biased relative to what a mast at the site would actually measure.

### Global Wind Atlas (GWA) — spatial calibration

The [Global Wind Atlas](https://globalwindatlas.info) is produced by DTU (Technical University of Denmark) using WAsP mesoscale modelling driven by ERA5, downscaled to a **250 m grid**. At each grid point GWA provides Weibull scale (A) and shape (k) parameters at multiple heights (typically 10, 50, 100, 150, 200 m) for each of 12 wind direction sectors, along with sector frequencies. These statistics encode the effects of local terrain, land cover, and roughness at much finer resolution than ERA5. GWA Weibull parameters represent the long-term mean wind climate — they have no temporal dimension, but they are far more spatially accurate than ERA5 alone.

---

## Two Modes

### Single Site
Enter coordinates (WGS84 decimal degrees or a projected CRS via EPSG code), select an ERA5 period and hub height, and run the pipeline interactively. Results are displayed with charts, statistics, and a download option.

### Batch
Upload an Excel or CSV file with one row per site. Required columns: `site_name`, `latitude`, `longitude`, `hub_height`, `turbine_type`. ERA5 period and output resolution are taken from the sidebar and applied to all sites. Results are bundled into a downloadable ZIP of per-site CSVs plus a batch summary spreadsheet. Optionally generates a multi-page PDF report.

---

## Inputs

| Parameter | Description |
|---|---|
| **Latitude / Longitude** | Site coordinates. Accepts WGS84 decimal degrees or projected coordinates (easting/northing + EPSG code). |
| **Start / End Year** | ERA5 fetch period. Longer periods reduce inter-annual noise. |
| **Years to average** | Subset of the fetched period used to compute mean statistics. |
| **Output Resolution** | Hourly (native ERA5) or synthetic sub-hourly: 30-min or 10-min. |
| **Hub Height (m)** | Target height for wind speed extrapolation and GWA Weibull correction. Default 150 m. |

---

## Processing Pipeline

### Step 1 — Height extrapolation: ERA5 100 m → hub height

ERA5 provides wind at 100 m. A **power-law extrapolation** is applied hour-by-hour:

```
V_hub(t) = V_100(t) × (h_hub / 100) ^ α(h, t)
```

The shear exponent α is not constant — it varies by hour to capture the diurnal stability cycle:

- **Magnitude:** The long-term mean α is anchored to GWA. The GWA mean wind speeds at 100 m and hub height (log-linearly interpolated from the GWC Weibull parameters) give:
  ```
  α_mean = ln(V_hub_GWA / V_100_GWA) / ln(h_hub / 100)
  ```
  Clamped to the plausible range [0.05, 0.60].

- **Diurnal shape:** The hour-of-day pattern of α is derived from the ERA5 10 m / 100 m wind ratio across the full record. This 24-hour profile is normalised so its mean equals α_mean, preserving both the physically correct diurnal shape and the GWA-calibrated magnitude.

### Step 2 — Weibull quantile transform: bias correction to GWA

After height extrapolation, the ERA5-derived distribution at hub height will still differ from GWA due to ERA5's spatial resolution bias. A **Weibull quantile transform** re-shapes the ERA5 distribution to match GWA's locally-calibrated Weibull:

```
V*(t) = A_GWA × (V_hub(t) / A_ERA5) ^ (k_ERA5 / k_GWA)
```

where A_ERA5 and k_ERA5 are fitted to the ERA5 hub-height series (scipy `weibull_min.fit`), and A_GWA and k_GWA come from the GWA grid node interpolated to hub height. This transform is **rank-preserving**: the hour-by-hour sequence, storm timing, and seasonal patterns are all unchanged — only the speed distribution is reshaped to match GWA.

### Step 3 — Roughness class selection

The GWA GWC file contains Weibull parameters for multiple roughness classes (0.0, 0.03, 0.1, 0.4, 3.0 m). The tool queries OpenStreetMap within 500 m of the site to determine whether the point is over water, beach/bare ground, or general land, then selects the closest matching roughness class. This avoids the large positive bias (~25%) that would result from using the sea-surface roughness class (r = 0.0) for a land site.

### Step 4 — Air density correction (IEC 61400-12-1)

Hub-height air density is derived from ERA5 2 m temperature extrapolated to hub height using the ISA lapse rate (0.0065 K/m), and hub-height pressure from the standard barometric formula. The IEC equivalent wind speed is then:

```
V_eq(t) = V*(t) × (ρ(t) / 1.225) ^ (1/3)
```

This corrects for the effect of air density on turbine power output. Density is clipped to [0.9, 1.4] kg/m³ — a warning is shown if a significant fraction of hours are clipped (relevant for very high-altitude sites).

---

## Sub-hourly Disaggregation (optional)

When 30-min or 10-min output is selected, hourly ERA5+GWA values are stochastically disaggregated using an **AR(1) process** (discrete-time Ornstein-Uhlenbeck):

1. **Turbulence intensity (TI)** estimated from ERA5 gust factor at 10 m, then scaled to hub height using a standard height power law (`TI_hub = TI_10 × (10/h_hub)^0.11`).
2. **Sub-hourly standard deviation** reduced from instantaneous TI using von Kármán spectral theory (`σ_Tavg = TI × V × √(T_int / T_avg)`), with integral time scale T_int ≈ 350 s.
3. **AR(1) coefficient** per sub-hourly step derived from the ERA5 hourly autocorrelation assuming exponential decay (`φ = exp(−Δt / T_decorr)`).
4. Each hourly block is **mean-corrected** so sub-hourly values average exactly to the ERA5+GWA hourly value — no bias is introduced.

Each run produces one plausible realisation — it is not the real historical record and is labelled accordingly.

---

## AEP Calculation (optional)

When a turbine type and nameplate capacity are selected, the tool computes Annual Energy Production:

1. **Power curve lookup**: `P_gross(t) = interp(V_eq(t), power_curve)` — scaled to nameplate capacity.
2. **Wake losses**: applied via a 2D wake loss matrix (wind speed × total park capacity), bilinearly interpolated using `RegularGridInterpolator`. Optional — can be disabled.
3. **Other losses**: availability, electrical, turbine performance, and degradation losses applied as independent derates:
   ```
   P_net = P_gross × (1 − wake%) × (1 − avail%) × (1 − elec%) × (1 − tp%) × (1 − deg%)
   ```
4. **AEP** (MWh/yr) = sum of net power × interval_h, divided by number of years.
5. **Capacity Factor** = net AEP / (nameplate MW × 8760).

---

## Outputs & KPIs

### Wind Resource Statistics (Single Site)

- ERA5 mean wind speed at 100 m
- GWA-corrected mean wind speed at hub height
- Weibull A and k parameters (ERA5 and GWA)
- Mean wind shear exponent α (100 m → hub height)
- Diurnal α profile (hour-of-day)
- Mean air density at hub height (kg/m³)
- Roughness class selected

### Charts

- Wind speed time series (daily mean, full record)
- Monthly mean wind speed (seasonal pattern)
- Diurnal wind speed profile by month
- Wind speed frequency distribution (histogram vs. GWA Weibull target)
- Wind shear — diurnal α profile with GWA mean reference line
- Monthly mean AEP (gross and net) — after AEP calculation

### CSV Download (wind time series)

Full hourly (or sub-hourly) time series with columns: `datetime`, `ws_100m_ms`, `ws_hub_raw_ms`, `ws_hub_corrected_ms`, `alpha_h`, `air_density_kgm3`, `era5_wd_100m_deg`. Metadata header lines embedded as comments.

### CSV Download (AEP time series)

Full time series with columns: `datetime`, `wind_speed_ms`, `equiv_wind_speed_ms`, `gross_power_mw`, `wake_loss_pct`, `net_power_mw`. Summary statistics in header.

### Batch Outputs

- ZIP archive of per-site wind and AEP CSVs
- Summary Excel/CSV: one row per site with key statistics (mean wind speed, Weibull parameters, α, air density, gross AEP, net AEP, CF, wake loss)
- Optional multi-page PDF report (satellite map, methodology, charts, results tables, conclusions)

---

## PDF Report

When generated, the PDF includes:
1. **Cover page** — site map, project title, date
2. **Introduction** — background, objectives, scope and limitations
3. **Methodology** — full pipeline description with equations
4. **Results** — per-site charts and statistics tables
5. **Conclusions and limitations** — summary findings, uncertainty statements
6. **Appendix A** — wake loss matrix (if applied)

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dependencies

```
streamlit>=1.35
pandas>=2.0
numpy>=1.26
scipy>=1.13
plotly>=5.22
requests>=2.31
folium>=0.17
streamlit-folium>=0.21
timezonefinder>=6.5
pyproj>=3.6
openpyxl>=3.1
matplotlib>=3.9
contextily>=0.5
```

---

## Limitations

- **ERA5 spatial resolution ~28 km.** Terrain channelling, coastal jets, valley drainage winds, and roughness transitions at sub-28 km scales are not resolved. In complex terrain the actual wind climate may differ substantially from the ERA5+GWA synthesis.
- **GWA is a long-term climatology.** GWA Weibull parameters represent a multi-decadal mean driven by ERA5. They do not capture inter-annual variability — all year-to-year variation in the output comes from ERA5.
- **Diurnal shear uses the 10–100 m ERA5 layer.** At hub heights well above 100 m the actual diurnal shear profile in the upper layer may differ from this proxy.
- **Wake loss model is parametric.** The 2D wake matrix does not account for park layout, turbine spacing, wind direction, or atmospheric stability. Actual wake losses may differ by several percentage points.
- **Sub-hourly output is synthetic.** Each run produces one plausible realisation. Do not use it for fatigue-load analysis or as a real historical record.
- **Onshore use only.** The roughness-class selection and GWA terrain modelling assumptions are not appropriate for offshore environments or complex coastal locations.
- **Not validated against measurements.** Without on-site data, uncertainty in long-term mean wind speed is typically ±10–15% at P90 confidence.

> Results are indicative only. Use for early-stage screening and feasibility — not as a substitute for a bankable wind resource assessment based on IEC 61400-12-1 compliant on-site measurement.
