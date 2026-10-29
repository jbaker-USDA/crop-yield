# Databricks notebook source
# DBTITLE 1,Pipeline overview
# MAGIC %md
# MAGIC # Crop Yield Prediction Pipeline — Coleman County, Texas
# MAGIC
# MAGIC This notebook builds a **county-level crop yield prediction model** by combining:
# MAGIC
# MAGIC 1. **Sentinel-2 satellite NDVI** (vegetation health, 10m resolution, via Microsoft Planetary Computer — no auth required)
# MAGIC 2. **NASA POWER weather data** (temperature, precipitation, growing degree days, solar radiation — no auth required)
# MAGIC 3. **USDA NRCS SSURGO soil data** (organic matter, water capacity, drainage — no auth required)
# MAGIC 4. **USDA NASS county-level yield data** (the prediction target — free API key required)
# MAGIC
# MAGIC **Pipeline stages:** data ingestion → feature engineering → EDA → model training (Ridge regression with MLflow) → evaluation.
# MAGIC
# MAGIC **Note:** This notebook requires compute to be attached. The Sentinel-2 NDVI computation takes 5–15 minutes depending on network speed. Set `USE_SAMPLE_NDVI = True` in the configuration cell to skip imagery processing for a quick end-to-end test.
# MAGIC
# MAGIC **Important:** With ~8 years of county-level data per crop, the model demonstrates the **methodology and pipeline architecture**. Production-quality predictions require more data — more counties, more years, or field-level yield ground truth. See the final cell for limitations and next steps.

# COMMAND ----------

# DBTITLE 1,Install packages
# MAGIC %pip install -q pystac-client planetary-computer rasterio geopandas matplotlib numpy shapely requests Pillow scikit-learn mlflow seaborn

# COMMAND ----------

# DBTITLE 1,Imports and configuration
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature

# === Configuration ===

# Study area: Coleman County, Texas (same bbox as existing imagery notebook)
BBOX = [-99.424, 31.815, -99.374, 31.865]  # [west, south, east, north]
CENTROID_LON = -99.3990
CENTROID_LAT = 31.8400

# Years to analyze (Sentinel-2 has reliable coverage from 2017+)
START_YEAR = 2017
END_YEAR = 2024
YEARS = list(range(START_YEAR, END_YEAR + 1))

# Growing season: March through November
GS_START_MONTH = 3
GS_END_MONTH = 11

# Crops to model (common in Central Texas)
CROPS = ["WHEAT", "COTTON", "SORGHUM", "CORN"]

# USDA NASS Quick Stats API key (free — get one at https://quickstats.nass.usda.gov/api)
NASS_API_KEY = ""  # Paste your key here, or leave blank to use sample data

# Flags for using sample data (set to True for quick testing without API keys)
USE_SAMPLE_NDVI = False  # True = skip Sentinel-2 processing, use synthetic NDVI
USE_SAMPLE_YIELD = (NASS_API_KEY == "")  # Auto-set: True if no NASS key

# Output directory for saved images
OUTPUT_DIR = "/Workspace/Users/jason.baker@usda.gov/yield_prediction_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Configuration loaded for Coleman County, Texas")
print(f"Years: {START_YEAR}–{END_YEAR} ({len(YEARS)} years)")
print(f"Crops: {', '.join(CROPS)}")
print(f"Growing season: months {GS_START_MONTH}–{GS_END_MONTH}")
print(f"Sample yield data: {'YES (no NASS key)' if USE_SAMPLE_YIELD else 'NO (using NASS API)'}")
print(f"Sample NDVI data: {'YES' if USE_SAMPLE_NDVI else 'NO (computing from Sentinel-2)'}")

# COMMAND ----------

# DBTITLE 1,Weather data section
# MAGIC %md
# MAGIC ## 1. Weather Data — NASA POWER API
# MAGIC
# MAGIC NASA POWER provides free daily weather data (no authentication required) for any location worldwide. We fetch daily temperature, precipitation, solar radiation, humidity, and wind speed for the centroid of Coleman County, then aggregate into growing-season features per year.
# MAGIC
# MAGIC API docs: https://power.larc.nasa.gov/docs/methodology/

# COMMAND ----------

# DBTITLE 1,Fetch NASA POWER weather data
# Fetch daily weather from NASA POWER API (free, no auth)
power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "T2M_MAX,T2M_MIN,T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,RH2M,WS2M",
    "community": "AG",
    "longitude": CENTROID_LON,
    "latitude": CENTROID_LAT,
    "start": f"{START_YEAR}0101",
    "end": f"{END_YEAR}1231",
    "format": "JSON",
}

print(f"Fetching daily weather from NASA POWER for ({CENTROID_LAT}, {CENTROID_LON})...")
resp = requests.get(power_url, params=params, timeout=60)
resp.raise_for_status()
power_data = resp.json()

# Parse into daily DataFrame
param_data = power_data["properties"]["parameter"]
dates = sorted(param_data["T2M_MAX"].keys())

weather_daily = pd.DataFrame({
    "date": pd.to_datetime(dates, format="%Y%m%d"),
    "tmax": [param_data["T2M_MAX"][d] for d in dates],
    "tmin": [param_data["T2M_MIN"][d] for d in dates],
    "tavg": [param_data["T2M"][d] for d in dates],
    "precip": [param_data["PRECTOTCORR"][d] for d in dates],
    "solar_rad": [param_data["ALLSKY_SFC_SW_DWN"][d] for d in dates],
    "humidity": [param_data["RH2M"][d] for d in dates],
    "wind_speed": [param_data["WS2M"][d] for d in dates],
})

# Replace -999 (NASA missing value indicator) with NaN
weather_daily = weather_daily.replace(-999.0, np.nan)
weather_daily["year"] = weather_daily["date"].dt.year
weather_daily["month"] = weather_daily["date"].dt.month

# Filter to growing season (March–November)
gs_mask = (weather_daily["month"] >= GS_START_MONTH) & (weather_daily["month"] <= GS_END_MONTH)
gs = weather_daily[gs_mask].copy()

# Compute growing degree days (base 10°C)
gs["gdd"] = ((gs["tmax"] + gs["tmin"]) / 2 - 10).clip(lower=0)

# Aggregate to seasonal features per year
weather_features = gs.groupby("year").agg(
    gdd_total=("gdd", "sum"),
    precip_total_mm=("precip", "sum"),
    temp_avg_c=("tavg", "mean"),
    temp_max_avg_c=("tmax", "mean"),
    temp_min_avg_c=("tmin", "mean"),
    solar_rad_avg=("solar_rad", "mean"),
    humidity_avg_pct=("humidity", "mean"),
    wind_speed_avg=("wind_speed", "mean"),
    heat_stress_days=("tmax", lambda x: (x > 35).sum()),
    frost_days=("tmin", lambda x: (x < 0).sum()),
).reset_index()

display(weather_features)
print(f"\nWeather features computed for {len(weather_features)} years")

# COMMAND ----------

# DBTITLE 1,NASS yield data section
# MAGIC %md
# MAGIC ## 2. Crop Yield Data — USDA NASS Quick Stats
# MAGIC
# MAGIC USDA NASS provides county-level crop yield statistics. A free API key is required (get one at https://quickstats.nass.usda.gov/api — instant approval). If no key is provided, the notebook uses sample yield data based on typical Coleman County ranges so the pipeline remains testable.

# COMMAND ----------

# DBTITLE 1,Fetch NASS yield data
if not USE_SAMPLE_YIELD:
    # Fetch real yield data from NASS Quick Stats API
    print("Fetching yield data from USDA NASS Quick Stats API...")
    nass_url = "https://quickstats.nass.usda.gov/api/api_GET/"
    all_yields = []

    for crop in CROPS:
        params = {
            "key": NASS_API_KEY,
            "commodity_desc": crop,
            "year__GE": START_YEAR,
            "year__LE": END_YEAR,
            "county_name": "COLEMAN",
            "state_name": "TEXAS",
            "statisticcat_desc": "YIELD",
            "format": "JSON",
        }
        resp = requests.get(nass_url, params=params, timeout=60)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for r in data:
                val = r.get("Value", "")
                try:
                    yield_val = float(val)
                except (ValueError, TypeError):
                    continue  # Skip suppressed/missing values like "(D)", "(Z)"
                all_yields.append({
                    "year": int(r["year"]),
                    "crop": r["commodity_desc"],
                    "yield": yield_val,
                    "unit": r.get("unit_desc", ""),
                })
            print(f"  {crop}: {len([y for y in all_yields if y['crop'] == crop])} records")
        else:
            print(f"  {crop}: API error {resp.status_code}")

    yield_df = pd.DataFrame(all_yields)
    if yield_df.empty:
        print("No data returned from NASS API. Falling back to sample data.")
        USE_SAMPLE_YIELD = True

if USE_SAMPLE_YIELD:
    # Sample yield data (realistic ranges for Coleman County, TX)
    # 2022 was a severe drought year in Central Texas — lower yields
    print("Using sample yield data (no NASS API key provided).")
    print("Get a free key at https://quickstats.nass.usda.gov/api for real data.\n")

    sample_yields = {
        "WHEAT":   [28, 35, 30, 38, 32, 18, 25, 35],   # bu/acre
        "COTTON":  [420, 580, 480, 620, 510, 280, 380, 560],  # lb/acre
        "SORGHUM": [42, 55, 48, 62, 55, 28, 40, 52],   # bu/acre
        "CORN":    [65, 85, 72, 92, 78, 42, 60, 88],    # bu/acre
    }
    sample_units = {
        "WHEAT": "BU / ACRE", "COTTON": "LB / ACRE",
        "SORGHUM": "BU / ACRE", "CORN": "BU / ACRE",
    }

    records = []
    for crop, values in sample_yields.items():
        for i, year in enumerate(YEARS):
            records.append({
                "year": year,
                "crop": crop,
                "yield": values[i],
                "unit": sample_units[crop],
            })
    yield_df = pd.DataFrame(records)

display(yield_df)
print(f"\nYield data: {len(yield_df)} records across {yield_df['crop'].nunique()} crops")

# COMMAND ----------

# DBTITLE 1,Soil data section
# MAGIC %md
# MAGIC ## 3. Soil Data — USDA NRCS SSURGO
# MAGIC
# MAGIC SSURGO (Soil Survey Geographic Database) provides detailed soil properties. We query the NRCS Soil Data Access (SDA) REST API for Coleman County (areasymbol TX083) to get area-weighted organic matter, available water capacity, drainage class, and texture.

# COMMAND ----------

# DBTITLE 1,Fetch SSURGO soil data
# Query SSURGO Soil Data Access (SDA) REST API for Coleman County, TX
# Coleman County FIPS = 48083, SSURGO areasymbol = "TX083"
sda_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/post.rest"

sql_query = """
SELECT 
  co.comppct_r as comp_pct,
  co.om_r as om,
  co.drainagecl as drainage,
  co.slope_r as slope,
  ch.awcr_r as awc,
  ch.sandtotal_r as sand,
  ch.claytotal_r as clay,
  ch.silttotal_r as silt
FROM legend l
JOIN mapunit mu ON mu.lkey = l.lkey
JOIN component co ON co.mukey = mu.mukey
LEFT JOIN chorizon ch ON ch.cokey = co.cokey AND ch.hzdept_r = 0
WHERE l.areasymbol = 'TX083'
"""

soil_features = {}

try:
    resp = requests.post(sda_url, json={"QUERY": sql_query, "FORMAT": "JSON"}, timeout=30)
    resp.raise_for_status()
    soil_json = resp.json()
    
    if "Table" in soil_json and len(soil_json["Table"]) > 1:
        headers = soil_json["Table"][0]
        rows = soil_json["Table"][1:]
        soil_df = pd.DataFrame(rows, columns=headers)
        
        # Convert numeric columns
        for col in ["comp_pct", "om", "slope", "awc", "sand", "clay", "silt"]:
            soil_df[col] = pd.to_numeric(soil_df[col], errors="coerce")
        
        # Compute area-weighted averages (weight by component percent)
        soil_df["weight"] = soil_df["comp_pct"].fillna(0)
        
        def weighted_avg(col):
            vals = soil_df[col]
            mask = vals.notna() & (soil_df["weight"] > 0)
            if mask.sum() == 0:
                return np.nan
            return np.average(vals[mask], weights=soil_df["weight"][mask])
        
        soil_features = {
            "soil_om_pct": weighted_avg("om"),
            "soil_awc": weighted_avg("awc"),
            "soil_slope_pct": weighted_avg("slope"),
            "soil_sand_pct": weighted_avg("sand"),
            "soil_clay_pct": weighted_avg("clay"),
            "soil_silt_pct": weighted_avg("silt"),
            "soil_drainage": soil_df.loc[soil_df["weight"].idxmax(), "drainage"] if len(soil_df) > 0 else "Unknown",
        }
        print("Soil data fetched from SSURGO SDA for Coleman County, TX (TX083)")
    else:
        raise ValueError("Unexpected SDA response format")
        
except Exception as e:
    print(f"SSURGO SDA query failed ({e}). Using known soil properties for Coleman County, TX.")
    # Fallback: representative soil values for Coleman County, TX
    # (Cross Timbers / Rolling Plains ecoregion — sandy loam to clay loam)
    soil_features = {
        "soil_om_pct": 1.2,       # Organic matter %
        "soil_awc": 0.14,          # Available water capacity (cm/cm)
        "soil_slope_pct": 3.0,    # Slope %
        "soil_sand_pct": 45.0,    # Sand %
        "soil_clay_pct": 22.0,    # Clay %
        "soil_silt_pct": 33.0,    # Silt %
        "soil_drainage": "Well drained",
    }

print("\nSoil features (area-weighted):")
for k, v in soil_features.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# DBTITLE 1,NDVI features section
# MAGIC %md
# MAGIC ## 4. Sentinel-2 NDVI Features with CDL Crop Mask
# MAGIC
# MAGIC We search Sentinel-2 L2A imagery (10m, 5-day revisit) via the free Planetary Computer STAC API. For each month of the growing season, we select the least-cloudy scene, read the red (B04) and near-infrared (B08) bands for our AOI, and compute NDVI = (B08 − B04) / (B08 + B04).
# MAGIC
# MAGIC **Crop mask:** We apply the USDA NASS Cropland Data Layer (CDL) to exclude non-cropland pixels (rangeland, forest, water, urban). NDVI is averaged only over cropland pixels, providing a cleaner crop-specific vegetation signal.
# MAGIC
# MAGIC Monthly NDVI values are then aggregated into seasonal features (peak, area-under-curve, green-up rate, etc.).
# MAGIC
# MAGIC Set `USE_SAMPLE_NDVI = True` in the configuration cell to skip this step (~10 min) and use synthetic NDVI data.

# COMMAND ----------

# DBTITLE 1,CDL crop mask section
# MAGIC %md
# MAGIC ## 4a. CDL Crop Mask — USDA NASS Cropland Data Layer
# MAGIC
# MAGIC The Cropland Data Layer (CDL) provides 30m crop type classification for the entire US. We download it from Microsoft Planetary Computer (free, no auth) and create a binary **cropland mask** that excludes rangeland, forest, water, and urban areas. Only cropland pixels (CDL codes 1–61) are included.
# MAGIC
# MAGIC This mask is applied to the Sentinel-2 NDVI computation so that NDVI reflects crop vegetation only, not mixed land cover.
# MAGIC
# MAGIC CDL is available 2013–2021 on Planetary Computer. For 2022–2024, we use the 2021 CDL as a proxy (crop patterns change slowly).

# COMMAND ----------

# DBTITLE 1,Download CDL and create crop masks
# Download USDA NASS Cropland Data Layer (CDL) from Planetary Computer
# CDL provides 30m crop type classification, available 2013-2021 on Planetary Computer
# For years beyond 2021, we use the most recent available CDL as a proxy.

import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from shapely.geometry import box, mapping

print("Downloading CDL crop masks from Planetary Computer...")
catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
aoi = box(*BBOX)
aoi_geom = mapping(aoi)

# Search for all CDL items intersecting our AOI
search = catalog.search(collections=["usda-cdl"], intersects=aoi_geom)
all_items = list(search.get_items())

# Organize cropland items by year (skip "cultivated" items)
cropland_items = {}
for item in all_items:
    if item.id.startswith("cropland_"):
        year = int(item.id.split("_")[1])
        cropland_items[year] = item

print(f"CDL cropland available for years: {sorted(cropland_items.keys())}")

# Common CDL crop codes for Central Texas
crop_names = {
    1: "Corn", 2: "Cotton", 3: "Sorghum", 12: "Winter Wheat",
    24: "Winter Wheat", 27: "Rye", 28: "Oats",
    36: "Alfalfa", 37: "Other Hay", 61: "Fallow/Idle",
    63: "Forest", 64: "Shrubland", 65: "Barren",
    152: "Shrubland", 176: "Grassland/Pasture",
}

# Download and create crop masks
# CDL codes 1-61 = agricultural crops (cropland)
# CDL codes 63+  = forest, water, urban, wetlands, barren (non-cropland)
crop_masks = {}

# Crop-specific CDL codes for per-crop NDVI masks
CROP_CDL_CODES = {"WHEAT": 24, "COTTON": 2, "SORGHUM": 3, "CORN": 1}
crop_specific_masks = {crop: {} for crop in CROPS}

for year in YEARS:
    # Use most recent available CDL year
    cdl_year = year
    if year not in cropland_items:
        cdl_year = max(y for y in cropland_items if y <= year)
        print(f"\n  {year}: using CDL {cdl_year} (not available for {year})")
    else:
        print(f"\n  {year}: using CDL {cdl_year}")
    
    item = cropland_items[cdl_year]
    signed = planetary_computer.sign(item)
    cdl_url = signed.assets["cropland"].href
    
    with rasterio.open(cdl_url) as src:
        # Transform AOI bbox from WGS84 to CDL CRS (typically Albers EPSG:5070)
        w, s, e, n = transform_bounds("EPSG:4326", src.crs, *BBOX)
        win = from_bounds(w, s, e, n, src.transform)
        
        # Read at 200x200 to match Sentinel-2 processing resolution
        cdl_data = src.read(1, window=win, out_shape=(200, 200), resampling=Resampling.nearest)
    
    # Binary crop mask: 1 = cropland (codes 1-61), 0 = non-cropland
    crop_mask = ((cdl_data > 0) & (cdl_data <= 61)).astype(np.uint8)
    crop_fraction = crop_mask.mean()
    crop_masks[year] = crop_mask
    
    # Create crop-specific masks for each crop type
    for crop_name, cdl_code in CROP_CDL_CODES.items():
        crop_specific_masks[crop_name][year] = (cdl_data == cdl_code).astype(np.uint8)
    
    # Print land cover breakdown
    unique, counts = np.unique(cdl_data, return_counts=True)
    print(f"    Cropland fraction: {crop_fraction:.1%} ({int(crop_mask.sum())}/40000 pixels)")
    print(f"    Land cover breakdown:")
    for u, c in zip(unique, counts):
        name = crop_names.get(u, f"code_{u}")
        is_crop = "[CROP]" if 0 < u <= 61 else "[non-crop]"
        print(f"      {is_crop} {name} (code {u}): {c} px ({c/40000:.1%})")

print(f"\nGeneral crop masks created for {len(crop_masks)} years")
print(f"Crop-specific masks created for: {', '.join(CROPS)}")
for crop_name in CROPS:
    total_px = sum(int(mask.sum()) for mask in crop_specific_masks[crop_name].values())
    print(f"  {crop_name} (CDL code {CROP_CDL_CODES[crop_name]}): {total_px} total crop pixels across all years")
print("\nNDVI will be computed per crop type using crop-specific masks.")

# COMMAND ----------

# DBTITLE 1,Compute Sentinel-2 NDVI time series
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from rasterio.enums import Resampling
from shapely.geometry import box, mapping

ndvi_records = []

if USE_SAMPLE_NDVI:
    print("Using sample NDVI data (USE_SAMPLE_NDVI = True) -- crop-specific phenology curves")
    # Generate realistic monthly NDVI with crop-specific phenology for Central Texas
    crop_base_curves = {
        "WHEAT":  {3: 0.40, 4: 0.55, 5: 0.65, 6: 0.60, 7: 0.35, 8: 0.25, 9: 0.25, 10: 0.25, 11: 0.20},
        "COTTON": {3: 0.15, 4: 0.20, 5: 0.30, 6: 0.45, 7: 0.65, 8: 0.70, 9: 0.55, 10: 0.35, 11: 0.20},
        "SORGHUM":{3: 0.15, 4: 0.20, 5: 0.35, 6: 0.55, 7: 0.70, 8: 0.65, 9: 0.45, 10: 0.25, 11: 0.20},
        "CORN":   {3: 0.20, 4: 0.35, 5: 0.50, 6: 0.65, 7: 0.60, 8: 0.35, 9: 0.25, 10: 0.20, 11: 0.15},
    }
    for year in YEARS:
        drought_factor = 0.82 if year == 2022 else (0.90 if year == 2023 else 1.0)
        for crop in CROPS:
            np.random.seed(year * 100 + ord(crop[0]))
            for month in range(GS_START_MONTH, GS_END_MONTH + 1):
                variation = np.random.uniform(-0.04, 0.04)
                ndvi = max(0.1, crop_base_curves[crop][month] * drought_factor + variation)
                ndvi_records.append({"year": year, "month": month, "crop": crop, "ndvi": ndvi})
    print(f"Generated {len(ndvi_records)} sample NDVI records ({len(CROPS)} crops)")

else:
    print("Computing NDVI from Sentinel-2 imagery via Planetary Computer...")
    catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    aoi = box(*BBOX)
    aoi_geom = mapping(aoi)

    for year in YEARS:
        for month in range(GS_START_MONTH, GS_END_MONTH + 1):
            start = f"{year}-{month:02d}-01"
            # Use first day of next month as end date (exclusive in STAC)
            if month < 12:
                end = f"{year}-{month+1:02d}-01"
            else:
                end = f"{year+1}-01-01"

            search = catalog.search(
                collections=["sentinel-2-l2a"],
                intersects=aoi_geom,
                datetime=f"{start}/{end}",
                query={"eo:cloud_cover": {"lt": 20}},
            )
            items = list(search.get_items())
            if not items:
                print(f"  {year}-{month:02d}: no cloud-free scenes found")
                continue

            # Select least cloudy scene
            items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
            item = items[0]
            cloud_cover = item.properties.get("eo:cloud_cover", "?")
            signed = planetary_computer.sign(item)

            try:
                b04_url = signed.assets["B04"].href
                b08_url = signed.assets["B08"].href

                with rasterio.open(b04_url) as src_b04, rasterio.open(b08_url) as src_b08:
                    # Transform AOI bbox from WGS84 to raster CRS
                    w, s, e, n = transform_bounds("EPSG:4326", src_b04.crs, *BBOX)
                    win = from_bounds(w, s, e, n, src_b04.transform)

                    # Read at reduced resolution for efficiency
                    out_shape = (200, 200)
                    b04 = src_b04.read(1, window=win, out_shape=out_shape, resampling=Resampling.average)
                    b08 = src_b08.read(1, window=win, out_shape=out_shape, resampling=Resampling.average)

                # Compute NDVI
                ndvi = (b08.astype(float) - b04.astype(float)) / (b08.astype(float) + b04.astype(float) + 1e-10)
                # Mask invalid values
                ndvi = np.where((ndvi >= -1) & (ndvi <= 1), ndvi, np.nan)

                # Apply crop-specific CDL masks: compute NDVI for each crop type
                crop_ndvis = {}
                for crop_name in CROPS:
                    crop_mask = crop_specific_masks.get(crop_name, {}).get(year)
                    if crop_mask is not None and crop_mask.shape == ndvi.shape and crop_mask.sum() > 0:
                        masked_ndvi = np.where(crop_mask > 0, ndvi, np.nan)
                        mean_ndvi = np.nanmean(masked_ndvi)
                        crop_px = int(crop_mask.sum())
                    else:
                        # Fallback to general cropland mask if no pixels for this crop
                        gen_mask = crop_masks.get(year)
                        if gen_mask is not None and gen_mask.sum() > 0:
                            masked_ndvi = np.where(gen_mask > 0, ndvi, np.nan)
                            mean_ndvi = np.nanmean(masked_ndvi)
                        else:
                            mean_ndvi = np.nanmean(ndvi)
                        crop_px = 0
                    crop_ndvis[crop_name] = mean_ndvi
                    ndvi_records.append({"year": year, "month": month, "crop": crop_name, "ndvi": mean_ndvi})

                ndvi_str = ", ".join(f"{c}={crop_ndvis[c]:.3f}" for c in CROPS)
                print(f"  {year}-{month:02d}: cloud={cloud_cover}% -- {ndvi_str}")

            except Exception as e:
                print(f"  {year}-{month:02d}: error reading bands — {e}")

ndvi_df = pd.DataFrame(ndvi_records)
print(f"\nNDVI records: {len(ndvi_records)} ({ndvi_df['year'].nunique()} years × {ndvi_df['crop'].nunique()} crops × ~9 months)")
display(ndvi_df.head(10))

# COMMAND ----------

# DBTITLE 1,Compute NDVI seasonal features
# Compute seasonal NDVI features per crop
ndvi_features_list = []

for crop in CROPS:
    crop_ndvi = ndvi_df[ndvi_df["crop"] == crop]
    if len(crop_ndvi) == 0:
        continue
    ndvi_pivot = crop_ndvi.pivot(index="year", columns="month", values="ndvi")

    features_df = pd.DataFrame({"year": ndvi_pivot.index})
    features_df["crop"] = crop

    # Overall seasonal metrics
    features_df["ndvi_peak"] = ndvi_pivot.max(axis=1).values
    features_df["ndvi_auc"] = ndvi_pivot.sum(axis=1).values
    features_df["ndvi_mean"] = ndvi_pivot.mean(axis=1).values
    features_df["ndvi_std"] = ndvi_pivot.std(axis=1).values

    # Phenology metrics
    march = ndvi_pivot.get(3, pd.Series(0.0, index=ndvi_pivot.index))
    july = ndvi_pivot.get(7, pd.Series(0.0, index=ndvi_pivot.index))
    nov = ndvi_pivot.get(11, pd.Series(0.0, index=ndvi_pivot.index))
    features_df["ndvi_greenup"] = (july - march).values
    features_df["ndvi_senescence"] = (july - nov).values

    # Monthly NDVI values as individual features
    for month in range(GS_START_MONTH, GS_END_MONTH + 1):
        if month in ndvi_pivot.columns:
            features_df[f"ndvi_m{month}"] = ndvi_pivot[month].values
        else:
            features_df[f"ndvi_m{month}"] = np.nan

    ndvi_features_list.append(features_df)

ndvi_features = pd.concat(ndvi_features_list, ignore_index=True)

display(ndvi_features)
print(f"\nNDVI seasonal features: {ndvi_features.shape[1]-2} features for {len(ndvi_features)} year-crop combinations")

# COMMAND ----------

# DBTITLE 1,Feature combination section
# MAGIC %md
# MAGIC ## 5. Combine Features into Training Dataset
# MAGIC
# MAGIC Merge yield data (target) with weather, NDVI, and soil features (predictors) at the county-year-crop level. Soil features are constant within the county — they provide no within-county variation but are included for completeness and multi-county extensibility.

# COMMAND ----------

# DBTITLE 1,Merge into training dataset
# Merge yield data with weather and NDVI features on year
training_df = yield_df.merge(weather_features, on="year", how="left")
training_df = training_df.merge(ndvi_features, on=["year", "crop"], how="left")

# Add soil features as constant columns (same for all rows in this county)
for k, v in soil_features.items():
    training_df[k] = v

# Drop rows with missing yield or critical features
training_df = training_df.dropna(subset=["yield"])
training_df = training_df.reset_index(drop=True)

print(f"Training dataset shape: {training_df.shape}")
print(f"  Rows: {len(training_df)} ({training_df['crop'].nunique()} crops × ~{len(YEARS)} years)")
print(f"  Columns: {training_df.shape[1]} (target + features)")
print(f"\nColumn summary:")
for col in training_df.columns:
    print(f"  {col}: {training_df[col].dtype}")

display(training_df)

# COMMAND ----------

# DBTITLE 1,EDA section
# MAGIC %md
# MAGIC ## 6. Exploratory Data Analysis
# MAGIC
# MAGIC Examine the training dataset: descriptive statistics, feature distributions, and correlations between features and yield.

# COMMAND ----------

# DBTITLE 1,EDA visualizations
# Summary statistics
print("=== Training Data Summary ===")
display(training_df.describe())

# Feature types table
print("\n=== Feature Types ===")
feature_types = []
for col in training_df.columns:
    if col == "yield":
        ftype = "target (numeric)"
    elif col in ("crop", "unit", "soil_drainage"):
        ftype = "categorical"
    elif col == "year":
        ftype = "numeric (integer)"
    else:
        ftype = "numeric (float)"
    feature_types.append({"feature": col, "type": ftype})
display(pd.DataFrame(feature_types))

# Correlation heatmap (numeric features only)
numeric_cols = training_df.select_dtypes(include=[np.number]).columns.tolist()
if "year" in numeric_cols:
    numeric_cols.remove("year")

corr = training_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, 
            square=True, linewidths=0.5, ax=ax, annot_kws={"size": 7})
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# Yield by crop over years
fig, ax = plt.subplots(figsize=(10, 5))
for crop in CROPS:
    crop_data = training_df[training_df["crop"] == crop]
    ax.plot(crop_data["year"], crop_data["yield"], marker="o", label=crop)
ax.set_xlabel("Year")
ax.set_ylabel("Yield")
ax.set_title("Crop Yield Over Time — Coleman County, TX")
ax.legend()
ax.set_xticks(YEARS)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_yield_trend.png", dpi=150, bbox_inches="tight")
plt.show()

# Scatter: peak NDVI vs yield (per crop)
fig, ax = plt.subplots(figsize=(8, 6))
for crop in CROPS:
    crop_data = training_df[training_df["crop"] == crop]
    ax.scatter(crop_data["ndvi_peak"], crop_data["yield"], label=crop, s=60)
ax.set_xlabel("Peak NDVI")
ax.set_ylabel("Yield")
ax.set_title("Peak NDVI vs Yield by Crop")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_ndvi_vs_yield.png", dpi=150, bbox_inches="tight")
plt.show()

# Scatter: GDD vs yield
fig, ax = plt.subplots(figsize=(8, 6))
for crop in CROPS:
    crop_data = training_df[training_df["crop"] == crop]
    ax.scatter(crop_data["gdd_total"], crop_data["yield"], label=crop, s=60)
ax.set_xlabel("Total Growing Degree Days (°C)")
ax.set_ylabel("Yield")
ax.set_title("GDD vs Yield by Crop")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eda_gdd_vs_yield.png", dpi=150, bbox_inches="tight")
plt.show()

# COMMAND ----------

# DBTITLE 1,Model training section
# MAGIC %md
# MAGIC ## 7. Model Training — Ridge Regression with MLflow
# MAGIC
# MAGIC We train a **separate Ridge regression model per crop** (yields have different units and scales across crops). Each model uses:
# MAGIC
# MAGIC - **Features:** NDVI seasonal metrics + weather features + year (captures technology trend)
# MAGIC - **Preprocessing:** StandardScaler (in a sklearn Pipeline, encapsulated in the logged model)
# MAGIC - **Target transform:** log1p (yields are right-skewed, especially for cotton)
# MAGIC - **Validation:** Leave-One-Out Cross-Validation (LOOCV) — appropriate for small datasets
# MAGIC
# MAGIC Ridge is chosen over OLS for stability with correlated features and few data points. With ~8 samples per crop, this is a demonstration of methodology; production use requires more data.

# COMMAND ----------

# DBTITLE 1,Train models with MLflow
# Features to use for prediction
# Exclude target, identifiers, and constant soil features (no within-county variation)
exclude_cols = {"yield", "crop", "unit",
                "soil_om_pct", "soil_awc", "soil_slope_pct", "soil_sand_pct",
                "soil_clay_pct", "soil_silt_pct", "soil_drainage"}
feature_cols = [c for c in training_df.columns if c not in exclude_cols]

print(f"Features used ({len(feature_cols)}): {feature_cols}\n")

# MLflow experiment
mlflow.set_experiment("/Users/jason.baker@usda.gov/Crop Yield Prediction - Coleman County TX")

results = {}

for crop in CROPS:
    crop_df = training_df[training_df["crop"] == crop].sort_values("year").reset_index(drop=True)
    if len(crop_df) < 5:
        print(f"{crop}: skipped (only {len(crop_df)} samples)")
        continue

    X = crop_df[feature_cols].values
    y = crop_df["yield"].values

    # Build pipeline: StandardScaler → Ridge, with log1p target transform
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42)),
    ])
    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # --- Leave-One-Out Cross-Validation ---
    loo = LeaveOneOut()
    y_pred_loo = np.zeros_like(y, dtype=float)

    for train_idx, test_idx in loo.split(X):
        model.fit(X[train_idx], y[train_idx])
        y_pred_loo[test_idx] = model.predict(X[test_idx])

    # LOOCV metrics
    r2 = r2_score(y, y_pred_loo)
    rmse = np.sqrt(mean_squared_error(y, y_pred_loo))
    mae = mean_absolute_error(y, y_pred_loo)

    # Fit final model on all data
    model.fit(X, y)
    y_pred_train = model.predict(X)

    # Log to MLflow
    with mlflow.start_run(run_name=f"{crop}_Ridge_LOOCV"):
        mlflow.log_param("crop", crop)
        mlflow.log_param("model", "Ridge")
        mlflow.log_param("alpha", 1.0)
        mlflow.log_param("n_samples", len(crop_df))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("cv", "LOOCV")

        mlflow.log_metric("loocv_r2", r2)
        mlflow.log_metric("loocv_rmse", rmse)
        mlflow.log_metric("loocv_mae", mae)
        mlflow.log_metric("train_r2", r2_score(y, y_pred_train))

        X_example = crop_df[feature_cols].head(3)
        signature = infer_signature(X_example, model.predict(X_example))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_example,
        )

    results[crop] = {
        "model": model,
        "y_true": y,
        "y_pred_loo": y_pred_loo,
        "y_pred_train": y_pred_train,
        "years": crop_df["year"].values,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "feature_cols": feature_cols,
    }

    print(f"{crop}: LOOCV R²={r2:.3f}, RMSE={rmse:.1f}, MAE={mae:.1f} (n={len(crop_df)})")

print(f"\nModels trained and logged to MLflow for {len(results)} crops")

# COMMAND ----------

# DBTITLE 1,Evaluation section
# MAGIC %md
# MAGIC ## 8. Model Evaluation
# MAGIC
# MAGIC Evaluate each crop model using LOOCV predictions: R², RMSE, MAE, predicted-vs-actual plots, residual distributions, and feature importance (Ridge coefficients).

# COMMAND ----------

# DBTITLE 1,Evaluate and visualize results
# Summary metrics table
metrics_rows = []
for crop, res in results.items():
    metrics_rows.append({
        "Crop": crop,
        "Samples": len(res["y_true"]),
        "LOOCV R²": f"{res['r2']:.3f}",
        "LOOCV RMSE": f"{res['rmse']:.1f}",
        "LOOCV MAE": f"{res['mae']:.1f}",
    })
print("=== Model Performance Summary (LOOCV) ===")
display(pd.DataFrame(metrics_rows))

# Predicted vs Actual plots
n_crops = len(results)
fig, axes = plt.subplots(1, n_crops, figsize=(5 * n_crops, 5), squeeze=False)
for i, (crop, res) in enumerate(results.items()):
    ax = axes[0][i]
    ax.scatter(res["y_true"], res["y_pred_loo"], s=80, zorder=3)
    min_val = min(res["y_true"].min(), res["y_pred_loo"].min())
    max_val = max(res["y_true"].max(), res["y_pred_loo"].max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", zorder=2)
    ax.set_xlabel("Actual Yield")
    ax.set_ylabel("Predicted Yield (LOOCV)")
    ax.set_title(f"{crop}\nR²={res['r2']:.3f}, RMSE={res['rmse']:.1f}")
plt.suptitle("Predicted vs Actual Yield (Leave-One-Out CV)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eval_pred_vs_actual.png", dpi=150, bbox_inches="tight")
plt.show()

# Residual plots with distribution
fig, axes = plt.subplots(2, n_crops, figsize=(5 * n_crops, 8), squeeze=False)
for i, (crop, res) in enumerate(results.items()):
    residuals = res["y_true"] - res["y_pred_loo"]
    # Residuals vs predicted
    ax = axes[0][i]
    ax.scatter(res["y_pred_loo"], residuals, s=80, zorder=3)
    ax.axhline(0, color="r", linestyle="--")
    ax.set_xlabel("Predicted Yield")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{crop} Residuals")
    # Residual distribution
    ax = axes[1][i]
    ax.hist(residuals, bins=8, edgecolor="black", color="steelblue")
    ax.axvline(0, color="r", linestyle="--")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title(f"{crop} Residual Distribution\n(mean={residuals.mean():.1f})")
plt.suptitle("Residual Analysis (LOOCV)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eval_residuals.png", dpi=150, bbox_inches="tight")
plt.show()

# Feature importance (Ridge coefficients, unscaled)
fig, axes = plt.subplots(1, n_crops, figsize=(6 * n_crops, 6), squeeze=False)
for i, (crop, res) in enumerate(results.items()):
    ax = axes[0][i]
    # Extract Ridge coefficients from the fitted model
    ridge = res["model"].regressor_.named_steps["ridge"]
    scaler = res["model"].regressor_.named_steps["scaler"]
    coefs = ridge.coef_
    # Scale coefficients back to original units for interpretability
    coefs_unscaled = coefs / scaler.scale_
    
    sorted_idx = np.argsort(np.abs(coefs_unscaled))[::-1]
    ax.barh(range(len(feature_cols)), coefs_unscaled[sorted_idx], color="steelblue")
    ax.set_yticks(range(len(feature_cols)))
    ax.set_yticklabels([res["feature_cols"][j] for j in sorted_idx], fontsize=8)
    ax.set_xlabel("Coefficient (unscaled)")
    ax.set_title(f"{crop}")
    ax.invert_yaxis()
plt.suptitle("Feature Importance (Ridge Coefficients)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/eval_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nEvaluation plots saved to {OUTPUT_DIR}/")

# COMMAND ----------

# DBTITLE 1,Limitations and next steps
# MAGIC %md
# MAGIC ## 9. Limitations and Next Steps
# MAGIC
# MAGIC ### Limitations of this pipeline
# MAGIC
# MAGIC 1. **Small sample size** — 8 years × 4 crops = 32 rows. Each crop model has only ~8 training samples. ML with <10 samples is exploratory, not predictive. Expand to more counties and years for production use.
# MAGIC 2. **County-level aggregation** — NASS yields are county-level, but NDVI is computed for a ~5km AOI within the county. The NDVI may not represent all cropland. A crop mask (CDL) would improve this.
# MAGIC 3. **Single growing season** — We use March–November for all crops. Winter wheat actually grows October–June, so its features don't capture the full season.
# MAGIC 4. **No crop mask** — NDVI includes all land cover (rangeland, water, urban) in the AOI, not just crop fields. Using the USDA NASS Cropland Data Layer (CDL) to mask non-crop pixels would improve NDVI accuracy.
# MAGIC 5. **No field-level ground truth** — County-level yields limit field-level prediction. Field-level yield data (combine yield monitors, crop insurance, farmer-reported) would enable much more precise models.
# MAGIC 6. **Soil features are constant** — Within one county, soil features have no variation and provide no predictive power. They become valuable in a multi-county model.
# MAGIC
# MAGIC ### Next steps to improve the pipeline
# MAGIC
# MAGIC 1. **Expand spatial scope** — Include multiple Texas counties for more training data and spatial variation in soil features.
# MAGIC 2. **Crop-specific growing seasons** — Use October–June for winter wheat, March–October for summer crops.
# MAGIC 3. **Apply crop masks** — Use USDA NASS CDL (30m crop type classification) to compute NDVI only over cropland pixels.
# MAGIC 4. **Add more features** — Topography (USGS DEM), drought indices (Palmer DDI), planting dates, irrigation status, fertilizer application.
# MAGIC 5. **Field-level yield data** — Partner with farmers or use RMA crop insurance data for field-level ground truth.
# MAGIC 6. **Advanced models** — With more data, try gradient boosting (XGBoost/LightGBM), time series models (ARIMA with exogenous variables), or deep learning (LSTMs for NDVI time series).
# MAGIC 7. **Operationalize** — Schedule the notebook as a weekly Lakeflow Job to update predictions with new Sentinel-2 imagery throughout the growing season.