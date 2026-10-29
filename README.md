# crop-yield
repo for my crop yield prediction work
Crop Yield Prediction Pipeline — Coleman County, Texas
This notebook builds a county-level crop yield prediction model by combining:

Sentinel-2 satellite NDVI (vegetation health, 10m resolution, via Microsoft Planetary Computer — no auth required)
NASA POWER weather data (temperature, precipitation, growing degree days, solar radiation — no auth required)
USDA NRCS SSURGO soil data (organic matter, water capacity, drainage — no auth required)
USDA NASS county-level yield data (the prediction target — free API key required)
Pipeline stages: data ingestion → feature engineering → EDA → model training (Ridge regression with MLflow) → evaluation.

Note: This notebook requires compute to be attached. The Sentinel-2 NDVI computation takes 5–15 minutes depending on network speed. Set USE_SAMPLE_NDVI = True in the configuration cell to skip imagery processing for a quick end-to-end test.

Important: With ~8 years of county-level data per crop, the model demonstrates the methodology and pipeline architecture. Production-quality predictions require more data — more counties, more years, or field-level yield ground truth. See the final cell for limitations and next steps.
