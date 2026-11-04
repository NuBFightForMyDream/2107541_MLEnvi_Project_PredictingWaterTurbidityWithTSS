# Predicting Lake Turbidity (TSS) from Landsat Imagery

**2107541 Machine Learning for Environmental Applications** · Individual Final Project · AY2026 Semester 1
Chulalongkorn University, Faculty of Engineering

**Contributor:** Chatrphol Ovanonchai (6730084521)

**Live visualizer:** [Click Here](https://2107541-final-project-waterturbidity.netlify.app/)

**Disclaimer from Contributor** 
> Most of this project were coded by myself , Claude is just assistant to get my work faster and more efficient , not a main contributor.
---

## Problem

Turbidity, measured as **Total Suspended Solids (TSS)**, is a key indicator of lake water quality. It reflects sediment, erosion and runoff. Field sampling is expensive, so it covers only a few lakes on a few dates.

Landsat images almost every lake about every 16 days. If TSS can be predicted from satellite reflectance, water quality can be monitored at a much larger scale.

**Goal:** predict TSS from Landsat surface reflectance, and check that the model works on **lakes it has never seen** (train/test split by lake, `SiteID`).

## Dataset

[**AquaSat**](https://figshare.com/articles/dataset/AquaSat/8139383) (Ross et al., 2019, *Water Resources Research*, CC BY 4.0) pairs in-situ water quality measurements with Landsat 5/7/8 surface reflectance taken within ±1 day.

| | Rows |
|---|---|
| Raw matchups | 603,432 |
| After cleaning | 128,603 |

**Cleaning steps**
1. Keep rows with a valid TSS value (> 0).
2. Keep valid reflectance (0 < band ≤ 10000).
3. Keep clear-water pixels only (`qa` = 68 for Landsat 5/7, 324 for Landsat 8).
4. Drop pixels near the shoreline (`pwater` ≥ 50).

The raw CSV (291 MB) is over GitHub's 100 MB file limit. The repo stores a gzip copy, [`Resources/Dataset/aquasat.csv.gz`](Resources/Dataset/aquasat.csv.gz) (74 MB). Unzipped, it is byte-identical to the original (same MD5 as Figshare). The column descriptions are in [`aquasat_column_dictionary.pdf`](Resources/Dataset/aquasat_column_dictionary.pdf).

## Method

**Target:** `log(1 + TSS)`. TSS is strongly right-skewed.

**Features (16)**
- 6 Landsat bands: blue, green, red, NIR, SWIR1, SWIR2 (scaled to reflectance)
- Water indices: `NDTI`, `NDWI`, `NDSSI`
- Band ratios: `R/G`, `R/B`, `G/B`, `NIR/R`
- Brightness: `log(R + G + B)`
- Season: `sin` / `cos` of day-of-year

**Models**
- Linear Regression (baseline, with `StandardScaler`)
- Random Forest
- XGBoost
- Multi-Layer Perceptron (64 → 32, ReLU, Adam, early stopping)

**Validation without data leakage.** The same lake appears many times in the data. A random row split would let the model "remember" lakes it is tested on. All splits are therefore grouped by `SiteID`:
- Train/test: `GroupShuffleSplit` (80/20 by lake)
- Cross-validation: `GroupKFold` (5 folds)
- Tuning: `RandomizedSearchCV` with `GroupKFold`. The test set is used **only once**, at the end.

## Results

All scores are on `log(1 + TSS)`, for **lakes not seen during training**.

| Model | CV R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | 0.372 | 0.423 | 0.862 | 0.646 |
| Random Forest | 0.453 | 0.491 | 0.809 | 0.598 |
| XGBoost | 0.429 | 0.467 | 0.828 | 0.618 |
| MLP | 0.427 | 0.468 | 0.827 | 0.621 |
| **Random Forest (tuned)** | **0.465** | **0.503** | **0.800** | **0.586** |

Tuning also raised XGBoost's CV R² from 0.429 to 0.464.

**Findings**
- Non-linear models beat the linear baseline (0.423 → 0.503), so the reflectance–TSS relationship is non-linear.
- `red`, `R/B`, `NDTI` and `NIR` drive the predictions (feature importance and SHAP). This matches water optics: suspended sediment scatters more red and NIR light.
- All models compress the range. Low TSS is over-predicted and high TSS is under-predicted. Tuning helped only slightly (+0.01 R²), which points to the features, not the hyperparameters, as the main limit.
- A test MAE of 0.586 in log scale means predictions are typically off by about **1.8×** the measured TSS. The model is useful for **relative** turbidity (clear vs. turbid lakes) at large scale, not as a replacement for field measurements.

**Room for improvement:** add non-satellite features (lake depth, watershed land use, recent rainfall), include sensor type (Landsat 5/7/8), weight high-TSS samples more, and run a wider hyperparameter search.

## Web visualizer

[`Visualizer/`](Visualizer/) is a static, interactive explainer built with D3. It shows the study sites on a US map, the model comparison, predicted vs. measured values for unseen lakes, and the most important features. It needs no build step: open `Visualizer/index.html` in a browser, or host the folder on any static host.

`data.js` is generated from the same pipeline as the notebook:

```bash
python3 Visualizer/export_data.py   # takes about 1–2 minutes, rewrites Visualizer/data.js
```

## How to run

```bash
git clone https://github.com/NuBFightForMyDream/2107541_MLEnvi_Project_PredictingWaterTurbidityWithTSS.git
cd 2107541_MLEnvi_Project_PredictingWaterTurbidityWithTSS
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn
jupyter notebook ENV541_Project_Predicting_Water_Turbidity_with_TSS.ipynb
```

The notebook reads the local `aquasat.csv.gz`. If the file is missing, for example on Colab, it reads the file from this repo on GitHub instead.

## Repository structure

```
├── ENV541_Project_Predicting_Water_Turbidity_with_TSS.ipynb   # main notebook (Parts 0–VII)
├── Visualizer/
│   ├── index.html        # interactive web explainer
│   ├── data.js           # model results exported for the web
│   ├── us_map.js         # US map geometry
│   └── export_data.py    # regenerates data.js
└── Resources/
    ├── Dataset/          # aquasat.csv.gz + column dictionary
    ├── Assignment Brief/
    ├── Research/         # background reading
    └── Claude Examples/  # AI-generated reference notebook (not the submission)
```

## Acknowledgements

The analysis, model design and write-up were done by the author. AI assistance was used for occasional guidance, and the AI-generated reference notebook is kept separately in `Resources/Claude Examples/`.

**Data:** Ross, M. R. V., Topp, S. N., Appling, A. P., Yang, X., Kuhn, C., Butman, D., Simard, M., & Pavelsky, T. M. (2019). AquaSat: A data set to enable remote sensing of water quality for inland waters. *Water Resources Research*, 55, 10012–10025. https://doi.org/10.1029/2019WR024883
