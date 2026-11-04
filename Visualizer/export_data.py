### Export results for the web visualizer (Visualizer/index.html)
## Same cleaning , features , split and models as the notebook --> writes Visualizer/data.js
## run from anywhere : python3 Visualizer/export_data.py   (~1-2 min)

import json , warnings
from pathlib import Path

import numpy as np , pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE.parent / "Resources" / "Dataset" / "aquasat_landsat_lake_water_quality.csv"

## ---------------- Part I : cleaning (same as notebook) ----------------
df = pd.read_csv(DATA_PATH)
raw_rows = len(df)

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
df_cleaned = df.drop(columns = ["chl_a" , "doc" , "secchi" , "tis" , "p_sand"])
df_cleaned = df_cleaned[(df_cleaned["tss"].notna()) & (df_cleaned["tss"] > 0)]
df_cleaned = df_cleaned[((df_cleaned[BANDS] > 0) & (df_cleaned[BANDS] <= 10000)).all(axis = 1)]
df_cleaned = df_cleaned[df_cleaned["qa"].round().isin([68, 324])]
df_cleaned = df_cleaned[df_cleaned["pwater"] >= 50]

df_cleaned["tss_log"] = np.log1p(df_cleaned["tss"])
df_cleaned[BANDS] = df_cleaned[BANDS] * 1e-4

b , g , r , n = (df_cleaned[col] for col in ["blue", "green", "red", "nir"])
df_cleaned["NDTI"] = (r - g) / (r + g)
df_cleaned["NDWI"] = (g - n) / (g + n)
df_cleaned["NDSSI"] = (b - n) / (b + n)
df_cleaned["R_G"] , df_cleaned["R_B"] = r / g , r / b
df_cleaned["G_B"] , df_cleaned["NIR_R"] = g / b , n / r
df_cleaned["log_visible"] = np.log(b + g + r)
doy = pd.to_datetime(df_cleaned["date"]).dt.dayofyear
df_cleaned["doy_sin"] = np.sin(2 * np.pi * doy / 365)
df_cleaned["doy_cos"] = np.cos(2 * np.pi * doy / 365)

FEATURES = BANDS + ["NDTI", "NDWI", "NDSSI", "R_G", "R_B", "G_B", "NIR_R", "log_visible", "doy_sin", "doy_cos"]

## ---------------- Part II : split by SiteID (same as notebook) ----------------
features = df_cleaned[FEATURES]
targets = df_cleaned["tss_log"]
groups = df_cleaned["SiteID"]

gss = GroupShuffleSplit(n_splits = 1 , train_size = 0.8 , test_size = 0.2 , random_state = 42)
train_idx , test_idx = next(gss.split(features , targets , groups))
X_train , X_test = features.iloc[train_idx] , features.iloc[test_idx]
y_train , y_test = targets.iloc[train_idx] , targets.iloc[test_idx]

## ---------------- Part III / VI : models (before + tuned best params) ----------------
models = {
    "Linear Regression" : Pipeline([("scaler" , StandardScaler()) , ("model" , LinearRegression())]) ,
    "Random Forest" : Pipeline([("model" , RandomForestRegressor(n_estimators = 200 , min_samples_leaf = 5 , n_jobs = -1 , random_state = 42 , max_features = 0.33 , max_samples = 0.5))]) ,
    "XGBoost" : Pipeline([("model" , XGBRegressor(n_estimators = 300 , max_depth = 4 , learning_rate = 0.05 , subsample = 0.8 , colsample_bytree = 0.8))]) ,
    "Neural Network (MLP)" : Pipeline([("scaler" , StandardScaler()) , ("model" , MLPRegressor(hidden_layer_sizes = (64 , 32) , activation = "relu" , solver = "adam" , learning_rate_init = 0.001 , batch_size = 256 , max_iter = 200 , alpha = 1e-4 , early_stopping = True , validation_fraction = 0.1 , n_iter_no_change = 10 , random_state = 42))]) ,
    ## best params from RandomizedSearchCV (notebook Part VI)
    "Random Forest (tuned)" : Pipeline([("model" , RandomForestRegressor(n_estimators = 200 , min_samples_leaf = 1 , n_jobs = -1 , random_state = 42 , max_features = 0.5 , max_depth = None , max_samples = 0.5))]) ,
}

predictions = {}
for name , pipe in models.items() :
    print("training :" , name)
    pipe.fit(X_train , y_train)
    predictions[name] = pipe.predict(X_test)

## CV results copied from notebook outputs (Part V & VI , GroupKFold 5 folds) --> not re-run here to save time
CV_RESULTS = {
    "Linear Regression"     : {"train_r2" : 0.383 , "cv_r2" : 0.372 , "cv_r2_std" : 0.032} ,
    "Random Forest"         : {"train_r2" : 0.656 , "cv_r2" : 0.453 , "cv_r2_std" : 0.026} ,
    "XGBoost"               : {"train_r2" : 0.462 , "cv_r2" : 0.429 , "cv_r2_std" : 0.027} ,
    "Neural Network (MLP)"  : {"train_r2" : 0.451 , "cv_r2" : 0.427 , "cv_r2_std" : 0.026} ,
    "Random Forest (tuned)" : {"train_r2" : 0.805 , "cv_r2" : 0.465 , "cv_r2_std" : 0.026} ,
}

y_true_tss = np.expm1(y_test.values)
metrics = []
for name , y_pred in predictions.items() :
    pred_tss = np.expm1(y_pred)
    ratio = np.maximum(pred_tss , 1e-3) / y_true_tss
    metrics.append({
        "model" : name ,
        "r2" : round(float(r2_score(y_test , y_pred)) , 3) ,
        "rmse" : round(float(np.sqrt(mean_squared_error(y_test , y_pred))) , 3) ,
        "mae" : round(float(mean_absolute_error(y_test , y_pred)) , 3) ,
        "within_2x" : round(float(np.mean((ratio >= 0.5) & (ratio <= 2))) , 3) ,
        "median_factor" : round(float(np.exp(np.median(np.abs(np.log(ratio))))) , 2) ,
        **CV_RESULTS[name] ,
    })

## ---------------- sample of test points for the scatter ----------------
rng = np.random.default_rng(42)
sample = rng.choice(len(y_test) , size = 2500 , replace = False)
scatter = {
    "real" : [round(float(v) , 2) for v in y_true_tss[sample]] ,
    "pred" : {name : [round(float(v) , 2) for v in np.expm1(p[sample])] for name , p in predictions.items()} ,
}

## binned trend : median prediction inside each band of real TSS (shows range compression)
bin_edges = np.quantile(y_true_tss , np.linspace(0 , 1 , 13))
bin_id = np.clip(np.searchsorted(bin_edges , y_true_tss , side = "right") - 1 , 0 , 11)
trend = {}
for name , p in predictions.items() :
    pred_tss = np.expm1(p)
    trend[name] = [[round(float(np.median(y_true_tss[bin_id == k])) , 2) , round(float(np.median(pred_tss[bin_id == k])) , 2)] for k in range(12)]

## ---------------- feature importance (tuned RF) + direction (XGB SHAP , same as notebook) ----------------
rf_tuned = models["Random Forest (tuned)"].named_steps["model"]
X_shap = X_test.sample(2000 , random_state = 42)
shap_values = shap.TreeExplainer(models["XGBoost"].named_steps["model"])(X_shap).values
importance = []
for i , feat in enumerate(FEATURES) :
    direction = pd.Series(X_shap[feat].values).corr(pd.Series(shap_values[: , i]) , method = "spearman")
    importance.append({"feature" : feat ,
                       "importance" : round(float(rf_tuned.feature_importances_[i]) , 4) ,
                       "direction" : round(float(direction) , 2)})
importance.sort(key = lambda d : -d["importance"])

## ---------------- spectrum : clear vs turbid water ----------------
q25 , q75 = df_cleaned["tss"].quantile([0.25 , 0.75])
spectrum = {
    "clear_cutoff" : round(float(q25) , 1) , "turbid_cutoff" : round(float(q75) , 1) ,
    "clear" : [round(float(df_cleaned.loc[df_cleaned["tss"] <= q25 , band].median()) , 4) for band in BANDS] ,
    "turbid" : [round(float(df_cleaned.loc[df_cleaned["tss"] >= q75 , band].median()) , 4) for band in BANDS] ,
}

## ---------------- sites for the map ----------------
test_sites = set(groups.iloc[test_idx].unique())
site_df = df_cleaned.groupby("SiteID").agg(lat = ("lat" , "first") , lon = ("long" , "first") ,
                                         n = ("tss" , "size") , tss = ("tss" , "median")).reset_index()
pred_by_site = pd.DataFrame({"SiteID" : groups.iloc[test_idx].values ,
                             "pred" : np.expm1(predictions["Random Forest (tuned)"])}).groupby("SiteID")["pred"].median()
site_df["pred"] = site_df["SiteID"].map(pred_by_site)
sites = [[round(float(s.lat) , 3) , round(float(s.lon) , 3) , int(s.n) , round(float(s.tss) , 1) ,
          1 if s.SiteID in test_sites else 0 ,
          None if pd.isna(s.pred) else round(float(s.pred) , 1)] for s in site_df.itertuples()]

years = pd.to_datetime(df_cleaned["date"]).dt.year
summary = {
    "raw_rows" : raw_rows , "clean_rows" : len(df_cleaned) ,
    "n_sites" : int(groups.nunique()) , "n_test_sites" : len(test_sites) ,
    "n_train_rows" : len(X_train) , "n_test_rows" : len(X_test) ,
    "year_min" : int(years.min()) , "year_max" : int(years.max()) ,
    "tss_median" : round(float(df_cleaned["tss"].median()) , 1) ,
    "tss_p10" : round(float(df_cleaned["tss"].quantile(0.1)) , 1) ,
    "tss_p90" : round(float(df_cleaned["tss"].quantile(0.9)) , 1) ,
}

out = {"summary" : summary , "metrics" : metrics , "scatter" : scatter , "trend" : trend ,
       "importance" : importance , "spectrum" : spectrum , "bands" : BANDS ,
       "sites_columns" : ["lat" , "lon" , "n_samples" , "median_tss" , "is_test" , "median_pred"] , "sites" : sites}

with open(HERE / "data.js" , "w") as f :
    f.write("// generated by export_data.py\nwindow.TSS_DATA = ")
    json.dump(out , f , separators = (",", ":"))
    f.write(";\n")

print(json.dumps(summary , indent = 1))
for m in metrics : print(m)
print("wrote" , HERE / "data.js")
