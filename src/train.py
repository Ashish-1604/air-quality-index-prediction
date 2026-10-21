"""Train and evaluate XGBoost on Kaggle's measured city-day AQI data."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

try:
    from .preprocessing import make_preprocessor
except ImportError:
    from preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]


def main():
    data_path = ROOT / "data" / "kaggle_city_day.csv"
    if not data_path.exists():
        import runpy
        runpy.run_path(str(ROOT / "data" / "download_kaggle_data.py"), run_name="__main__")
    data = pd.read_csv(data_path)
    # Keep rows with observed, in-range AQI labels. The Kaggle source has
    # occasional AQI values above the CPCB scale's 500-point upper bound.
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")
    data = data.dropna(subset=["Date", "AQI"])
    data = data[data["AQI"].between(0, 500)].copy()
    data = data.sort_values("Date")
    # A chronological holdout tests later dates and avoids future-to-past leakage.
    unique_dates = np.sort(data["Date"].unique())
    cutoff = unique_dates[int(len(unique_dates) * 0.8)]
    train_rows = data["Date"] < cutoff
    X_train, y_train = data.loc[train_rows, FEATURES], data.loc[train_rows, "AQI"]
    X_test, y_test = data.loc[~train_rows, FEATURES], data.loc[~train_rows, "AQI"]
    print("Dataset: Kaggle Air Quality Data in India (city_day.csv)")
    print(f"Rows after valid-AQI filtering: {len(data):,}")
    print(f"Chronological cutoff: {pd.Timestamp(cutoff).date()}")
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    pipeline = Pipeline([("preprocessing", make_preprocessor()), ("regressor", XGBRegressor(objective="reg:squarederror", eval_metric="rmse", random_state=42, n_jobs=1, tree_method="hist"))])
    search = GridSearchCV(pipeline, {"regressor__n_estimators": [120, 220], "regressor__max_depth": [3, 5], "regressor__learning_rate": [0.05, 0.1], "regressor__subsample": [0.9]}, cv=KFold(3, shuffle=True, random_state=42), scoring="neg_root_mean_squared_error", n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"Best parameters: {search.best_params_}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mean_absolute_error(y_test, pred):.3f}")
    print(f"R2:   {r2_score(y_test, pred):.4f}")
    model_path = ROOT / "models" / "aqi_xgboost_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": search.best_estimator_, "features": FEATURES,
                 "dataset": "Kaggle Air Quality Data in India / city_day.csv",
                 "cutoff_date": str(pd.Timestamp(cutoff).date())}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
