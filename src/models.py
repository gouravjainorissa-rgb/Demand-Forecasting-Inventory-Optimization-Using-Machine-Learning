"""
Model Training, Validation, and Evaluation Pipeline (Week 3).
Trains and tunes multiple demand forecasting models:
1. Ridge Regression (regularized linear baseline)
2. Random Forest Regressor (non-linear ensemble)
3. HistGradientBoostingRegressor (fast, lightweight GBDT)
Evaluates on chronological test set and compares with Week 2 baselines.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
    PRODUCT_COL,
    STORE_COL,
    DATE_COL
)
from src.baselines import (
    calculate_metrics,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    RollingAverageForecaster
)

# Feature subsets
NUMERICAL_FEATURES = [
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_min_7", "rolling_max_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28",
    "momentum_7_28",
    "sin_dow", "cos_dow", "sin_month", "cos_month",
    "is_weekend", "is_holiday",
    "discount_ratio", "is_promoted", "lag_1_promo",
    "lag_1_stockout", "stockout_last_7"
]

CATEGORICAL_FEATURES = ["product_id", "store_id"]

def build_preprocessor() -> ColumnTransformer:
    """Build unified feature preprocessing pipeline."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor

def train_and_evaluate_models() -> tuple[dict, pd.DataFrame, object]:
    """
    Train candidate models on train_set, tune on val_set, evaluate on test_set.
    Returns comparison metrics, test predictions dataframe, and champion model.
    """
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train_set.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val_set.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test_set.csv")
    
    train_df[DATE_COL] = pd.to_datetime(train_df[DATE_COL])
    val_df[DATE_COL] = pd.to_datetime(val_df[DATE_COL])
    test_df[DATE_COL] = pd.to_datetime(test_df[DATE_COL])
    
    y_train = train_df[TARGET_COL].values
    y_val = val_df[TARGET_COL].values
    y_test = test_df[TARGET_COL].values
    
    # 1. Evaluate Baselines on Test Set for direct comparison
    comparison_results = []
    
    naive = NaiveForecaster()
    s_naive = SeasonalNaiveForecaster()
    rolling = RollingAverageForecaster()
    
    comparison_results.append(calculate_metrics(y_test, naive.predict(test_df), "Baseline: Naive (Lag 1)"))
    comparison_results.append(calculate_metrics(y_test, s_naive.predict(test_df), "Baseline: Seasonal-Naive (Lag 7)"))
    comparison_results.append(calculate_metrics(y_test, rolling.predict(test_df), "Baseline: Rolling 7-Day Mean"))
    
    # 2. Candidate ML Models Setup
    preprocessor = build_preprocessor()
    
    models = {
        "Ridge Regression": Pipeline([
            ("prep", preprocessor),
            ("reg", Ridge(alpha=10.0, random_state=RANDOM_SEED))
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocessor),
            ("reg", RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_leaf=4,
                n_jobs=1,
                random_state=RANDOM_SEED
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor),
            ("reg", HistGradientBoostingRegressor(
                max_iter=100,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=15,
                l2_regularization=0.5,
                random_state=RANDOM_SEED
            ))
        ])
    }
    
    val_scores = {}
    fitted_models = {}
    
    for name, pipeline in models.items():
        print(f"Fitting model: {name}...")
        pipeline.fit(train_df, y_train)
        fitted_models[name] = pipeline
        
        # Validation evaluation
        val_pred = pipeline.predict(val_df)
        val_metric = calculate_metrics(y_val, val_pred, name)
        val_scores[name] = val_metric
        print(f"  [Validation] {name}: MAE = {val_metric['mae']}, RMSE = {val_metric['rmse']}, WAPE = {val_metric['wape_pct']}%")
        
        # Test evaluation
        test_pred = pipeline.predict(test_df)
        test_metric = calculate_metrics(y_test, test_pred, f"ML: {name}")
        comparison_results.append(test_metric)
        
    # Select Champion Model based on lowest Validation WAPE
    champion_name = min(val_scores, key=lambda k: val_scores[k]["wape_pct"])
    champion_pipeline = fitted_models[champion_name]
    print(f"\nChampion Model Selected: {champion_name} (Validation WAPE: {val_scores[champion_name]['wape_pct']}%)")
    
    # Save champion model
    champion_path = MODELS_DIR / "champion_demand_model.joblib"
    joblib.dump(champion_pipeline, champion_path)
    print(f"Saved champion model to: {champion_path}")
    
    # Generate test prediction dataframe with champion predictions
    test_results_df = test_df.copy()
    test_results_df["forecast_demand"] = np.maximum(0.0, champion_pipeline.predict(test_df))
    test_results_df["forecast_error"] = test_results_df["forecast_demand"] - test_results_df[TARGET_COL]
    test_results_df["abs_forecast_error"] = np.abs(test_results_df["forecast_error"])
    
    # Save forecast outputs
    forecast_out_path = PROCESSED_DATA_DIR / "forecast_output.csv"
    test_results_df.to_csv(forecast_out_path, index=False)
    print(f"Saved test forecast outputs to: {forecast_out_path}")
    
    comparison_df = pd.DataFrame(comparison_results)
    return comparison_df, test_results_df, champion_pipeline

if __name__ == "__main__":
    train_and_evaluate_models()
