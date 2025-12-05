import os
import joblib
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load preprocessed data with relevant features
df = pd.read_csv("data/earthquakes_data_preprocessed.csv")

# Select features matching XGBoost training snippet
features = [
    "depth",
    "rms",
    "Mw",
    "damage_potential",
    "urbanity_indicator",
    "decade"
]

# Prepare feature matrix and target vector
X = df[features]
y = df["risk_score"]  # Use risk_score as target from preprocessing pipeline

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost regressor with optimized hyperparameters
model = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.3f}, R²: {r2:.3f}")

# Save model and feature list
os.makedirs("models", exist_ok=True)
joblib.dump(
    {"model": model, "features": features},
    "models/xgb_risk_model.pkl"
)
print("Saved XGBoost model and feature list")
