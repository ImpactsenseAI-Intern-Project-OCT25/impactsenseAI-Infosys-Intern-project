"""
data_preprocessing_pipeline.py

End-to-end data preprocessing and feature engineering pipeline
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the earthquakes dataset."""
    df = pd.read_csv(csv_path)
    print("Sample Data:")
    print(df.sample(10))
    print("\n" + "=" * 80 + "\n")

    print("Dataset Information:")
    df.info()
    print("\n" + "=" * 80 + "\n")

    print(f"Dataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\n" + "=" * 80 + "\n")

    print("Column Names:")
    print(df.columns.tolist())
    print("\n" + "=" * 80 + "\n")

    return df


def analyze_missing_values(df: pd.DataFrame) -> pd.Series:
    """Analyze missing values and return percentage of missing by column."""
    print("Missing Value Analysis:")
    print("-" * 80)

    missing_count = df.isnull().sum()
    print("Missing Value Counts:")
    print(missing_count[missing_count > 0])
    print()

    missing_percent = round(df.isnull().sum() * 100 / len(df), 2)
    print("Missing Value Percentage:")
    print(missing_percent[missing_percent > 0])
    print("\n" + "=" * 80 + "\n")

    return missing_percent


def handle_missing_values(df: pd.DataFrame, missing_percent: pd.Series) -> pd.DataFrame:
    """Handle missing values with column dropping and regression imputation."""
    print("Handling Missing Values:")
    print("-" * 80)

    # Step 1: Drop columns with >30% missing values
    threshold = 30
    cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
    print(f"Columns dropped (>{threshold}% missing): {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
    print(f"Shape after dropping columns: {df.shape}\n")

    # Step 2: Drop ID-like and irrelevant columns
    id_like_cols = [
        "Unnamed: 0",
        "id",
        "net",
        "locationSource",
        "magSource",
        "place",
        "updated",
    ]
    existing_id_cols = [col for col in id_like_cols if col in df.columns]
    print(f"Dropping ID-like/irrelevant columns: {existing_id_cols}")
    df.drop(columns=existing_id_cols, inplace=True, errors="ignore")
    print(f"Shape after dropping ID columns: {df.shape}\n")

    # Step 3: Drop rows with missing 'depth' values
    if "depth" in df.columns:
        initial_rows = len(df)
        df = df.dropna(subset=["depth"])
        print(f"Rows dropped due to missing depth: {initial_rows - len(df)}")
        print(f"Shape after dropping missing depth rows: {df.shape}\n")
    else:
        print("Column 'depth' not found; skipping depth-based row dropping.\n")

    # Step 4: Regression-based imputation for 'rms' column
    if "rms" in df.columns and df["rms"].isnull().sum() > 0:
        print("Applying regression-based imputation for 'rms'...")

        predictors = ["mag", "depth", "nst", "gap", "dmin"]
        available_features = [f for f in predictors if f in df.columns]

        df_known = df[df["rms"].notnull()].dropna(subset=available_features)
        df_missing = df[df["rms"].isnull()].dropna(subset=available_features)

        if not df_known.empty and not df_missing.empty:
            X_train = df_known[available_features]
            y_train = df_known["rms"]

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_missing = df_missing[available_features]
            df.loc[
                df["rms"].isnull() & df.index.isin(df_missing.index), "rms"
            ] = model.predict(X_missing)

            print(f"RMS missing values after imputation: {df['rms'].isnull().sum()}")
        else:
            print("Insufficient data for regression imputation")
    else:
        print("No 'rms' column or no missing values in 'rms'")

    print("\n" + "=" * 80 + "\n")

    print("Remaining Missing Values:")
    remaining_missing = round(df.isnull().sum() * 100 / len(df), 2)
    print(remaining_missing[remaining_missing > 0])
    print("\n" + "=" * 80 + "\n")

    return df


def convert_to_mw(mag_value: float, mag_type: str) -> float:
    """Convert various earthquake magnitude types to moment magnitude (Mw)."""
    if pd.isna(mag_value) or pd.isna(mag_type):
        return np.nan

    mag_type = str(mag_type).lower()

    # Moment Magnitude already
    if mag_type.startswith("mw"):
        return mag_value
    # Surface Wave Magnitude
    if mag_type == "ms":
        return 1.05 * mag_value - 0.2
    # Body Wave Magnitude
    if mag_type == "mb":
        if mag_value > 6.5:
            return 6.5 + (mag_value - 6.5) * 1.5
        return 0.67 * mag_value + 3.2
    # Local Magnitude
    if mag_type == "ml":
        return 1.2 * mag_value - 1.0

    # Unknown magnitude type
    return np.nan


def calculate_damage_potential_hazus(magnitude: float, depth: float) -> float:
    """Calculate earthquake damage potential using an empirical HAZUS-style formula."""
    if pd.isna(magnitude) or pd.isna(depth):
        return np.nan
    actual_depth = max(abs(depth), 1.0)
    log_pga = magnitude - 3.5 * np.log10(actual_depth + 7) + 1.8
    pga = 10 ** log_pga
    damage_potential = max(0.0, 2.5 * np.log10(pga + 0.01) + 7.5)
    return damage_potential


def feature_engineering(
    df: pd.DataFrame, urban_shp_path: str
) -> pd.DataFrame:
    """Perform temporal, magnitude, damage potential and urbanity feature engineering."""
    print("Feature Engineering:")
    print("-" * 80)

    # Step 1: Temporal feature extraction from 'time' column
    if "time" in df.columns:
        print("Extracting temporal features from 'time' column...")
        df["Datetime"] = pd.to_datetime(df["time"], errors="coerce")

        df["Year"] = df["Datetime"].dt.year
        df["Month"] = df["Datetime"].dt.month
        df["Day"] = df["Datetime"].dt.day
        df["Hour"] = df["Datetime"].dt.hour
        df["Minute"] = df["Datetime"].dt.minute
        df["Second"] = df["Datetime"].dt.second
        df["DayOfWeek"] = df["Datetime"].dt.dayofweek

        print(
            "Temporal features created: Year, Month, Day, Hour, Minute, Second, DayOfWeek"
        )

        df.drop(["time", "Datetime"], axis=1, inplace=True)
        print("Dropped original 'time' and 'Datetime' columns\n")
    else:
        print("Column 'time' not found; skipping temporal feature extraction.\n")

    # Step 2: Magnitude type conversion to Mw
    print("Converting earthquake magnitudes to Moment Magnitude (Mw)...")
    print("-" * 80)

    if "mag" in df.columns and "magType" in df.columns:
        df["Mw"] = df.apply(
            lambda row: convert_to_mw(row["mag"], row["magType"]), axis=1
        )
        print(
            f"✓ Moment Magnitude (Mw) created from {df['magType'].nunique()} different magnitude types"
        )
        print("\nMagnitude Type Distribution:")
        print(df["magType"].value_counts())
        print("\nSample Magnitude Conversions:")
        print(df[["mag", "magType", "Mw"]].head(10))
        print()
    else:
        print("Warning: 'mag' or 'magType' columns not found\n")

    # Step 3: Damage potential calculation
    print("Calculating earthquake damage potential using HAZUS methodology...")
    print("-" * 80)

    if "Mw" in df.columns and "depth" in df.columns:
        df["damage_potential"] = df.apply(
            lambda row: calculate_damage_potential_hazus(row["Mw"], row["depth"]),
            axis=1,
        )
        print("✓ Damage Potential Score calculated")
        print("\nDamage Potential Statistics:")
        print(df["damage_potential"].describe())
        print("\nSample Damage Potential Calculations:")
        print(df[["Mw", "depth", "damage_potential"]].head(10))
        print()
    else:
        print("Warning: 'Mw' or 'depth' columns not found\n")

    print(f"Shape after additional feature engineering: {df.shape}")
    print("\n" + "=" * 80 + "\n")

    # Step 4: Urbanity indicator creation via spatial join
    if {"longitude", "latitude"}.issubset(df.columns):
        print("Creating urbanity indicator via spatial join...")
        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )

        urban_areas = gpd.read_file(urban_shp_path)
        urban_areas = urban_areas.to_crs(gdf_points.crs)

        gdf_joined = gpd.sjoin(gdf_points, urban_areas, how="left", predicate="within")

        gdf_joined["urbanity_indicator"] = (
            gdf_joined["index_right"].notnull().astype(int)
        )

        print("Urbanity indicator distribution:")
        print(gdf_joined["urbanity_indicator"].value_counts())

        # Risk score computation
        def compute_risk_score(row):
            mw = row["Mw"]
            depth_val = row["depth"]
            if pd.isna(mw) or pd.isna(depth_val):
                return np.nan
            depth_val = max(depth_val, 1.0)
            urbanity = row["urbanity_indicator"]

            if urbanity == 1:
                return (mw ** 2) * (1.0 / depth_val) * 1.5
            return mw * (1.0 / depth_val)

        gdf_joined["risk_score"] = gdf_joined.apply(compute_risk_score, axis=1)

        # Risk category
        def assign_risk_category(score):
            if pd.isna(score):
                return np.nan
            if score >= 7.5:
                return "Very High"
            if score >= 5.0:
                return "High"
            if score >= 2.5:
                return "Medium"
            return "Low"

        gdf_joined["risk_category"] = gdf_joined["risk_score"].apply(
            assign_risk_category
        )

        print(
            gdf_joined[
                ["Mw", "depth", "urbanity_indicator", "risk_score", "risk_category"]
            ].head(10)
        )
        print("\nRisk Category Distribution:")
        print(gdf_joined["risk_category"].value_counts())

        # Add back to base df (drop geometry and spatial join columns)
        df["urbanity_indicator"] = gdf_joined["urbanity_indicator"].values
        df["risk_score"] = gdf_joined["risk_score"].values
        df["risk_category"] = gdf_joined["risk_category"].values
    else:
        print("Missing 'longitude' or 'latitude' columns; skipping urbanity features.")

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale continuous features and decade feature."""
    print("Feature Scaling:")
    print("-" * 80)

    std_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()

    continuous_features = ["depth", "rms", "Mw", "damage_potential"]
    existing_continuous = [c for c in continuous_features if c in df.columns]

    if existing_continuous:
        df[existing_continuous] = std_scaler.fit_transform(df[existing_continuous])
        print(f"Standard scaled features: {existing_continuous}")
    else:
        print("No continuous features found for standard scaling.")

    if "decade" in df.columns:
        df[["decade"]] = min_max_scaler.fit_transform(df[["decade"]])
        print("MinMax scaled 'decade' feature.")
    else:
        print("Column 'decade' not found; skipping decade scaling.")

    print()
    return df


def finalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Finalize DataFrame: drop intermediate columns and nulls."""
    print("Finalizing DataFrame:")
    print("-" * 80)

    # Drop nulls in 'risk_score' and 'Mw' if present
    subset_cols = [c for c in ["risk_score", "Mw"] if c in df.columns]
    if subset_cols:
        df = df.dropna(subset=subset_cols)
        print(f"Dropped rows with nulls in {subset_cols}.")

    # Create decade feature from Year if available
    if "Year" in df.columns:
        df["decade"] = (df["Year"] // 10) * 10
        df = df.drop(columns=["Year"], errors="ignore")
        print("Created 'decade' feature and dropped 'Year'.")

    # Drop columns as in notebook
    cols_to_drop = [
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Second",
        "DayOfWeek",
        "magType",
        "type",
        "status",
        "mag",
        "risk_category",
    ]
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop, errors="ignore")
    print(f"Dropped columns: {existing_drop}")

    print("Remaining columns:")
    print(df.columns.tolist())
    print()

    return df


def run_pipeline(
    csv_path: str = "../../../data/earthquakes_data.csv",
    urban_shp_path: str = "../../../shp/ne_110m_admin_0_countries.shp",
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the processed DataFrame."""
    df = load_dataset(csv_path)
    missing_percent = analyze_missing_values(df)
    df = handle_missing_values(df, missing_percent)
    df = feature_engineering(df, urban_shp_path)
    df = finalize_dataframe(df)
    df = scale_features(df)

    print("Final DataFrame preview:")
    print(df.head())
    print("Final DataFrame info:")
    print(df.describe(include="all"))
    return df


if __name__ == "__main__":
    # Example run with default paths
    processed_df = run_pipeline()
