import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sklearn.model_selection import train_test_split

# =============================================================================
# Custom Transformers from your data_preprocessing_pipeline.py
# =============================================================================

class MissingValueAnalyzer(BaseEstimator, TransformerMixin):
    """Analyze and handle missing values like in your pipeline"""
    def __init__(self, missing_threshold=30):
        self.missing_threshold = missing_threshold
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Drop columns with >30% missing values
        missing_percent = X.isnull().sum() * 100 / len(X)
        cols_to_drop = missing_percent[missing_percent > self.missing_threshold].index
        X.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        
        # Drop ID-like columns
        id_cols = ['Unnamed: 0', 'id', 'net', 'locationSource', 'magSource', 'place', 'updated']
        existing_id_cols = [col for col in id_cols if col in X.columns]
        X.drop(columns=existing_id_cols, inplace=True, errors='ignore')
        
        # Drop rows with missing depth
        if 'depth' in X.columns:
            X = X.dropna(subset=['depth'])
            
        return X

class RMSImputer(BaseEstimator, TransformerMixin):
    """Regression-based imputation for RMS column"""
    def fit(self, X, y=None):
        if 'rms' in X.columns and X['rms'].isnull().sum() > 0:
            predictors = ['mag', 'depth', 'nst', 'gap', 'dmin']
            available_features = [f for f in predictors if f in X.columns]
            df_known = X[X['rms'].notnull()].dropna(subset=available_features)
            
            if not df_known.empty and len(available_features) > 0:
                X_train = df_known[available_features]
                y_train = df_known['rms']
                self.imputer = LinearRegression().fit(X_train, y_train)
                self.features = available_features
        return self
    
    def transform(self, X):
        X = X.copy()
        if hasattr(self, 'imputer') and 'rms' in X.columns and X['rms'].isnull().sum() > 0:
            df_missing = X[X['rms'].isnull()].dropna(subset=self.features)
            if not df_missing.empty:
                X_pred = df_missing[self.features]
                predictions = self.imputer.predict(X_pred)
                mask = X['rms'].isnull()
                X.loc[mask & X.index.isin(df_missing.index), 'rms'] = predictions
        return X

class MagnitudeConverter(BaseEstimator, TransformerMixin):
    """Convert magnitude types to Moment Magnitude (Mw)"""
    def convert_to_mw(self, mag_value, mag_type):
        if pd.isna(mag_value) or pd.isna(mag_type):
            return np.nan
        mag_type = str(mag_type).lower()
        
        if mag_type.startswith('mw'):
            return mag_value
        elif mag_type == 'ms':
            return 1.05 * mag_value - 0.2
        elif mag_type == 'mb':
            if mag_value > 6.5:
                return 6.5 + (mag_value - 6.5) * 1.5
            return 0.67 * mag_value + 3.2
        elif mag_type == 'ml':
            return 1.2 * mag_value - 1.0
        return np.nan
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'mag' in X.columns and 'magType' in X.columns:
            X['Mw'] = X.apply(lambda row: self.convert_to_mw(row['mag'], row['magType']), axis=1)
        return X

class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from time column"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'time' in X.columns:
            X['Datetime'] = pd.to_datetime(X['time'], errors='coerce')
            X['Year'] = X['Datetime'].dt.year
            X['Month'] = X['Datetime'].dt.month
            X['Day'] = X['Datetime'].dt.day
            X['Hour'] = X['Datetime'].dt.hour
            X['DayOfWeek'] = X['Datetime'].dt.dayofweek
            X.drop(['time', 'Datetime'], axis=1, inplace=True)
        return X

class DamagePotentialCalculator(BaseEstimator, TransformerMixin):
    """Calculate HAZUS-style damage potential"""
    def calculate_damage_potential_hazus(self, magnitude, depth):
        if pd.isna(magnitude) or pd.isna(depth):
            return np.nan
        actual_depth = max(abs(depth), 1.0)
        log_pga = magnitude - 3.5 * np.log10(actual_depth + 7) + 1.8
        pga = 10 ** log_pga
        return max(0.0, 2.5 * np.log10(pga + 0.01) + 7.5)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'Mw' in X.columns and 'depth' in X.columns:
            X['damage_potential'] = X.apply(
                lambda row: self.calculate_damage_potential_hazus(row['Mw'], row['depth']), axis=1
            )
        return X

class UrbanityFeatureCreator(BaseEstimator, TransformerMixin):
    """Simplified urbanity indicator (no shapefile dependency for pipeline)"""
    def __init__(self, urban_threshold_lat=60):
        self.urban_threshold_lat = urban_threshold_lat
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'latitude' in X.columns and 'longitude' in X.columns:
            # Simplified urbanity: events within ±60° latitude are urban
            X['urbanity_indicator'] = (abs(X['latitude']) < self.urban_threshold_lat).astype(int)
            
            # Risk score calculation
            def compute_risk_score(row):
                if pd.isna(row['Mw']) or pd.isna(row['depth']):
                    return np.nan
                depth_val = max(row['depth'], 1.0)
                urbanity = row['urbanity_indicator']
                if urbanity == 1:
                    return (row['Mw'] ** 2) * (1.0 / depth_val) * 1.5
                return row['Mw'] * (1.0 / depth_val)
            
            X['risk_score'] = X.apply(compute_risk_score, axis=1)
        return X

class FinalFeatureSelector(BaseEstimator, TransformerMixin):
    """Finalize features: create decade, drop unnecessary columns, scale"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create decade feature
        if 'Year' in X.columns:
            X['decade'] = (X['Year'] // 10) * 10
            X.drop('Year', axis=1, inplace=True, errors='ignore')
        
        # Drop intermediate/temporal columns
        cols_to_drop = ['Month', 'Day', 'Hour', 'Minute', 'Second', 'DayOfWeek', 
                       'magType', 'type', 'status', 'mag']
        for col in cols_to_drop:
            X.drop(col, axis=1, inplace=True, errors='ignore')
        
        # Scale continuous features
        continuous_features = ['depth', 'rms', 'Mw', 'damage_potential']
        existing_continuous = [c for c in continuous_features if c in X.columns]
        if existing_continuous:
            scaler = StandardScaler()
            X[existing_continuous] = scaler.fit_transform(X[existing_continuous])
        
        if 'decade' in X.columns:
            decade_scaler = MinMaxScaler()
            X['decade'] = decade_scaler.fit_transform(X[['decade']])
        
        # Final feature order
        final_features = ['latitude','longitude','depth', 'rms', 'Mw', 'damage_potential', 'urbanity_indicator', 'decade', 'risk_score']
        return X[[col for col in final_features if col in X.columns]]

# =============================================================================
# Complete Pipeline
# =============================================================================

def create_complete_preprocessing_pipeline():
    """Create the full preprocessing pipeline matching your data_preprocessing_pipeline.py"""
    return Pipeline([
        ('missing_handler', MissingValueAnalyzer(missing_threshold=30)),
        ('rms_imputer', RMSImputer()),
        ('temporal_features', TemporalFeatureExtractor()),
        ('magnitude_converter', MagnitudeConverter()),
        ('damage_calculator', DamagePotentialCalculator()),
        ('urbanity_creator', UrbanityFeatureCreator()),
        ('final_selector', FinalFeatureSelector())
    ])

# Create and save pipeline
pipeline = create_complete_preprocessing_pipeline()

# Test the pipeline (uncomment to test)

# Example usage:
# df = pd.read_csv('data/earthquakes_data.csv')
# df_processed = pipeline.fit_transform(df)
# print("Pipeline shape:", df_processed.shape)
# print("Pipeline columns:", df_processed.columns.tolist())
# print("Sample data:\n", df_processed.head())


# Save pipeline
joblib.dump(pipeline, 'models/data_preprocessing_pipeline.pkl')
