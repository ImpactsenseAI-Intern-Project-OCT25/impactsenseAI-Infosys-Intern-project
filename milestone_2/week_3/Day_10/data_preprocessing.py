# ============================================================================
# 1. IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import chi2_contingency


# ============================================================================
# 2. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

# Load the dataset
df = pd.read_csv('C:/Users/divyanshu/OneDrive/Desktop/Projects/ImpactSense/impactsenseAI-Infosys-Intern-project/data/earthquakes_data.csv')


# Display sample data
print("Sample Data:")
print(df.sample(10))
print("\n" + "="*80 + "\n")

# Basic information
print("Dataset Information:")
df.info()
print("\n" + "="*80 + "\n")

# Dataset shape
print(f"Dataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\n" + "="*80 + "\n")

# Column names
print("Column Names:")
print(df.columns.tolist())
print("\n" + "="*80 + "\n")


# ============================================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================================

print("Missing Value Analysis:")
print("-" * 80)

# Count missing values
missing_count = df.isnull().sum()
print("Missing Value Counts:")
print(missing_count[missing_count > 0])
print("\n")

# Calculate missing value percentage
missing_percent = round(df.isnull().sum() * 100 / len(df), 2)
print("Missing Value Percentage:")
print(missing_percent[missing_percent > 0])
print("\n" + "="*80 + "\n")


# ============================================================================
# 4. HANDLING MISSING VALUES
# ============================================================================

print("Handling Missing Values:")
print("-" * 80)

# Step 4.1: Drop columns with >30% missing values
threshold = 30
cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
print(f"Columns dropped (>{threshold}% missing): {cols_to_drop}")
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Shape after dropping columns: {df.shape}\n")

# Step 4.2: Drop ID-like and irrelevant columns
id_like_cols = ['Unnamed: 0', 'id', 'net', 'locationSource', 'magSource', 'place', 'updated']
existing_id_cols = [col for col in id_like_cols if col in df.columns]
print(f"Dropping ID-like/irrelevant columns: {existing_id_cols}")
df.drop(columns=existing_id_cols, inplace=True, errors='ignore')
print(f"Shape after dropping ID columns: {df.shape}\n")

# Step 4.3: Drop rows with missing 'depth' values
initial_rows = len(df)
df = df.dropna(subset=['depth'])
print(f"Rows dropped due to missing depth: {initial_rows - len(df)}")
print(f"Shape after dropping missing depth rows: {df.shape}\n")

# Step 4.4: Regression-based imputation for 'rms' column
if 'rms' in df.columns and df['rms'].isnull().sum() > 0:
    print("Applying regression-based imputation for 'rms'...")
    
    # Select predictor features
    predictors = ['mag', 'depth', 'nst', 'gap', 'dmin']
    available_features = [f for f in predictors if f in df.columns]
    
    # Separate known and missing RMS data
    df_known = df[df['rms'].notnull()].dropna(subset=available_features)
    df_missing = df[df['rms'].isnull()].dropna(subset=available_features)
    
    # Train and predict if sufficient data exists
    if not df_known.empty and not df_missing.empty:
        X_train = df_known[available_features]
        y_train = df_known['rms']
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Impute missing values
        X_missing = df_missing[available_features]
        df.loc[df['rms'].isnull() & df.index.isin(df_missing.index), 'rms'] = model.predict(X_missing)
        
        print(f"RMS missing values after imputation: {df['rms'].isnull().sum()}")
    else:
        print("Insufficient data for regression imputation")
else:
    print("No 'rms' column or no missing values in 'rms'")

print("\n" + "="*80 + "\n")

# Check remaining missing values
print("Remaining Missing Values:")
remaining_missing = round(df.isnull().sum() * 100 / len(df), 2)
print(remaining_missing[remaining_missing > 0])
print("\n" + "="*80 + "\n")


# ============================================================================
# 5. FEATURE ENGINEERING
# ============================================================================

print("Feature Engineering:")
print("-" * 80)

# Step 5.1: Temporal feature extraction from 'time' column
if 'time' in df.columns:
    print("Extracting temporal features from 'time' column...")
    
    # Convert to datetime
    df['Datetime'] = pd.to_datetime(df['time'], errors='coerce')
    
    # Extract time components
    df['Year'] = df['Datetime'].dt.year
    df['Month'] = df['Datetime'].dt.month
    df['Day'] = df['Datetime'].dt.day
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Second'] = df['Datetime'].dt.second
    df['DayOfWeek'] = df['Datetime'].dt.dayofweek  # Monday=0, Sunday=6
    
    print("Temporal features created: Year, Month, Day, Hour, Minute, Second, DayOfWeek")
    
    # Optional: Cyclical encoding for periodic features (commented out)
    # df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    # df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    # df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    # df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    # df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    # df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
    
    # Drop original time columns
    df.drop(['time', 'Datetime'], axis=1, inplace=True)
    print("Dropped original 'time' and 'Datetime' columns")

print(f"\nShape after feature engineering: {df.shape}")
print("\n" + "="*80 + "\n")


# ============================================================================
# 6. CORRELATION ANALYSIS
# ============================================================================

print("Correlation Analysis:")
print("-" * 80)

# Step 6.1: Numerical feature correlation
print("Calculating correlation matrix for numerical features...")

# Identify categorical columns
categorical_cols = ['type', 'magType', 'status']
categorical_cols = [col for col in categorical_cols if col in df.columns]

# Create copy for correlation analysis
df_corr = df.copy()

# Encode categorical variables as numeric codes
for col in categorical_cols:
    if col in df_corr.columns:
        df_corr[col] = df_corr[col].astype('category').cat.codes

# Calculate correlation matrix
corr_matrix = df_corr.corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            cbar=True, square=True, linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("Correlation heatmap saved as 'correlation_heatmap.png'\n")


# Step 6.2: Categorical feature correlation (Cramér's V)
if len(categorical_cols) > 1:
    print("Calculating Cramér's V for categorical features...")
    
    def cramers_v(x, y):
        """Calculate Cramér's V statistic for categorical-categorical association"""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n-1))
        rcorr = r - ((r-1)**2) / (n-1)
        kcorr = k - ((k-1)**2) / (n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0
    
    # Initialize results DataFrame
    cramers_results = pd.DataFrame(np.zeros((len(categorical_cols), len(categorical_cols))),
                                   index=categorical_cols, columns=categorical_cols)
    
    # Calculate Cramér's V for each pair
    for col1 in categorical_cols:
        for col2 in categorical_cols:
            cramers_results.loc[col1, col2] = cramers_v(df[col1], df[col2])
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cramers_results, annot=True, cmap='coolwarm', vmin=0, vmax=1,
                square=True, linewidths=0.5, cbar_kws={'label': "Cramér's V"})
    plt.title("Cramér's V Correlation - Categorical Features", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cramers_v_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Cramér's V heatmap saved as 'cramers_v_heatmap.png'\n")

print("="*80 + "\n")


# ============================================================================
# 7. FEATURE SCALING/NORMALIZATION
# ============================================================================

print("Feature Scaling:")
print("-" * 80)

# Step 7.1: Auto-detect numeric features OR specify manually
# Option A: Auto-detect all numeric columns (recommended for flexibility)
numeric_cols_auto = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude categorical codes if any
for cat_col in categorical_cols:
    if cat_col in numeric_cols_auto:
        numeric_cols_auto.remove(cat_col)

# Option B: Manually specify numeric features to scale (recommended for specific features)
numeric_features = ['depth', 'mag', 'rms']  # Specify critical features to scale
numeric_features = [col for col in numeric_features if col in df.columns]

# Use Option B (manual) if features are specified, otherwise use Option A (auto-detect)
if numeric_features:
    numeric_cols = numeric_features
    print(f"Scaling specified numeric features: {numeric_cols}\n")
else:
    numeric_cols = numeric_cols_auto
    print(f"Scaling auto-detected numeric features: {numeric_cols}\n")

# Step 7.2: Apply Standard Scaling (Z-score normalization)
print("Applying StandardScaler (z-score normalization)...")
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print(f"✓ StandardScaler applied to {len(numeric_cols)} features")
print(f"  Features scaled: {numeric_cols}")

# Step 7.3: Alternative Min-Max Normalization (uncomment to use instead)
# print("Applying MinMaxScaler (0-1 normalization)...")
# normalizer = MinMaxScaler()
# df[numeric_cols] = normalizer.fit_transform(df[numeric_cols])
# print(f"✓ MinMaxScaler applied to {len(numeric_cols)} features")
# print(f"  Features scaled: {numeric_cols}")

print(f"\nScaling Summary:")
print(f"  Total features scaled: {len(numeric_cols)}")
print(f"  Scaling method: StandardScaler (z-score)")
print(f"  Mean of scaled features: {df[numeric_cols].mean().round(4).to_dict()}")
print(f"  Std Dev of scaled features: {df[numeric_cols].std().round(4).to_dict()}")

print("\n" + "="*80 + "\n")



# ============================================================================
# 8. FINAL DATA SUMMARY
# ============================================================================

print("Final Preprocessed Data Summary:")
print("-" * 80)

# Display data info
df.info()
print("\n")

# Display sample of preprocessed data
print("Sample Preprocessed Data:")
print(df.head(10))
print("\n")

# Final shape
print(f"Final Dataset Shape: {df.shape}")
print(f"Final Rows: {df.shape[0]}, Final Columns: {df.shape[1]}")
print("\n" + "="*80 + "\n")


# ============================================================================
# 9. SAVE PREPROCESSED DATA
# ============================================================================

print("Saving Preprocessed Data:")
print("-" * 80)

# Save to CSV
output_filename = 'preprocessed_earthquake_data.csv'
df.to_csv(output_filename, index=False)
print(f"Preprocessed data saved to: {output_filename}")
print(f"File saved with {df.shape[0]} rows and {df.shape[1]} columns")

print("\n" + "="*80)
print("DATA PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*80)
