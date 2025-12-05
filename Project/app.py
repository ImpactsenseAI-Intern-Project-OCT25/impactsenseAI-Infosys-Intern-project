import os
import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor

# Import custom transformers (REQUIRED for joblib.load)
from data_preprocessing_pipeline import (
    MissingValueAnalyzer,
    RMSImputer,
    MagnitudeConverter,
    TemporalFeatureExtractor,
    DamagePotentialCalculator,
    UrbanityFeatureCreator,
    FinalFeatureSelector
)

st.set_page_config(page_title="ImpactSenseAI", layout="wide")

# =============================================================================
# Utility Functions
# =============================================================================

def standardize_column_names(df):
    """Map CSV columns to pipeline expected names"""
    column_mapping = {
        'latitude': ['latitude'],
        'longitude': ['longitude'], 
        'depth': ['depth'],
        'mag': ['mag'],
        'magType': ['magType'],
        'rms': ['rms'],
        'time': ['time'],
        'type': ['type']
    }
    
    df = df.copy()
    new_cols = {}
    
    for expected, variants in column_mapping.items():
        for variant in variants:
            if variant in df.columns:
                new_cols[variant] = expected
                break
    
    df = df.rename(columns=new_cols)
    
    # Validate critical columns
    required_cols = ['latitude', 'longitude', 'depth', 'mag']
    available_required = [col for col in required_cols if col in df.columns]
    
    if len(available_required) < 3:
        st.error(f"❌ Missing required columns. Available: {list(df.columns[:10])}")
        st.stop()
    
    # Drop rows with missing critical data
    df = df.dropna(subset=available_required[:3])
    return df

def create_risk_category(score):
    """Risk categorization matching your pipeline"""
    if score < 2.5:
        return 0  # Low
    elif score < 5.0:
        return 1  # Medium
    elif score < 7.5:
        return 2  # High
    return 3  # Very High

def create_urban_risk(lat, lon, score):
    """Urban risk adjustment"""
    return score * (1 + (abs(lat) + abs(lon)) / 360)

def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    fig, _ = plt.subplots(figsize=(10,6))
    shap.summary_plot(sv, X, show=False, plot_type="bar")
    plt.tight_layout()
    return fig

def transform_input_for_prediction(input_dict, feat_list):
    """Transform single prediction to match pipeline features"""
    X = pd.DataFrame([input_dict])
    
    # Rename to pipeline expected names
    col_mapping = {
        'Latitude': 'latitude',
        'Longitude': 'longitude', 
        'Depth': 'depth',
        'Magnitude': 'mag'
    }
    X = X.rename(columns=col_mapping)
    
    # Add required columns if missing
    if 'magType' not in X.columns:
        X['magType'] = 'ml'
    if 'type' not in X.columns:
        X['type'] = 'earthquake'
    
    # Apply pipeline (subset of transformations for single row)
    if X['type'].iloc[0] != 'earthquake':
        return pd.DataFrame([[0] * len(feat_list)], columns=feat_list)
    
    # Simplified pipeline logic for single prediction
    X['time'] = pd.Timestamp.now().isoformat()
    X['Datetime'] = pd.to_datetime(X['time'])
    X['Year'] = X['Datetime'].dt.year
    X['decade'] = (X['Year'] // 10) * 10
    
    # Mw conversion (assume ML)
    X['Mw'] = 1.2 * X['mag'] - 1.0
    
    # HAZUS damage potential
    mag = X['Mw'].iloc[0]
    depth = max(abs(X['depth'].iloc[0]), 1.0)
    log_pga = mag - 3.5 * np.log10(depth + 7) + 1.8
    pga = 10 ** log_pga
    X['damage_potential'] = 2.5 * np.log10(pga + 0.01) + 7.5
    
    # Urbanity indicator
    X['urbanity_indicator'] = 1 if abs(X['latitude'].iloc[0]) < 60 else 0
    
    # Feature standardization (approximate)
    feature_cols = ['depth', 'rms', 'Mw', 'damage_potential']
    for col in feature_cols:
        if col in X.columns:
            X[col] = (X[col] - X[col].mean()) / (X[col].std() + 1e-8)
    
    # Select exact training features
    X_input = pd.DataFrame(index=X.index, columns=feat_list).fillna(0)
    for col in feat_list:
        if col in X.columns:
            X_input[col] = X[col]
    
    return X_input

# =============================================================================
# Main App
# =============================================================================

st.title("🌍 ImpactSenseAI – Earthquake Risk Prediction")

# Sidebar
with st.sidebar:
    uploaded_file = st.file_uploader("📁 Upload earthquakes CSV", type=["csv"])
    page = st.radio("📋 Navigate", ["📊 Dashboard", "🔮 Predict", "🗺️ Map", "📈 Model", "ℹ️ About"], index=0)

if uploaded_file is None:
    st.info("👆 Please upload your earthquakes CSV to get started")
    st.stop()

# Fixed paths
preprocessing_path = "models/data_preprocessing_pipeline.pkl"
model_path = "models/xgb_risk_model.pkl"

# Load and standardize raw data
with st.spinner("📥 Loading & standardizing data..."):
    df_raw = pd.read_csv(uploaded_file)
    df_raw = standardize_column_names(df_raw)


# Load preprocessing pipeline
if not os.path.exists(preprocessing_path):
    st.error("❌ Preprocessing pipeline missing. Run: `python preprocessing_pipeline.py`")
    st.stop()

preprocessing_pipeline = joblib.load(preprocessing_path)

# Preprocess data
with st.spinner("🔄 Applying complete preprocessing pipeline..."):
    df_processed = preprocessing_pipeline.fit_transform(df_raw)

# Load XGBoost model
if not os.path.exists(model_path):
    st.error("❌ XGBoost model missing. Run: `python train_xgboost.py`")
    st.stop()

saved = joblib.load(model_path)
model, feat_list = saved["model"], saved["features"]

# Add risk categories
risk_labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
df_processed["Risk_Category"] = df_processed["risk_score"].apply(create_risk_category)
df_processed["Risk_Label"] = df_processed["Risk_Category"].map(risk_labels)


# =============================================================================
# Pages
# =============================================================================

if page == "📊 Dashboard":
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Earthquakes", f"{len(df_processed):,}")
        st.metric("High/Very High Risk", len(df_processed[df_processed['Risk_Category'] >= 2]))
        st.metric("Avg Risk Score", f"{df_processed['risk_score'].mean():.2f}")
    
    with col2:
        risk_dist = df_processed['Risk_Label'].value_counts()
        fig_pie = px.pie(values=risk_dist.values, names=risk_dist.index, 
                        title="Risk Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.subheader("📋 Processed Features")
    display_cols = feat_list + ['risk_score', 'Risk_Label']
    st.dataframe(df_processed[display_cols].head(10))
    
    st.subheader("📈 Feature Distributions")
    selected = st.multiselect("Features", feat_list, default=feat_list[:3])
    for feat in selected:
        fig, ax = plt.subplots(figsize=(8,4))
        df_processed[feat].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f'{feat} Distribution')
        st.pyplot(fig)

elif page == "🔮 Predict":
    st.subheader("🎯 Single Earthquake Risk Prediction")
    st.info("🔬 Enter earthquake parameters to predict urban risk score using your trained XGBoost model")
    
    # Fixed current decade
    current_decade = (pd.Timestamp.now().year // 10) * 10
    st.info(f"📅 **Current Decade**: {current_decade}s")
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Seismic Parameters**")
            depth = st.slider("Depth (km)", 0.0, 700.0, 10.0, help="Hypocenter depth in kilometers")
            rms = st.slider("RMS (Root Mean Square)", 0.0, 10.0, 0.5, help="Error/quality metric")
        
        with col2:
            st.markdown("**Magnitude & Urbanity**")
            mw = st.slider("Moment Magnitude (Mw)", 0.0, 10.0, 4.0, help="Moment Magnitude (standardized)")
            damage_potential = st.slider("Damage Potential", 0.0, 20.0, 5.0, help="HAZUS-style damage potential score")
            urbanity_indicator = st.selectbox("Urbanity Indicator", options=[0, 1], index=0, help="0=Rural, 1=Urban")
        
        st.markdown("---")
        st.markdown("**Input Summary**")
        col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
        col_summary1.metric("Depth", f"{depth:.1f} km")
        col_summary2.metric("RMS", f"{rms:.2f}")
        col_summary3.metric("Moment Magnitude (Mw)", f"{mw:.2f}")
        col_summary4.metric("Damage Potential", f"{damage_potential:.2f}")
        
        # Fixed decade display
        col_decade = st.columns(1)
        col_decade[0].metric("Decade", f"{current_decade}s")
        st.markdown(f"🏙️ **Urbanity**: {'Urban' if urbanity_indicator == 1 else 'Rural'}")
        
        submitted = st.form_submit_button("🚀 Predict Risk Score", use_container_width=True)
    
    if submitted:
        with st.spinner("🔄 Processing through XGBoost pipeline..."):
            input_data = {
                'depth': depth,
                'rms': rms,
                'Mw': mw,
                'damage_potential': damage_potential,
                'urbanity_indicator': urbanity_indicator,
                'decade': current_decade  # Fixed current decade value
            }
            
            # Transform input to DataFrame with feature list columns
            X_input = pd.DataFrame([input_data])[feat_list]
            
            # Predict risk score
            risk_score = model.predict(X_input)[0]
            rc = create_risk_category(risk_score)
            urban_risk = risk_score * (1.5 if urbanity_indicator == 1 else 1.0)
            
            # Display results
            st.success("✅ Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Risk Score", f"{risk_score:.3f}/10", delta=f"Category: {risk_labels[rc]}")
            col2.metric("📊 Risk Category", risk_labels[rc])
            col3.metric("🏙️ Urban Risk Score", f"{urban_risk:.3f}")
            
            # SHAP explainability
            st.markdown("### 🔍 Feature Importance (SHAP Explanation)")
            with st.expander("Show SHAP Plot - Feature Contribution to Prediction"):
                try:
                    fig_shap = shap_explain(model, X_input)
                    st.pyplot(fig_shap)
                    st.caption("📌 Shows feature impact on prediction")
                except Exception as e:
                    st.warning(f"⚠️ SHAP plot unavailable: {e}")
elif page == "🗺️ Map":
    st.subheader("🌐 Global Risk Map")
    
    # Clean data for mapping - remove NaN risk_score and ensure lat/lon exist
    df_map = df_processed.dropna(subset=['risk_score', 'latitude', 'longitude']).copy()
    
    if len(df_map) == 0:
        st.warning("⚠️ No valid data for mapping (missing risk_score/lat/lon). Check preprocessing.")
        st.stop()
    
    # Replace any remaining NaN risk_score with minimum valid value for sizing
    df_map['risk_score_plot'] = df_map['risk_score'].fillna(df_map['risk_score'].min())
    df_map['risk_score_plot'] = df_map['risk_score_plot'].clip(lower=0.1)  # Minimum size
    
    # FIXED: Define color_map FIRST (always available)
    color_map = {
        "Low": "#00FF41", 
        "Medium": "#FFD700", 
        "High": "#FF8C00", 
        "Very High": "#FF1744"
    }
    
    # NEW: Risk Category Filter + Download (Columns)
    col_filter, col_download = st.columns([2, 1])
    
    with col_filter:
        selected_risk = st.selectbox(
            "🎯 Filter by Risk Category", 
            options=["All"] + list(risk_labels.values()), 
            index=0,
            help="Filter map markers by risk level"
        )
    
    with col_download:
        # Download FILTERED dataset (respects selection)
        csv_download = df_map.to_csv(index=False)
        st.download_button(
            label="💾 Download Dataset",
            data=csv_download,
            file_name=f"ImpactSenseAI_earthquakes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Filter data based on selection
    if selected_risk != "All":
        risk_num = {v: k for k, v in risk_labels.items()}[selected_risk]
        df_map_filtered = df_map[df_map['Risk_Category'] == risk_num].copy()
        st.info(f"🗺️ Showing **{len(df_map_filtered):,}** '{selected_risk}' risk earthquakes")
    else:
        df_map_filtered = df_map.copy()
        st.info(f"🗺️ Showing **{len(df_map_filtered):,}** earthquakes (All categories)")
    
    # Limit for performance
    df_map_sample = df_map_filtered.head(5000)
    
    # FIXED: Conditional color_discrete_map (now uses defined color_map)
    if selected_risk != "All":
        color_discrete_map = {selected_risk: color_map[selected_risk]}
    else:
        color_discrete_map = color_map
    
    # FIXED: Hover data - ONLY use columns that exist in processed data
    hover_columns = ['risk_score', 'Mw', 'depth', 'urbanity_indicator']
    
    # Interactive Map (NOW WORKS - no magType)
    fig = px.scatter_mapbox(
        df_map_sample,
        lat="latitude", 
        lon="longitude",
        color="Risk_Label",
        size="risk_score_plot",
        size_max=15,
        opacity=0.7,
        color_discrete_map=color_discrete_map,  # ✅ Properly defined
        hover_data=hover_columns,  # ✅ Only existing columns
        zoom=1,
        mapbox_style="carto-positron",
        title=f"Earthquake Risk Distribution ({selected_risk if selected_risk != 'All' else 'All Categories'})"
    )
    
    fig.update_layout(height=600, margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)
    
    # Data summary (Updated for filtered data)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Available", f"{len(df_map):,}")
    with col2:
        st.metric("Filtered Shown", f"{len(df_map_filtered):,}")
    with col3:
        st.metric("Risk Score Range", f"{df_map_filtered['risk_score'].min():.2f} - {df_map_filtered['risk_score'].max():.2f}")
    with col4:
        st.metric("Avg Risk Score", f"{df_map_filtered['risk_score'].mean():.2f}")
    
    # Sample table (Filtered data)
    st.markdown("### 📋 Sample Filtered Data")
    display_cols = ['latitude', 'longitude', 'risk_score', 'Risk_Label', 'Mw', 'depth', 'urbanity_indicator']
    st.dataframe(df_map_filtered[display_cols].head(10))


elif page == "📈 Model":
    st.subheader("🤖 XGBoost Regressor")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    # Extract features and target
    X_temp = df_processed[feat_list]
    y_temp = df_processed['risk_score']

    # Remove any rows with NaNs in features or target before splitting
    valid_mask = y_temp.notna() & X_temp.notna().all(axis=1)
    X_temp_clean = X_temp[valid_mask]
    y_temp_clean = y_temp[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_temp_clean, y_temp_clean, test_size=0.2, random_state=42
    )

    # Predict on clean test set
    y_pred = model.predict(X_test)

    # Filter NaNs if any remain (unlikely after clean split)
    mask = ~np.isnan(y_test) & ~np.isnan(y_pred)
    y_test_clean = y_test[mask]
    y_pred_clean = y_pred[mask]

    col1, col2, col3 = st.columns(3)
    if len(y_test_clean) == 0:
        st.warning("No valid samples for model metrics after NaN filtering.")
    else:
        col1.metric("MAE", f"{mean_absolute_error(y_test_clean, y_pred_clean):.4f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test_clean, y_pred_clean)):.4f}")
        col3.metric("R²", f"{r2_score(y_test_clean, y_pred_clean):.4f}")

    # Feature importance plot
    importance_df = pd.DataFrame({
        'feature': feat_list,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig = px.bar(importance_df, x='importance', y='feature', orientation='h', title="Feature Importance")
    st.plotly_chart(fig)

else:  # About
    st.subheader("ℹ️ About ImpactSenseAI")
    
    st.markdown("""
    ## 🎯 **Urban Earthquake Risk Assessment**
    
    ImpactSenseAI integrates geospatial analysis, seismic physics, and machine learning to 
    predict earthquake impact with **urbanity-aware risk scoring**.
    
    ---
    
    ## 📊 **Risk Score Calculation**
    
    ### Formula
    
    **For Urban Areas** (urbanity_indicator = 1):
    ```
    Risk Score = min((Mw²) × (1/max(d, 1)) × 1.5, 10)
    ```
    
    **For Rural Areas** (urbanity_indicator = 0):
    ```
    Risk Score = min(Mw × (1/max(d, 1)), 10)
    ```
    
    Where:
    - **Mw** = Moment Magnitude (standardized across all magnitude types)
    - **d** = Depth in kilometers
    - **1.5** = Urban multiplier (50% additional risk factor)
    - **10** = Normalization cap for interpretability
    
    ### Justification
    
    ✅ **Magnitude Squaring (Urban)**: Captures non-linear increase in urban damage with magnitude
    
    ✅ **Reciprocal Depth**: Reflects physical attenuation; shallow earthquakes produce higher ground motion
    
    ✅ **Urbanity Multiplier (1.5)**: Empirically calibrated factor reflecting concentrated population, 
    critical infrastructure, and economic assets in urban zones
    
    ✅ **Capping at 10**: Normalizes scores to meaningful [0, 10] range and prevents extreme outliers
    
    ---
    
    ## 📋 **Risk Categories**
    
    | Category | Range | Interpretation |
    |----------|-------|-----------------|
    | **Low** | < 2.5 | Minimal impact potential |
    | **Medium** | 2.5 - 5.0 | Moderate damage potential |
    | **High** | 5.0 - 7.5 | Severe damage likely |
    | **Very High** | ≥ 7.5 | Extreme urban damage potential |
    
    ---
    
    ## 🔧 **Complete Processing Pipeline**
    
    ```
    Raw CSV Data
        ↓
    Missing Value Handling (>30% threshold)
        ↓
    RMS Imputation (LinearRegression on mag, depth, nst, gap, dmin)
        ↓
    Magnitude Conversion to Mw (ML, MS, MB → Mw standardization)
        ↓
    Temporal Feature Extraction (Year, Month, Hour, DayOfWeek)
        ↓
    HAZUS Damage Potential Calculation
        ↓
    Urbanity Indicator (Geospatial: latitude threshold < ±60°)
        ↓
    Risk Score Computation (Urban vs Rural formulas)
        ↓
    Feature Scaling (StandardScaler + MinMaxScaler)
        ↓
    XGBoost Training (1000 trees, max_depth=6, learning_rate=0.1)
        ↓
    Risk Prediction & Categorization
    ```
    
    ---
    
    ## 🎓 **Key Features**
    
    | Feature | Type | Range | Purpose |
    |---------|------|-------|---------|
    | **depth** | Continuous | 0-700 km | Seismic attenuation factor |
    | **rms** | Continuous | 0-10 | Root mean square error (quality metric) |
    | **Mw** | Continuous | 0-10 | Standardized moment magnitude |
    | **damage_potential** | Continuous | 0-10+ | HAZUS-based impact potential |
    | **urbanity_indicator** | Binary | 0 or 1 | Urban (1) vs Rural (0) classification |
    | **decade** | Continuous | 0-10 | Temporal clustering feature |
    
    ### Target: **risk_score** (0-10)
    Urban-adjusted severity score reflecting earthquake impact potential
    
    ---
    
    ## 🚀 **Production Architecture**
    
    - **Preprocessing**: `complete_earthquake_preprocessing_pipeline.pkl` (scikit-learn Pipeline)
    - **Model**: `xgb_risk_model.pkl` (XGBoost Regressor with 1000 estimators)
    - **Framework**: Streamlit for interactive deployment
    - **Data Format**: USGS earthquake CSV (time, latitude, longitude, depth, mag, magType, etc.)
    
    ---
    
    ## 📚 **Data Quality Standards**
    
    ✓ Magnitude source: Prioritize Mw or variants (mwc, mww, mwb, mwr)  
    ✓ Avoid saturated scales (mb, ms) for M > 6.5  
    ✓ CRS: WGS84 (EPSG:4326) for geographic consistency  
    ✓ Null handling: Regression imputation for RMS, binary assignment for urbanity  
    ✓ Regularization: L1/L2 penalties + cross-validation to prevent overfitting
    
    ---
    
    ## 🌍 **Use Cases**
    
    🔴 **Disaster Management**: Real-time urban earthquake risk alerts  
    🟠 **Urban Planning**: Infrastructure vulnerability assessment  
    🟡 **Insurance**: Risk-based premium calculation for urban zones  
    🟢 **Research**: Geospatial earthquake impact modeling
    """)
    
    st.markdown("---")
    # st.markdown("*Developed with scientifically-grounded urban risk assessment methodology | Production-ready ML pipeline*")
