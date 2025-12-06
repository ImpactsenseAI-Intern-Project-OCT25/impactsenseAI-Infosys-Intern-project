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

st.set_page_config(page_title="ImpactSense XGBoost", layout="wide", page_icon="🌍")

# XGBoost Model Configuration
@st.cache_resource
def load_xgboost_model():
    """Load the trained XGBoost model"""
    model_path = "models/xgboost.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        st.sidebar.success(f"✅ XGBoost Model loaded: {model_path}")
        return model
    else:
        st.error(f"❌ Model not found: {model_path}")
        st.stop()

# Load model
model = load_xgboost_model()

# Your 6 exact features from training
FEATURES = ['depth', 'rms', 'Mw', 'damage_potential', 'urbanity_indicator', 'decade']

# Utility functions
def create_risk_category(score):
    if score < 4.0: return 0  # Low
    if score < 6.0: return 1  # Moderate
    return 2  # High

def get_risk_label(score):
    return ["🟢 Low", "🟡 Moderate", "🔴 High"][create_risk_category(score)]

def calculate_damage_potential(Mw, depth):
    """Same formula used during training"""
    return 0.6 * Mw + 0.2 * (700 - depth) / 700 * 10

def shap_explain(model, X):
    """SHAP explanation optimized for XGBoost"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.title("🔍 SHAP: Feature Impact on Prediction", fontsize=14, fontweight='bold')
    return plt.gcf()

def prepare_input_for_model(inputs):
    """Transform user input to match exact 6-feature model format"""
    X = pd.DataFrame([inputs])
    
    # Ensure all features exist with correct names
    feature_data = {
        'depth': inputs.get('depth', 10.0),
        'rms': inputs.get('rms', 0.5),
        'Mw': inputs.get('Mw', 5.0),
        'damage_potential': calculate_damage_potential(inputs.get('Mw', 5.0), inputs.get('depth', 10.0)),
        'urbanity_indicator': inputs.get('urbanity_indicator', 0.5),
        'decade': inputs.get('decade', 2020)
    }
    
    return pd.DataFrame([feature_data])[FEATURES]

# Enhanced Sidebar
with st.sidebar:
    st.markdown("## 🌍 **ImpactSense XGBoost**")
    st.markdown("**No Preprocessing Pipeline**")
    st.markdown(f"**Features**: {len(FEATURES)}")
    st.markdown("**Trees**: 1000")
    st.markdown("**Status**: 🚀 Production Ready")
    
    # Model metrics display
    metrics_path = "models/xgboost_metrics.csv"
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path).iloc[0]
        st.markdown("---")
        st.metric("MAE", f"{metrics['mae']:.4f}")
        st.metric("RMSE", f"{metrics['rmse']:.4f}")
        st.metric("R²", f"{metrics['r2']:.4f}")
    
    st.markdown("---")
    uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"], help="Upload earthquake data for analysis")

# Main Header
st.title("🌍 ImpactSense – XGBoost Earthquake Risk Predictor")
st.markdown("**Production ML model trained on 6 key geophysical features** | No preprocessing required")

# Navigation
page = st.sidebar.radio("📋 Go to", ["📊 Dashboard", "🔮 Predict", "🗺️ Risk Map", "📈 Explainability", "ℹ️ Model Info"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    st.session_state.df_raw = df_raw
    st.sidebar.success(f"✅ Data loaded: {len(df_raw):,} rows")
else:
    # Create sample data if no upload
    st.info("👆 Upload CSV or use prediction interface below")
    df_raw = pd.DataFrame({
        'Latitude': [34.0, -118.0],
        'Longitude': [-118.0, 34.0],
        'Depth': [10.0, 15.0],
        'Magnitude': [5.2, 6.1]
    })
    st.session_state.df_raw = df_raw

if page == "📊 Dashboard":
    st.header("📈 Model Performance & Data Explorer")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset Size", f"{len(df_raw):,}")
    with col2:
        st.metric("Features", len(FEATURES))
    with col3:
        st.metric("Model Trees", 1000)
    with col4:
        st.metric("Prediction Speed", "⚡ Instant")
    
    # Feature importance
    st.subheader("🎯 Feature Importance (XGBoost Native)")
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    ax.set_xlabel('Importance Score')
    ax.set_title('Top Features Driving Risk Predictions', fontsize=16, fontweight='bold')
    ax.bar_label(bars, fmt='%.3f')
    st.pyplot(fig)
    
    # Data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df_raw.head(10).style.format("{:.2f}"))

elif page == "🔮 Predict":
    st.header("🎯 Real-time Risk Prediction")
    
    # Enhanced input form
    with st.form(key="prediction_form", clear_on_submit=True):
        st.subheader("🌍 Earthquake Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            depth = st.slider("💧 Depth (km)", 0.0, 700.0, 10.0, help="Earthquake hypocenter depth")
            rms = st.slider("📊 RMS (s)", 0.0, 10.0, 0.5, help="Root Mean Square of P/S arrivals")
            decade = st.slider("📅 Decade", 1900, 2030, 2020, step=10, help="Temporal grouping")
        
        with col2:
            Mw = st.slider("⚡ Magnitude (Mw)", 0.0, 10.0, 5.0, 0.1, help="Moment magnitude")
            urbanity = st.slider("🏙️ Urbanity (0-1)", 0.0, 1.0, 0.5, 0.05, help="Population density proxy")
        
        # Auto-calculate damage potential (training formula)
        damage_potential = calculate_damage_potential(Mw, depth)
        st.info(f"🔧 **Damage Potential**: {damage_potential:.2f} (auto-calculated)")
        
        col1, col2 = st.columns(2)
        predict_btn = col1.form_submit_button("🚀 Predict Risk", use_container_width=True)
        col2.markdown("")

    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    if predict_btn:
        # Prepare exact input format
        input_data = {
            'depth': depth,
            'rms': rms,
            'Mw': Mw,
            'damage_potential': damage_potential,
            'urbanity_indicator': urbanity,
            'decade': decade
        }
        
        X_input = prepare_input_for_model(input_data)
        risk_score = model.predict(X_input)[0]
        
        # Store results
        st.session_state.prediction_results = {
            'input': input_data,
            'X_input': X_input,
            'risk_score': risk_score,
            'risk_category': create_risk_category(risk_score)
        }
        
        # Display results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🎯 Risk Score", f"{risk_score:.3f}", delta=None)
        with col2:
            st.metric("📊 Risk Level", get_risk_label(risk_score))
        with col3:
            st.metric("⚙️ Features Used", len(FEATURES))
        
        # Input details
        with st.expander("📋 Input Details", expanded=True):
            st.dataframe(X_input.T.style.format("{:.3f}"), use_container_width=True)
        
        st.balloons()

    # Show last prediction
    if st.session_state.prediction_results:
        st.success("✅ Latest prediction saved!")

elif page == "🗺️ Risk Map":
    st.header("🗺️ Global Risk Visualization")
    if 'Latitude' in df_raw.columns and 'Longitude' in df_raw.columns:
        # Predict on entire dataset
        predictions = model.predict(df_raw[['depth', 'rms', 'Mw', 'damage_potential', 'urbanity_indicator', 'decade']].fillna(0))
        df_map = df_raw.copy()
        df_map['risk_score'] = predictions
        df_map['risk_label'] = df_map['risk_score'].apply(get_risk_label)
        
        fig = px.scatter_mapbox(
            df_map, lat="Latitude", lon="Longitude",
            color="risk_label", size="risk_score", size_max=15,
            color_discrete_map={"🟢 Low": "green", "🟡 Moderate": "orange", "🔴 High": "red"},
            zoom=2, mapbox_style="carto-positron",
            hover_data=['risk_score', 'depth', 'Mw'],
            title="🌍 Earthquake Risk Heatmap (XGBoost Predictions)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**{len(df_map[df_map['risk_label'] == '🔴 High']):,} High Risk** | "
                   f"**{len(df_map[df_map['risk_label'] == '🟡 Moderate']):,} Moderate** | "
                   f"**{len(df_map[df_map['risk_label'] == '🟢 Low']):,} Low**")
    else:
        st.warning("📍 Latitude/Longitude columns required for mapping")

elif page == "📈 Explainability":
    st.header("🔍 SHAP Model Interpretability")
    
    if st.session_state.prediction_results:
        X_explain = st.session_state.prediction_results['X_input']
        st.subheader("🎯 Latest Prediction Explanation")
        fig = shap_explain(model, X_explain)
        st.pyplot(fig)
    else:
        # Sample explanation
        sample_input = prepare_input_for_model({
            'depth': 15.0, 'rms': 0.8, 'Mw': 6.2,
            'damage_potential': 6.1, 'urbanity_indicator': 0.7, 'decade': 2020
        })
        st.subheader("📊 Sample Prediction Explanation")
        fig = shap_explain(model, sample_input)
        st.pyplot(fig)
    
    st.markdown("""
    **SHAP shows:**
    - 🔴 **Red**: Features **increasing** risk score
    - 🔵 **Blue**: Features **decreasing** risk score
    - 📏 **Bar length**: Magnitude of feature impact
    """)

else:  # Model Info
    st.header("ℹ️ XGBoost Model Technical Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🏗️ **Model Architecture**
        - **Algorithm**: XGBoost Regressor
        - **Trees**: 1,000 (full training)
        - **Max Depth**: 6
        - **Learning Rate**: 0.10
        - **Subsample**: 0.8
        - **Regularization**: L1=0.1, L2=1.0
        
        ### ✨ **Key Advantages**
        - ✅ No preprocessing pipeline
        - ✅ Handles missing values natively
        - ✅ Production-ready inference
        - ⚡ Instant predictions
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 **Feature Engineering**
        | Feature | Description | Source |
        |---------|-------------|--------|
        | `depth` | Hypocenter depth (km) | Raw |
        | `rms` | Arrival time residuals | Raw |
        | `Mw` | Moment magnitude | Raw |
        | `damage_potential` | 0.6*Mw + 0.2*(700-depth)/700*10 | Engineered |
        | `urbanity_indicator` | Population proxy (0-1) | Engineered |
        | `decade` | Year//10 grouping | Engineered |
        """)

# Footer
st.markdown("---")
st.markdown("*ImpactSense AI | XGBoost Production Model`*")
