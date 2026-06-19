import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

# Set up browser window parameters
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Render customized visual style layers
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #6a0dad;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-card {
    background-color: #f0f8ff;
    padding: 2rem;
    border-radius: 10px;
    border-left: 5px solid #6a0dad;
    margin: 1rem 0;
}
.confidence-bar {
    height: 20px;
    background-color: #e0e0e0;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.confidence-fill {
    height: 100%;
    border-radius: 10px;
    background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# Optimization: Cache model/metadata resources to prevent unneeded heavy processing reloads
@st.cache_resource
def load_model(format_type='joblib'):
    try:
        if format_type == 'joblib':
            return joblib.load('models/iris_model.joblib')
        elif format_type == 'pickle':
            with open('models/iris_model.pickle', 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model binary: {e}")
        return None

@st.cache_resource
def load_model_info():
    try:
        with open('models/model_info.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error reading model_info.json metadata: {e}")
        return None

@st.cache_resource
def load_feature_ranges():
    try:
        with open('models/feature_ranges.json', 'r') as f:
            return json.load(f)
    except:
        return {
            'sepal_length': {'min': 4.0, 'max': 8.0, 'default': 5.8},
            'sepal_width': {'min': 2.0, 'max': 4.5, 'default': 3.0},
            'petal_length': {'min': 1.0, 'max': 7.0, 'default': 4.0},
            'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2}
        }

# Read model details
model_info = load_model_info()
feature_ranges = load_feature_ranges()

# Build Control Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    model_format = st.radio(
        "Model Extension Target:",
        ["joblib", "pickle"],
        help="Select your chosen backend serial compression layout strategy."
    )
    
    # Initialize targeted model mapping execution
    model = load_model(model_format)
    
    if st.button("🔄 Clear Server App Cache"):
        st.cache_resource.clear()
        st.success("Internal system cache flushed successfully.")
        
    st.divider()
    st.subheader("📊 Model Blueprint Metrics")
    if model_info:
        st.write(f"**Architecture:** {model_info.get('model_type')}")
        st.write(f"**Baseline Accuracy:** {model_info.get('accuracy', 0.90):.1%}")
        st.write(f"**Input Structural Dimension:** {len(model_info.get('feature_names', []))} Features")
        st.write(f"**Output Targets Classification:** {len(model_info.get('target_names', []))} Labels")

# Construct Main Stage Layout View
st.markdown('<h1 class="main-header">🌸 Iris Flower Classification Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Modify flower dimension sliders down below to compute algorithmic inference targets in near real-time.")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Feature Measurements Input")
    sepal_length = st.slider("**Sepal Length (cm)**", float(feature_ranges['sepal_length']['min']), float(feature_ranges['sepal_length']['max']), float(feature_ranges['sepal_length']['default']), 0.1)
    sepal_width = st.slider("**Sepal Width (cm)**", float(feature_ranges['sepal_width']['min']), float(feature_ranges['sepal_width']['max']), float(feature_ranges['sepal_width']['default']), 0.1)
    petal_length = st.slider("**Petal Length (cm)**", float(feature_ranges['petal_length']['min']), float(feature_ranges['petal_length']['max']), float(feature_ranges['petal_length']['default']), 0.1)
    petal_width = st.slider("**Petal Width (cm)**", float(feature_ranges['petal_width']['min']), float(feature_ranges['petal_width']['max']), float(feature_ranges['petal_width']['default']), 0.1)

with col2:
    st.header("📊 Current Data State")
    features_df = pd.DataFrame({
        'Feature Metric': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Evaluated Vector Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    })
    st.dataframe(features_df, hide_index=True, use_container_width=True)

# Format structured inputs matrix arrays
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

st.markdown("---")

# Prediction Handler
if st.button("🎯 Execute Species Prediction Target", type="primary", use_container_width=True):
    if model is not None and model_info is not None:
        try:
            prediction = model.predict(input_features)[0]
            prediction_proba = model.predict_proba(input_features)[0]
            predicted_class = model_info['target_names'][prediction]
            
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            st.markdown(f"### 📋 Classified Specimen Results: **{predicted_class.upper()}**")
            st.markdown("#### 📈 Distribution Confidence Matrix:")
            
            for i, prob in enumerate(prediction_proba):
                species = model_info['target_names'][i]
                percentage = prob * 100
                
                col_text, col_prog = st.columns([1, 3])
                with col_text:
                    st.write(f"**{species.title()}**")
                with col_prog:
                    st.markdown(f"""
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {percentage}%;">
                            {percentage:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction Pipeline Error Trace: {e}")
    else:
        st.error("Validation Halt: Dependencies or serialization files missing. Verify file routes.")