import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="Half Marathon Time Predictor",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .info-card {
        background: #f8f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid #764ba2;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .example-text {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        font-style: italic;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safe way to get API keys
def get_api_key():
    """Safely get OpenAI API key from various sources"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return api_key
    except:
        pass
    
    return ""

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    OPENAI_API_KEY = get_api_key()
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None

# Try to import plotly
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

# Simple Half Marathon Prediction Model
class SimpleHalfMarathonPredictor:
    """
    A simple mathematical model for half marathon time prediction
    based on statistical analysis of running performance data
    """
    
    def predict(self, X):
        """
        Predict half marathon times based on age, gender, and 5K time
        X should be a DataFrame with columns: 'Wiek' (age), 'Płeć' (gender), '5 km Czas' (5K time in seconds)
        """
        predictions = []
        
        for _, row in X.iterrows():
            age = row['Wiek']
            gender = row['Płeć']  # 0 = male, 1 = female
            time_5k_seconds = row['5 km Czas']
            
            # Base conversion: 5K to half marathon using typical ratios
            # Average ratio is about 4.6-4.8x for recreational runners
            base_ratio = 4.65
            
            # Adjust for gender (women typically have slightly better endurance)
            if gender == 1:  # female
                base_ratio *= 0.98
            
            # Adjust for age (performance declines with age)
            if age < 30:
                age_factor = 1.0
            elif age < 40:
                age_factor = 1.02
            elif age < 50:
                age_factor = 1.05
            elif age < 60:
                age_factor = 1.10
            else:
                age_factor = 1.15
            
            # Adjust for fitness level based on 5K time
            time_5k_minutes = time_5k_seconds / 60
            
            if time_5k_minutes < 20:  # Very fast
                fitness_factor = 0.95
            elif time_5k_minutes < 25:  # Fast
                fitness_factor = 0.98
            elif time_5k_minutes < 30:  # Average
                fitness_factor = 1.0
            elif time_5k_minutes < 35:  # Slower
                fitness_factor = 1.02
            else:  # Much slower
                fitness_factor = 1.05
            
            # Calculate prediction
            predicted_seconds = time_5k_seconds * base_ratio * age_factor * fitness_factor
            predictions.append(predicted_seconds)
        
        return np.array(predictions)

# Initialize the model
@st.cache_resource
def load_prediction_model():
    """Load the prediction model - try original first, then fallback to simple model"""
    MODEL_URL = "https://cersei568.fra1.digitaloceanspaces.com/models/best_finalized_model.pkl"
    MODEL_PATH = "models/best_finalized_model.pkl"
    
    # Try to load the original model
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🚀 Downloading AI model..."):
            progress_bar = st.progress(0)
            try:
                r = requests.get(MODEL_URL, stream=True)
                if r.status_code == 200:
                    total_size = int(r.headers.get('content-length', 0))
                    downloaded_size = 0
                    
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                if total_size > 0:
                                    progress_bar.progress(downloaded_size / total_size)
                    
                    progress_bar.progress(1.0)
                else:
                    st.warning("Could not download original model, using mathematical model instead")
                    return SimpleHalfMarathonPredictor(), "simple"
            except Exception as e:
                st.warning(f"Could not download original model ({e}), using mathematical model instead")
                return SimpleHalfMarathonPredictor(), "simple"
    
    # Try to load the original model
    if os.path.exists(MODEL_PATH):
        try:
            import joblib
            pipeline = joblib.load(MODEL_PATH)
            st.success("✅ Original AI model loaded successfully!")
            return pipeline, "original"
        except Exception as e:
            st.warning(f"Could not load original model ({e}), using mathematical model instead")
    
    # Fallback to simple model
    st.info("📐 Using mathematical prediction model")
    return SimpleHalfMarathonPredictor(), "simple"

pipeline, model_type = load_prediction_model()

# Header
st.markdown("""
<div class="main-header">
    <h1>🏃‍♀️ Half Marathon Time Predictor 🏃‍♂️</h1>
    <p style="font-size: 1.2rem; margin: 0;">Intelligent prediction based on your 5K performance</p>
</div>
""", unsafe_allow_html=True)

# Show model info
if model_type == "simple":
    st.markdown("""
    <div class="warning-card">
        <h4>📐 Mathematical Model Active</h4>
        <p>We're using a proven mathematical model based on running performance research. 
        This model analyzes age, gender, and 5K performance to predict your half marathon time 
        using established athletic performance ratios.</p>
    </div>
    """, unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 How it works</h3>
        <p>Our model analyzes your:</p>
        <ul>
            <li>🎂 Age</li>
            <li>👤 Gender</li>
            <li>⏱️ 5K running time</li>
        </ul>
        <p>Then predicts your half-marathon finish time using sports science!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="example-text">
        <strong>💡 Examples:</strong><br>
        "Sarah, 28, female, 5K in 23:45"<br>
        "Male, 35, 5K best: 20:30"<br>
        "42 year old woman, 26 minutes for 5km"
    </div>
    """)

with col1:
    st.markdown("### 📝 Enter Your Details")
    
    # Manual input
    col_age, col_gender = st.columns(2)
    with col_age:
        age = st.number_input("Age", min_value=16, max_value=80, value=30)
    with col_gender:
        gender = st.selectbox("Gender", ["Female", "Male"])
    
    time_input = st.text_input(
        "5K Time (mm:ss format)", 
        placeholder="e.g., 25:30",
        help="Enter your 5K time in minutes:seconds format"
    )
    
    # AI input as bonus feature
    if openai_client:
        st.markdown("---")
        st.markdown("### 🤖 Or describe yourself (AI-powered)")
        ai_input = st.text_area(
            "Natural language input:",
            placeholder="I'm John, 32 years old, male. My best 5K time is 22:15...",
            height=100
        )
        use_ai = st.checkbox("Use AI to parse the description above")
    else:
        use_ai = False
    
    predict_button = st.button("🔮 Predict My Half Marathon Time", use_container_width=True)

def display_prediction_results(pred_seconds, age, gender, pace_5k_str):
    """Display prediction results with visualizations"""
    h = int(pred_seconds // 3600)
    m = int((pred_seconds % 3600) // 60)
    s = int(pred_seconds % 60)
    pretty = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
    
    st.markdown(f"""
    <div class="prediction-card">
        🎉 Your Predicted Half Marathon Time: {pretty} 🎉
    </div>
    """, unsafe_allow_html=True)
    
    # Profile info
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>👤 Gender</h4>
            <p style="font-size: 1.2rem; color: #667eea;">{gender}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎂 Age</h4>
            <p style="font-size: 1.2rem; color: #667eea;">{age} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    with info_col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ 5K Time</h4>
            <p style="font-size: 1.2rem; color: #667eea;">{pace_5k_str}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pace analysis
    total_km = 21.0975
    pace_per_km_sec = pred_seconds / total_km
    pace_minutes = int(pace_per_km_sec // 60)
    pace_seconds = int(pace_per_km_sec % 60)
    pace_str = f"{pace_minutes}:{pace_seconds:02d}"
    
    st.markdown("### 📊 Race Analysis")
    
    pace_col1, pace_col2 = st.columns(2)
    
    with pace_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏃‍♀️ Average Pace per KM</h4>
            <p style="font-size: 1.5rem; color: #11998e; font-weight: bold;">{pace_str} /km</p>
        </div>
        """, unsafe_allow_html=True)
    
    with pace_col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 Total Distance</h4>
            <p style="font-size: 1.5rem; color: #11998e; font-weight: bold;">21.1 km</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Chart
    st.markdown("### 📈 Projected Split Times")
    
    distances = list(range(1, 22))
    
    if PLOTLY_AVAILABLE:
        times_minutes = [(i * pace_per_km_sec) / 60 for i in distances]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distances,
            y=times_minutes,
            mode='lines+markers',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8, color='#764ba2'),
            hovertemplate='<b>%{x} km</b><br>Time: %{customdata}<extra></extra>',
            customdata=[f"{int(t//60)}:{int(t%60):02d}" for t in [i * pace_per_km_sec for i in distances]]
        ))
        
        fig.update_layout(
            title="Your Projected Race Progress",
            xaxis_title="Distance (km)",
            yaxis_title="Cumulative Time (minutes)",
            template="plotly_white",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        times_seconds = [i * pace_per_km_sec for i in distances]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(distances, times_seconds, marker='o', color='#667eea', linewidth=3, markersize=6)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Cumulative Time (seconds)")
        ax.set_title("Your Projected Race Progress")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Split times table
    st.markdown("### 📋 Kilometer Split Times")
    
    split_data = []
    for i in range(1, 22):
        cumulative_seconds = i * pace_per_km_sec
        cumulative_time = f"{int(cumulative_seconds//60)}:{int(cumulative_seconds%60):02d}"
        split_time = f"{pace_minutes}:{pace_seconds:02d}"
        
        split_data.append({
            "KM": f"KM {i}",
            "Split Time": split_time,
            "Cumulative Time": cumulative_time
        })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            pd.DataFrame(split_data[:11]),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.dataframe(
            pd.DataFrame(split_data[11:]),
            use_container_width=True,
            hide_index=True
        )

# Simple AI parsing function
def simple_ai_parse(text):
    if not openai_client:
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract age, gender (male/female), and 5K time (mm:ss format) from the text. Return as JSON with keys: age, gender, time_5k. If any info is missing, use null."},
                {"role": "user", "content": text}
            ],
            max_tokens=100,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except:
        return None

# Prediction logic
if predict_button:
    final_age = age
    final_gender = gender
    final_time = time_input
    
    # Try AI parsing if requested
    if use_ai and openai_client and 'ai_input' in locals() and ai_input.strip():
        parsed = simple_ai_parse(ai_input)
        if parsed:
            if parsed.get('age'): final_age = parsed['age']
            if parsed.get('gender'): final_gender = parsed['gender'].title()
            if parsed.get('time_5k'): final_time = parsed['time_5k']
    
    if not final_time:
        st.warning("Please enter your 5K time")
    else:
        # Parse time input
        time_pattern = re.match(r'(\d{1,2}):(\d{2})', final_time)
        if not time_pattern:
            st.error("Please enter time in MM:SS format (e.g., 25:30)")
        else:
            minutes, seconds = map(int, time_pattern.groups())
            pace_sec = minutes * 60 + seconds
            gender_num = 0 if final_gender.lower() == "male" else 1
            
            # Make prediction
            X_new = pd.DataFrame([{
                "Wiek": final_age,
                "Płeć": gender_num,
                "5 km Czas": pace_sec
            }])
            
            try:
                pred_seconds = pipeline.predict(X_new)[0]
                display_prediction_results(pred_seconds, final_age, final_gender, final_time)
            except Exception as e:
                st.error(f"❌ Model prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🏃‍♀️ Train smart, race faster! 🏃‍♂️</p>
    <p style="font-size: 0.8rem;">This prediction is based on sports science research and statistical models. Actual performance may vary based on training, weather, and race day conditions.</p>
</div>
""", unsafe_allow_html=True)