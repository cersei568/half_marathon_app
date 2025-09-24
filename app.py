import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import observe

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
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        padding: 1rem;
        font-size: 1rem;
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
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .example-text {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        font-style: italic;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MODEL_URL = "https://cersei568.fra1.digitaloceanspaces.com/models/best_finalized_model.pkl"
MODEL_PATH = "models/best_finalized_model.pkl"

if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY not set in .env — put your OpenAI key there and restart.")
    st.stop()

# Fallback prediction function
def fallback_prediction(age, gender_num, pace_5k_seconds):
    """
    Simple fallback prediction based on statistical analysis
    This is used when the ML model can't be loaded
    """
    # Convert 5K time to half marathon time using empirical formulas
    # These constants are derived from running performance analysis
    
    # Base conversion factor from 5K to half marathon
    base_multiplier = 4.66  # Approximately 21.1km / 5km * pace adjustment
    
    # Age factor (performance typically decreases with age)
    if age < 30:
        age_factor = 1.0
    elif age < 40:
        age_factor = 1.05
    elif age < 50:
        age_factor = 1.12
    else:
        age_factor = 1.20
    
    # Gender factor (statistical difference in average performance)
    gender_factor = 1.0 if gender_num == 0 else 1.15  # 0 = male, 1 = female
    
    # Calculate predicted half marathon time
    predicted_time = pace_5k_seconds * base_multiplier * age_factor * gender_factor
    
    return predicted_time

# Model loading with better error handling
@st.cache_resource
def load_model():
    """Load the ML model with fallback to statistical prediction"""
    model_status = {"loaded": False, "model": None, "error": None}
    
    os.makedirs("models", exist_ok=True)
    
    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🚀 Downloading AI model... This might take a moment."):
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
                    model_status["error"] = f"Failed to download model. HTTP {r.status_code}"
                    return model_status
            except Exception as e:
                model_status["error"] = f"Error downloading model: {e}"
                return model_status
    
    # Try to load the model with different approaches
    try:
        # First, try with joblib (standard approach)
        import joblib
        model = joblib.load(MODEL_PATH)
        model_status["loaded"] = True
        model_status["model"] = model
        return model_status
    except ImportError as e:
        if "pycaret" in str(e).lower():
            model_status["error"] = "PyCaret dependency missing. Using fallback statistical model."
        else:
            model_status["error"] = f"Missing dependency: {e}. Using fallback statistical model."
    except Exception as e:
        model_status["error"] = f"Error loading model: {e}. Using fallback statistical model."
    
    return model_status

# Initialize model
model_info = load_model()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Langfuse setup
langfuse_enabled = True
try:
    langfuse = Langfuse()
except Exception as e:
    langfuse_enabled = False

# Header
st.markdown("""
<div class="main-header">
    <h1>🏃‍♀️ Half Marathon Time Predictor 🏃‍♂️</h1>
    <p style="font-size: 1.2rem; margin: 0;">AI-powered prediction based on your 5K performance</p>
</div>
""", unsafe_allow_html=True)

# Show model status
if not model_info["loaded"]:
    st.markdown(f"""
    <div class="warning-card">
        ⚠️ <strong>Note:</strong> {model_info.get("error", "Unknown error")} 
        <br>Don't worry! We're using a statistical fallback model that's still quite accurate.
    </div>
    """, unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 How it works</h3>
        <p>Our system analyzes your:</p>
        <ul>
            <li>🎂 Age</li>
            <li>👤 Gender</li>
            <li>⏱️ 5K running time</li>
        </ul>
        <p>Then predicts your half-marathon finish time using advanced algorithms!</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="example-text">
        <strong>💡 Example inputs:</strong><br>
        "Hi, I'm Sarah, 28 years old, female. I can run 5K in 23:45"<br><br>
        "Male, 35, my 5K personal best is 20:30"<br><br>
        "I'm 42, woman, running 5km takes me about 26 minutes"
    </div>
    """)

with col1:
    st.markdown("### 📝 Tell us about yourself")
    user_prompt = st.text_area(
        "Describe your running profile:",
        height=120,
        placeholder="Example: I'm John, 32 years old, male. My best 5K time is 22:15...",
        help="Include your age, gender, and 5K running time for the most accurate prediction"
    )
    
    predict_button = st.button("🔮 Predict My Half Marathon Time", use_container_width=True)

# System instruction and functions
SYSTEM_INSTRUCTION = (
    "You are an assistant that transforms text into the following plain format:\n"
    "Gender: <male/female>\n"
    "Age: <number>\n"
    "5K: <time in mm:ss>\n"
    "If some information is missing, write 'no data'. Answer concisely and ONLY output the requested fields (in JSON)."
)

@observe()
def get_llm_output(user_text: str) -> str:
    try:
        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_text}
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI request failed: {e}")
        return ""

def parse_mmss_from_any(text: str):
    if not text:
        return None
    m = re.search(r'(\d{1,3}:\d{2})', text)
    if m:
        return m.group(1)
    m2 = re.search(r'(\d{1,3}[.,]?\d*)\s*min', text, re.I)
    if m2:
        mins = float(m2.group(1).replace(",", "."))
        mm = int(mins)
        ss = int(round((mins - mm) * 60))
        return f"{mm}:{ss:02d}"
    m3 = re.search(r'\b(\d{1,3})\b', text)
    if m3:
        return f"{int(m3.group(1))}:00"
    return None

def fallback_parse(text: str):
    out = {"gender": "no data", "age": "no data", "pace_5k": "no data"}
    if not text:
        return out
    try:
        parsed = json.loads(text.strip())
        for k, v in parsed.items():
            lk, val = k.lower(), str(v).strip()
            if "gender" in lk or "płeć" in lk or "plec" in lk:
                if val.lower() in ("m", "male", "man", "mężczyzna", "mezczyzna"):
                    out["gender"] = "male"
                elif val.lower() in ("k", "female", "woman", "kobieta"):
                    out["gender"] = "female"
            elif "age" in lk or "wiek" in lk:
                m = re.search(r'(\d{1,3})', val)
                if m: out["age"] = int(m.group(1))
            elif "5k" in lk or "pace" in lk:
                mmss = parse_mmss_from_any(val)
                if mmss: out["pace_5k"] = mmss
        return out
    except Exception:
        pass
    for line in text.strip().splitlines():
        if "gender" in line.lower() or "płeć" in line.lower():
            if re.search(r'\b(m|male|man|mężczyzna)\b', line, re.I): out["gender"] = "male"
            elif re.search(r'\b(k|kobieta|woman|female)\b', line, re.I): out["gender"] = "female"
        if "age" in line.lower() or "wiek" in line.lower():
            m = re.search(r'(\d{1,3})', line)
            if m: out["age"] = int(m.group(1))
        if "5k" in line.lower():
            mmss = parse_mmss_from_any(line)
            if mmss: out["pace_5k"] = mmss
    return out

# Prediction logic
if predict_button:
    if not user_prompt or len(user_prompt.strip()) < 3:
        st.warning("⚠️ Please provide a short description including your age, gender, and 5K time.")
    else:
        with st.spinner("🤖 AI is analyzing your input..."):
            assistant_text = get_llm_output(user_prompt)
            
        parsed = fallback_parse(assistant_text)
        
        if parsed["gender"] in ("no data", None) or parsed["age"] in ("no data", None) or parsed["pace_5k"] in ("no data", None):
            st.error("❌ Missing required information. Please provide all three: gender, age, and 5K pace.")
            
            # Show what was detected
            st.markdown("### 🔍 What we detected:")
            detected_col1, detected_col2, detected_col3 = st.columns(3)
            
            with detected_col1:
                status = "✅" if parsed["gender"] not in ("no data", None) else "❌"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{status} Gender</h4>
                    <p>{parsed['gender'] if parsed['gender'] not in ('no data', None) else 'Not detected'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with detected_col2:
                status = "✅" if parsed["age"] not in ("no data", None) else "❌"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{status} Age</h4>
                    <p>{parsed['age'] if parsed['age'] not in ('no data', None) else 'Not detected'}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with detected_col3:
                status = "✅" if parsed["pace_5k"] not in ("no data", None) else "❌"
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{status} 5K Time</h4>
                    <p>{parsed['pace_5k'] if parsed['pace_5k'] not in ('no data', None) else 'Not detected'}</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Process the prediction
            gender_num = 0 if parsed["gender"].lower() in ("m", "male", "man", "mężczyzna", "mezczyzna") else 1
            mmss = parse_mmss_from_any(parsed["pace_5k"])
            mm, ss = map(int, mmss.split(":"))
            pace_sec = mm * 60 + ss
            age_val = int(parsed["age"])

            try:
                # Use ML model if available, otherwise use fallback
                if model_info["loaded"]:
                    X_new = pd.DataFrame([{
                        "Wiek": age_val,
                        "Płeć": gender_num,
                        "5 km Czas": pace_sec
                    }])
                    pred_seconds = model_info["model"].predict(X_new)[0]
                    prediction_method = "🤖 ML Model"
                else:
                    pred_seconds = fallback_prediction(age_val, gender_num, pace_sec)
                    prediction_method = "📊 Statistical Model"
                
                h = int(pred_seconds // 3600)
                m = int((pred_seconds % 3600) // 60)
                s = int(pred_seconds % 60)
                pretty = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
                
                # Beautiful prediction display
                st.markdown(f"""
                <div class="prediction-card">
                    🎉 Your Predicted Half Marathon Time: {pretty} 🎉
                    <br><small style="font-size: 0.8rem; opacity: 0.9;">Predicted using: {prediction_method}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Display extracted information
                st.markdown("### ✅ Extracted Information")
                info_col1, info_col2, info_col3 = st.columns(3)
                
                with info_col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>👤 Gender</h4>
                        <p style="font-size: 1.2rem; color: #667eea;">{parsed['gender'].title()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎂 Age</h4>
                        <p style="font-size: 1.2rem; color: #667eea;">{parsed['age']} years</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with info_col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>⏱️ 5K Time</h4>
                        <p style="font-size: 1.2rem; color: #667eea;">{parsed['pace_5k']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Calculate and display pace information
                total_km = 21.0975
                pace_per_km_sec = pred_seconds / total_km
                pace_minutes = int(pace_per_km_sec // 60)
                pace_seconds = int(pace_per_km_sec % 60)
                pace_str = f"{pace_minutes}:{pace_seconds:02d}"
                
                st.markdown("### 📊 Detailed Analysis")
                
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
                
                # Interactive chart using Plotly
                st.markdown("### 📈 Projected Split Times")
                
                distances = list(range(1, 22))
                times_minutes = [(i * pace_per_km_sec) / 60 for i in distances]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=distances,
                    y=times_minutes,
                    mode='lines+markers',
                    name='Cumulative Time',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8, color='#764ba2'),
                    hovertemplate='<b>%{x} km</b><br>Time: %{customdata}<extra></extra>',
                    customdata=[f"{int(t//60)}:{int(t%60):02d}" for t in [i * pace_per_km_sec for i in distances]]
                ))
                
                fig.update_layout(
                    title="Your Projected Race Progress",
                    xaxis_title="Distance (km)",
                    yaxis_title="Cumulative Time (minutes)",
                    hovermode='x unified',
                    template="plotly_white",
                    height=400,
                    showlegend=False
                )
                
                fig.update_traces(
                    line=dict(color='rgba(102, 126, 234, 0.8)', width=4),
                    marker=dict(size=8, color='#764ba2', line=dict(width=2, color='white'))
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Split times table
                st.markdown("### 📋 Kilometer Split Times")
                
                # Create a beautiful split times dataframe
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
                
                # Display in columns for better readability
                split_col1, split_col2 = st.columns(2)
                
                with split_col1:
                    st.dataframe(
                        pd.DataFrame(split_data[:11]),
                        use_container_width=True,
                        hide_index=True
                    )
                
                with split_col2:
                    st.dataframe(
                        pd.DataFrame(split_data[11:]),
                        use_container_width=True,
                        hide_index=True
                    )
                
                # Langfuse tracking
                if langfuse_enabled:
                    try:
                        dataset_name = "halfmarathon_dataset"
                        try:
                            dataset = langfuse.get_dataset(dataset_name)
                        except Exception:
                            dataset = langfuse.create_dataset(dataset_name, description="User half-marathon inputs")
                        
                        dataset.insert(input=user_prompt, expected_output=pretty)
                        
                        with langfuse.trace(
                            name="halfmarathon-llm-extraction",
                            input=user_prompt,
                            output=pretty,
                            metadata={
                                "model": OPENAI_MODEL,
                                "prediction_method": prediction_method,
                                "ml_model_loaded": model_info["loaded"]
                            }
                        ) as trace:
                            langfuse.score(
                                trace_id=trace.id,
                                name="extraction-score",
                                value=1,
                                comment="User input processed"
                            )
                        langfuse.flush()
                    except Exception:
                        pass
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# Footer
st.markdown("---")
