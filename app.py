import os
import re
import json
import joblib
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Only import these if available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langfuse import Langfuse
    from langfuse import observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    # Create dummy decorator if langfuse is not available
    def observe():
        def decorator(func):
            return func
        return decorator

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Half Marathon Time Predictor",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Safe way to get API keys
def get_api_key():
    """Safely get OpenAI API key from various sources"""
    # Try environment variables first (works locally with .env)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Try Streamlit secrets (works in deployment)
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        return api_key
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    
    # If nothing found, return empty string
    return ""

def get_openai_model():
    """Safely get OpenAI model name"""
    # Try environment variables first
    model = os.getenv("OPENAI_MODEL")
    if model:
        return model
    
    # Try Streamlit secrets
    try:
        model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
        return model
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    
    # Default fallback
    return "gpt-4o-mini"

# Get configuration
OPENAI_API_KEY = get_api_key()
OPENAI_MODEL = get_openai_model()

MODEL_URL = "https://cersei568.fra1.digitaloceanspaces.com/models/best_finalized_model.pkl"
MODEL_PATH = "models/best_finalized_model.pkl"

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
    
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model loading with progress bar
@st.cache_resource
def load_model():
    os.makedirs("models", exist_ok=True)
    
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
                    st.success("✅ Model downloaded successfully!")
                else:
                    st.error(f"❌ Failed to download model. HTTP {r.status_code}")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error downloading model: {e}")
                st.stop()
    
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

pipeline = load_model()

# Initialize OpenAI client if available
openai_client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")

# Show API key status
if not OPENAI_API_KEY:
    st.markdown("""
    <div class="warning-card">
        <h4>⚠️ OpenAI API Key Not Found</h4>
        <p>The app will work in manual input mode. To enable AI text processing:</p>
        <ul>
            <li><strong>For local development:</strong> Create a <code>.env</code> file with <code>OPENAI_API_KEY=your_key_here</code></li>
            <li><strong>For deployment:</strong> Add your API key to Streamlit secrets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Langfuse setup
langfuse_enabled = False
if LANGFUSE_AVAILABLE:
    try:
        langfuse = Langfuse()
        langfuse_enabled = True
    except Exception as e:
        langfuse_enabled = False

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
    if not openai_client:
        return ""
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

def display_prediction_results(pred_seconds, age, gender, pace_5k_str):
    """Display prediction results with visualizations"""
    h = int(pred_seconds // 3600)
    m = int((pred_seconds % 3600) // 60)
    s = int(pred_seconds % 60)
    pretty = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
    
    # Beautiful prediction display
    st.markdown(f"""
    <div class="prediction-card">
        🎉 Your Predicted Half Marathon Time: {pretty} 🎉
    </div>
    """, unsafe_allow_html=True)
    
    # Display extracted information
    st.markdown("### ✅ Your Profile")
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
    
    # Visualization
    st.markdown("### 📈 Projected Split Times")
    
    distances = list(range(1, 22))
    times_minutes = [(i * pace_per_km_sec) / 60 for i in distances]
    
    if PLOTLY_AVAILABLE:
        # Interactive chart using Plotly
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
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(distances, times_minutes, marker='o', color='#667eea', linewidth=3, markersize=6)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Cumulative Time (minutes)")
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
    
    # Display in columns for better readability
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

# Header
st.markdown("""
<div class="main-header">
    <h1>🏃‍♀️ Half Marathon Time Predictor 🏃‍♂️</h1>
    <p style="font-size: 1.2rem; margin: 0;">AI-powered prediction based on your 5K performance</p>
</div>
""", unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 How it works</h3>
        <p>Our AI analyzes your:</p>
        <ul>
            <li>🎂 Age</li>
            <li>👤 Gender</li>
            <li>⏱️ 5K running time</li>
        </ul>
        <p>Then predicts your half-marathon finish time using machine learning!</p>
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
    
    # Show appropriate input method based on API key availability
    if not openai_client:
        st.info("🔧 Manual input mode")
        
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
        
        predict_button = st.button("🔮 Predict My Half Marathon Time", use_container_width=True)
        
        # Manual processing
        if predict_button:
            if not time_input:
                st.warning("Please enter your 5K time")
            else:
                # Parse time input
                time_pattern = re.match(r'(\d{1,2}):(\d{2})', time_input)
                if not time_pattern:
                    st.error("Please enter time in MM:SS format (e.g., 25:30)")
                else:
                    minutes, seconds = map(int, time_pattern.groups())
                    pace_sec = minutes * 60 + seconds
                    gender_num = 0 if gender.lower() == "male" else 1
                    
                    # Make prediction
                    X_new = pd.DataFrame([{
                        "Wiek": age,
                        "Płeć": gender_num,
                        "5 km Czas": pace_sec
                    }])
                    
                    try:
                        pred_seconds = pipeline.predict(X_new)[0]
                        display_prediction_results(pred_seconds, age, gender, time_input)
                    except Exception as e:
                        st.error(f"❌ Model prediction failed: {e}")
    
    else:
        # AI-powered text input
        st.success("🤖 AI text processing enabled")
        user_prompt = st.text_area(
            "Describe your running profile:",
            height=120,
            placeholder="Example: I'm John, 32 years old, male. My best 5K time is 22:15...",
            help="Include your age, gender, and 5K running time for the most accurate prediction"
        )
        
        predict_button = st.button("🔮 Predict My Half Marathon Time", use_container_width=True)

# AI-powered prediction logic
if openai_client and predict_button and 'user_prompt' in locals():
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

            X_new = pd.DataFrame([{
                "Wiek": age_val,
                "Płeć": gender_num,
                "5 km Czas": pace_sec
            }])

            try:
                pred_seconds = pipeline.predict(X_new)[0]
                display_prediction_results(pred_seconds, age_val, parsed["gender"].title(), parsed["pace_5k"])
                
                # Langfuse tracking
                if langfuse_enabled:
                    try:
                        h = int(pred_seconds // 3600)
                        m = int((pred_seconds % 3600) // 60)
                        s = int(pred_seconds % 60)
                        pretty = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
                        
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
                            metadata={"model": OPENAI_MODEL}
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
                st.error(f"❌ Model prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🏃‍♀️ Train smart, race faster! 🏃‍♂️</p>
    <p style="font-size: 0.8rem;">Remember: This is a prediction based on statistical models. Your actual performance may vary based on training, weather, and race day conditions.</p>
</div>
""", unsafe_allow_html=True)