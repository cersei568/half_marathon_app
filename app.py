import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Half Marathon Time Predictor",
    page_icon="🏃‍♀️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Safe imports
def safe_import(module_name, package=None):
    """Safely import a module and return whether it was successful"""
    try:
        if package:
            module = __import__(module_name, fromlist=[package])
            return getattr(module, package)
        else:
            return __import__(module_name)
    except ImportError:
        return None

# Try importing optional dependencies
dotenv = safe_import('dotenv')
openai_module = safe_import('openai')
plotly_go = safe_import('plotly.graph_objects')

# Load environment if available
if dotenv:
    dotenv.load_dotenv()

# Custom CSS
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

# Simple Half Marathon Prediction Model
class HalfMarathonPredictor:
    """Mathematical model for half marathon time prediction"""
    
    def predict(self, X):
        """Predict half marathon times"""
        predictions = []
        
        for _, row in X.iterrows():
            age = float(row['Wiek'])
            gender = int(row['Płeć'])  # 0 = male, 1 = female
            time_5k_seconds = float(row['5 km Czas'])
            
            # Base conversion ratio from 5K to half marathon
            base_ratio = 4.65
            
            # Gender adjustment (women typically have better endurance)
            if gender == 1:  # female
                base_ratio *= 0.98
            
            # Age adjustments
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
            
            # Fitness level adjustment based on 5K time
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

# Get API key safely
def get_api_key():
    """Get OpenAI API key from environment or secrets"""
    # Try environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # Try Streamlit secrets
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return None

# Initialize components
@st.cache_resource
def initialize_app():
    """Initialize the app components"""
    
    # Initialize prediction model
    predictor = HalfMarathonPredictor()
    
    # Initialize OpenAI client if possible
    openai_client = None
    api_key = get_api_key()
    
    if openai_module and api_key:
        try:
            openai_client = openai_module.OpenAI(api_key=api_key)
        except Exception as e:
            st.warning(f"Could not initialize OpenAI: {e}")
    
    return predictor, openai_client

predictor, openai_client = initialize_app()

# AI text parsing function
def parse_with_ai(text):
    """Parse user input with AI if available"""
    if not openai_client or not text.strip():
        return {}
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Extract age (number), gender (male/female), and 5K time (mm:ss) from text. Return JSON with keys: age, gender, time_5k. Use null for missing data."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=150,
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
        
    except Exception as e:
        st.error(f"AI parsing failed: {e}")
        return {}

# Results display function
def display_results(pred_seconds, age, gender, time_5k):
    """Display prediction results"""
    
    # Format time
    hours = int(pred_seconds // 3600)
    minutes = int((pred_seconds % 3600) // 60)
    seconds = int(pred_seconds % 60)
    time_str = f"{hours}h {minutes}m {seconds}s" if hours > 0 else f"{minutes}m {seconds}s"
    
    # Main prediction card
    st.markdown(f"""
    <div class="prediction-card">
        🎉 Your Predicted Half Marathon Time: {time_str} 🎉
    </div>
    """, unsafe_allow_html=True)
    
    # Profile display
    st.markdown("### ✅ Your Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>👤 Gender</h4>
            <p style="font-size: 1.2rem; color: #667eea;">{gender}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎂 Age</h4>
            <p style="font-size: 1.2rem; color: #667eea;">{age} years</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⏱️ 5K Time</h4>
            <p style="font-size: 1.2rem; color: #667eea;">{time_5k}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pace analysis
    total_km = 21.0975
    pace_per_km_sec = pred_seconds / total_km
    pace_minutes = int(pace_per_km_sec // 60)
    pace_seconds_part = int(pace_per_km_sec % 60)
    pace_str = f"{pace_minutes}:{pace_seconds_part:02d}"
    
    st.markdown("### 📊 Race Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏃‍♀️ Average Pace</h4>
            <p style="font-size: 1.5rem; color: #11998e; font-weight: bold;">{pace_str} /km</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 Distance</h4>
            <p style="font-size: 1.5rem; color: #11998e; font-weight: bold;">21.1 km</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Simple chart with matplotlib
    st.markdown("### 📈 Projected Progress")
    
    import matplotlib.pyplot as plt
    
    distances = list(range(1, 22))
    cumulative_times = [i * pace_per_km_sec / 60 for i in distances]  # in minutes
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, cumulative_times, 'o-', color='#667eea', linewidth=3, markersize=6)
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Cumulative Time (minutes)', fontsize=12)
    ax.set_title('Your Projected Race Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22)
    
    # Format y-axis to show time
    y_ticks = ax.get_yticks()
    y_labels = [f"{int(t//60)}:{int(t%60):02d}" if t >= 60 else f"{int(t)}m" for t in y_ticks]
    ax.set_yticklabels(y_labels)
    
    st.pyplot(fig)
    
    # Split times table
    st.markdown("### 📋 Kilometer Splits")
    
    split_data = []
    for i in range(1, 22):
        cumulative_seconds = i * pace_per_km_sec
        cum_minutes = int(cumulative_seconds // 60)
        cum_seconds_part = int(cumulative_seconds % 60)
        
        split_data.append({
            "Kilometer": i,
            "Split": f"{pace_minutes}:{pace_seconds_part:02d}",
            "Total Time": f"{cum_minutes}:{cum_seconds_part:02d}"
        })
    
    # Show table in two columns
    mid_point = len(split_data) // 2
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            pd.DataFrame(split_data[:mid_point + 1]),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.dataframe(
            pd.DataFrame(split_data[mid_point + 1:]),
            use_container_width=True,
            hide_index=True
        )

# Main app
def main():
    """Main application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏃‍♀️ Half Marathon Time Predictor 🏃‍♂️</h1>
        <p style="font-size: 1.2rem; margin: 0;">Smart prediction based on your 5K performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 How it works</h3>
            <p>Our model analyzes:</p>
            <ul>
                <li>🎂 Your age</li>
                <li>👤 Your gender</li>
                <li>⏱️ Your 5K time</li>
            </ul>
            <p>Then predicts your half-marathon time using sports science!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-text">
            <strong>💡 Examples:</strong><br>
            "I'm Sarah, 28, female, run 5K in 23:45"<br>
            "Male, 35 years old, 5K PB is 20:30"<br>
            "42-year-old woman, 5km takes me 26 minutes"
        </div>
        """)
    
    with col1:
        st.markdown("### 📝 Enter Your Details")
        
        # Manual input form
        col_age, col_gender = st.columns(2)
        
        with col_age:
            age = st.number_input("Age", min_value=16, max_value=80, value=30, step=1)
        
        with col_gender:
            gender = st.selectbox("Gender", ["Female", "Male"])
        
        time_input = st.text_input(
            "5K Time (mm:ss)", 
            placeholder="e.g., 25:30",
            help="Enter in minutes:seconds format"
        )
        
        # AI input option
        if openai_client:
            st.markdown("---")
            st.markdown("### 🤖 Or Use AI Description")
            
            ai_text = st.text_area(
                "Describe yourself naturally:",
                placeholder="I'm John, 32 years old, male. My 5K best is 22:15...",
                height=100
            )
            
            use_ai = st.checkbox("Parse the description above with AI")
        else:
            use_ai = False
            st.info("💡 Add OpenAI API key for AI-powered text parsing")
        
        # Predict button
        if st.button("🔮 Predict My Half Marathon Time", use_container_width=True):
            
            # Use AI parsing if requested
            if use_ai and openai_client and 'ai_text' in locals() and ai_text.strip():
                ai_result = parse_with_ai(ai_text)
                
                if ai_result:
                    if ai_result.get('age'):
                        age = int(ai_result['age'])
                    if ai_result.get('gender'):
                        gender = str(ai_result['gender']).title()
                    if ai_result.get('time_5k'):
                        time_input = str(ai_result['time_5k'])
            
            # Validate inputs
            if not time_input:
                st.warning("⚠️ Please enter your 5K time")
                return
            
            # Parse time
            time_match = re.match(r'(\d{1,2}):(\d{2})', time_input.strip())
            if not time_match:
                st.error("❌ Please enter time in MM:SS format (e.g., 25:30)")
                return
            
            try:
                minutes, seconds = map(int, time_match.groups())
                total_seconds = minutes * 60 + seconds
                
                # Validate reasonable time range
                if total_seconds < 10*60 or total_seconds > 60*60:  # 10 to 60 minutes
                    st.error("❌ Please enter a realistic 5K time (10-60 minutes)")
                    return
                
                # Prepare data for prediction
                gender_code = 0 if gender.lower() == "male" else 1
                
                input_data = pd.DataFrame([{
                    "Wiek": age,
                    "Płeć": gender_code,
                    "5 km Czas": total_seconds
                }])
                
                # Make prediction
                prediction = predictor.predict(input_data)[0]
                
                # Display results
                display_results(prediction, age, gender, time_input)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

# Footer
def show_footer():
    """Show app footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p>🏃‍♀️ Train smart, race faster! 🏃‍♂️</p>
        <p style="font-size: 0.8rem;">
            Predictions based on sports science research. Actual results may vary based on 
            training, weather conditions, and race day factors.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
    show_footer()