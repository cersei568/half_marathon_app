import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import json

# Page config must be first
st.set_page_config(
    page_title="Half Marathon Predictor",
    page_icon="🏃‍♀️",
    layout="wide"
)

# CSS styling
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
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    width: 100%;
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

def predict_half_marathon(age, gender, time_5k_seconds):
    """
    Simple mathematical model for half marathon prediction
    Based on sports science research
    """
    # Base conversion ratio from 5K to half marathon
    base_ratio = 4.65
    
    # Gender adjustment (women typically have slightly better endurance)
    if gender.lower() == "female":
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
    return predicted_seconds

def parse_with_openai(text, api_key):
    """Parse user text with OpenAI API"""
    if not api_key or not text.strip():
        return None
    
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", 
                    "content": "Extract age (number), gender (male/female), and 5K time (in mm:ss format) from the user's text. Return only a JSON object with keys: age, gender, time_5k. Use null for missing information."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=100,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        st.error(f"AI parsing failed: {e}")
        return None

def display_results(pred_seconds, age, gender, time_5k_str):
    """Display the prediction results"""
    
    # Format the predicted time
    hours = int(pred_seconds // 3600)
    minutes = int((pred_seconds % 3600) // 60)
    seconds = int(pred_seconds % 60)
    
    if hours > 0:
        time_display = f"{hours}h {minutes}m {seconds}s"
    else:
        time_display = f"{minutes}m {seconds}s"
    
    # Main result card
    st.markdown(f"""
    <div class="prediction-card">
        🎉 Your Predicted Half Marathon Time: {time_display} 🎉
    </div>
    """, unsafe_allow_html=True)
    
    # Profile information
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
            <p style="font-size: 1.2rem; color: #667eea;">{time_5k_str}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pace analysis
    total_km = 21.0975
    pace_per_km_sec = pred_seconds / total_km
    pace_minutes = int(pace_per_km_sec // 60)
    pace_seconds = int(pace_per_km_sec % 60)
    pace_str = f"{pace_minutes}:{pace_seconds:02d}"
    
    st.markdown("### 📊 Race Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🏃‍♀️ Average Pace</h4>
            <p style="font-size: 1.5rem; color: #11998e; font-weight: bold;">{pace_str} per km</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📏 Total Distance</h4>
            <p style="font-size: 1.5rem; color: #11998e; font-weight: bold;">21.1 km</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress chart
    st.markdown("### 📈 Your Race Progress")
    
    distances = list(range(1, 22))
    cumulative_times = [i * pace_per_km_sec for i in distances]
    
    # Create chart data
    chart_data = pd.DataFrame({
        'Distance (km)': distances,
        'Time (minutes)': [t/60 for t in cumulative_times]
    })
    
    st.line_chart(chart_data.set_index('Distance (km)'), height=400)
    
    # Split times table
    st.markdown("### 📋 Kilometer Split Times")
    
    split_data = []
    for i in range(1, 22):
        cumulative_seconds = i * pace_per_km_sec
        cum_minutes = int(cumulative_seconds // 60)
        cum_secs = int(cumulative_seconds % 60)
        
        split_data.append({
            "KM": i,
            "Split Time": f"{pace_minutes}:{pace_seconds:02d}",
            "Total Time": f"{cum_minutes}:{cum_secs:02d}"
        })
    
    # Display table in two columns
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

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏃‍♀️ Half Marathon Time Predictor 🏃‍♂️</h1>
        <p style="font-size: 1.2rem; margin: 0;">Smart prediction based on your 5K performance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    # Sidebar info
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 How It Works</h3>
            <p>Our predictor analyzes:</p>
            <ul>
                <li>🎂 Your age</li>
                <li>👤 Your gender</li>
                <li>⏱️ Your 5K time</li>
            </ul>
            <p>Then uses sports science to predict your half-marathon time!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="example-text">
            <strong>💡 Example descriptions:</strong><br>
            "I'm Sarah, 28, female, I run 5K in 23:45"<br><br>
            "Male, 35 years old, my 5K best is 20:30"<br><br>
            "42-year-old woman, takes me 26 minutes for 5km"
        </div>
        """)
    
    # Main input area
    with col1:
        st.markdown("### 📝 Enter Your Details")
        
        # Manual input section
        st.markdown("#### Option 1: Manual Input")
        
        col_age, col_gender = st.columns(2)
        
        with col_age:
            age = st.number_input("Age", min_value=16, max_value=80, value=30)
        
        with col_gender:
            gender = st.selectbox("Gender", ["Female", "Male"])
        
        time_input = st.text_input(
            "5K Time (mm:ss)", 
            placeholder="e.g., 25:30",
            help="Enter your 5K time in minutes:seconds format"
        )
        
        # AI input section
        st.markdown("#### Option 2: AI-Powered Description")
        
        # API key input
        api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key to enable AI text parsing"
        )
        
        ai_description = st.text_area(
            "Describe yourself naturally:",
            placeholder="I'm John, 32 years old, male. My 5K personal best is 22:15...",
            height=100,
            help="Describe your age, gender, and 5K time in natural language"
        )
        
        use_ai_parsing = st.checkbox("Use AI to parse the description above")
        
        # Predict button
        if st.button("🔮 Predict My Half Marathon Time"):
            
            final_age = age
            final_gender = gender
            final_time = time_input
            
            # Try AI parsing if requested
            if use_ai_parsing and api_key and ai_description.strip():
                with st.spinner("🤖 AI is analyzing your description..."):
                    ai_result = parse_with_openai(ai_description, api_key)
                    
                    if ai_result:
                        if ai_result.get('age'):
                            final_age = int(ai_result['age'])
                            st.success(f"✅ AI detected age: {final_age}")
                        
                        if ai_result.get('gender'):
                            final_gender = str(ai_result['gender']).title()
                            st.success(f"✅ AI detected gender: {final_gender}")
                        
                        if ai_result.get('time_5k'):
                            final_time = str(ai_result['time_5k'])
                            st.success(f"✅ AI detected 5K time: {final_time}")
            
            # Validation
            if not final_time:
                st.error("❌ Please enter your 5K time")
                return
            
            # Parse the time
            time_pattern = re.match(r'(\d{1,2}):(\d{2})', final_time.strip())
            if not time_pattern:
                st.error("❌ Please enter time in MM:SS format (e.g., 25:30)")
                return
            
            try:
                minutes, seconds = map(int, time_pattern.groups())
                total_seconds = minutes * 60 + seconds
                
                # Validate time range
                if total_seconds < 600 or total_seconds > 3600:  # 10-60 minutes
                    st.error("❌ Please enter a realistic 5K time (10-60 minutes)")
                    return
                
                # Make prediction
                prediction = predict_half_marathon(final_age, final_gender, total_seconds)
                
                # Display results
                display_results(prediction, final_age, final_gender, final_time)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🏃‍♀️ Train smart, race faster! 🏃‍♂️</p>
    <p style="font-size: 0.8rem;">
        Predictions based on sports science research. Actual performance may vary 
        based on training, weather, and race day conditions.
    </p>
</div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()