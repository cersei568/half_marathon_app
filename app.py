
import os
import re
import json
import joblib
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from langfuse import Langfuse
from langfuse.openai import OpenAI as LangfuseOpenAI
from langfuse import observe


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


MODEL_URL = "https://cersei568.fra1.digitaloceanspaces.com/models/best_finalized_model.pkl"
MODEL_PATH = "models/best_finalized_model.pkl"


if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set in .env — put your OpenAI key there and restart.")
    st.stop()


os.makedirs("models", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model from DigitalOcean Spaces...")
    try:
        r = requests.get(MODEL_URL)
        if r.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
        else:
            st.error(f"Failed to download model. HTTP {r.status_code}")
            st.stop()
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()


try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

openai_client = OpenAI(api_key=OPENAI_API_KEY)


langfuse_enabled = True
try:
    langfuse = Langfuse()
except Exception as e:
    st.info(f"Langfuse disabled: {e}")
    langfuse_enabled = False


st.title("Predict Your Half-Marathon Time 🏃")
st.write("Let's predict your half-marathon time! Write something like: 'Hi, I'm Julia, female. I'm 34 and I run 5 km in 24:30 minutes'.")


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


user_prompt = st.text_area("Describe yourself:", height=120)

if st.button("Extract & Predict"):
    if not user_prompt or len(user_prompt.strip()) < 3:
        st.warning("Please provide a short description (age, gender, 5K).")
    else:
        
        assistant_text = get_llm_output(user_prompt)

        parsed = fallback_parse(assistant_text)
        if parsed["gender"] in ("no data", None) or parsed["age"] in ("no data", None) or parsed["pace_5k"] in ("no data", None):
            st.warning("Missing required info. Please provide gender, age, and 5K pace.")
        else:
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
                h = int(pred_seconds // 3600)
                m = int((pred_seconds % 3600) // 60)
                s = int(pred_seconds % 60)
                pretty = f"{h}h {m}m {s}s" if h > 0 else f"{m}m {s}s"
                st.success(f"Estimated finish time: **{pretty}**")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")

           
    if langfuse_enabled:
        dataset_name = "halfmarathon_dataset"
    try:
        
        try:
            dataset = langfuse.get_dataset(dataset_name)
        except Exception:
            dataset = langfuse.create_dataset(dataset_name, description="User half-marathon inputs")

        
        dataset.insert(
            input=user_prompt,
            expected_output=pretty
        )

       
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


import matplotlib.pyplot as plt


if 'pred_seconds' in locals():
    total_km = 21.0975  
    pace_per_km_sec = pred_seconds / total_km
    pace_minutes = int(pace_per_km_sec // 60)
    pace_seconds = int(pace_per_km_sec % 60)
    pace_str = f"{pace_minutes}m {pace_seconds}s per km"

    st.subheader("Estimated pace per kilometer")
    st.write(pace_str)

   
    distances = [round(i,1) for i in range(1, 22)]  
    times = [i * pace_per_km_sec for i in distances]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(distances, times, marker='o')
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Estimated cumulative time per km")
    ax.grid(True)

  
    y_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    st.pyplot(fig)