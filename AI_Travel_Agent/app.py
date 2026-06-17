import streamlit as st
import pandas as pd
import plotly.express as px
import re
import tempfile
import whisper
import pyttsx3

st.set_page_config(page_title="AI Travel Agent", page_icon="🌍", layout="wide")

st.title("🌍 AI Voice Travel Agent")
st.write("Text ya voice se travel query ask karo.")

@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
    engine.say(text)
    engine.runAndWait()

def voice_to_text(audio_file):
    model = load_whisper_model()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_path = temp_audio.name

    result = model.transcribe(temp_path)
    return result["text"]

def clean_cost(value):
    if pd.isna(value):
        return 0
    value = str(value)
    value = re.sub(r"[^\d.]", "", value)
    return float(value) if value else 0

def load_dataset(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def extract_budget(query):
    numbers = re.findall(r"\d+", query)
    if numbers:
        return int(numbers[-1])
    return None

def detect_gender(query):
    query = query.lower()
    if "female" in query or "girl" in query or "women" in query:
        return "Female"
    if "male" in query or "boy" in query or "men" in query:
        return "Male"
    return None

def detect_destination(query, destinations):
    query = query.lower()
    for dest in destinations:
        if str(dest).lower() in query:
            return dest
    return None

def ai_travel_agent(query, df):
    query_lower = query.lower()

    budget = extract_budget(query_lower)
    gender = detect_gender(query_lower)
    destination = detect_destination(query_lower, df["Destination"].dropna().unique())

    data = df.copy()

    if destination:
        data = data[data["Destination"] == destination]

    if gender:
        gender_data = data[data["Traveler gender"].str.lower() == gender.lower()]
        if not gender_data.empty:
            data = gender_data

    data["Total Cost"] = data["Accommodation cost"] + data["Transportation cost"]

    if budget:
        budget_data = data[data["Total Cost"] <= budget]
        if not budget_data.empty:
            data = budget_data

    if data.empty:
        return None, "Sorry, is query ke according data nahi mila."

    best_plan = data.sort_values("Total Cost").iloc[0]

    answer = f"""
Best travel plan found.

Destination: {best_plan['Destination']}
Duration: {best_plan['Duration (days)']} days
Traveler Gender: {best_plan['Traveler gender']}
Hotel Type: {best_plan['Accommodation type']}
Hotel Cost: ₹{best_plan['Accommodation cost']:,.0f}
Transport: {best_plan['Transportation type']}
Transport Cost: ₹{best_plan['Transportation cost']:,.0f}
Total Estimated Cost: ₹{best_plan['Total Cost']:,.0f}

AI Reason:
Maine aapki query se destination, gender aur budget ko samjha.
Uske baad dataset me se sabse low-cost matching travel option select kiya.
"""

    return best_plan, answer

uploaded_file = st.file_uploader(
    "Upload your Travel Excel/CSV file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    df = load_dataset(uploaded_file)

    required_columns = [
        "Destination",
        "Duration (days)",
        "Traveler gender",
        "Accommodation type",
        "Accommodation cost",
        "Transportation type",
        "Transportation cost"
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        st.error("Missing columns:")
        st.write(missing)
        st.stop()

    df["Accommodation cost"] = df["Accommodation cost"].apply(clean_cost)
    df["Transportation cost"] = df["Transportation cost"].apply(clean_cost)

    st.success("Dataset uploaded successfully!")

    tab1, tab2, tab3 = st.tabs(["🤖 AI Voice Agent", "📊 Analytics", "📄 Dataset"])

    with tab1:
        st.subheader("🎙️ Ask by Voice")

        voice_file = st.audio_input("Speak your travel query")

        if voice_file is not None:
            st.audio(voice_file)

            if st.button("Convert Voice & Ask Agent"):
                query_text = voice_to_text(voice_file)

                st.write("You said:")
                st.info(query_text)

                plan, answer = ai_travel_agent(query_text, df)

                if plan is None:
                    st.error(answer)
                    speak_text(answer)
                else:
                    st.success(answer)
                    speak_text(answer)

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric("Destination", plan["Destination"])

                    with c2:
                        st.metric("Days", int(plan["Duration (days)"]))

                    with c3:
                        st.metric("Hotel", plan["Accommodation type"])

                    with c4:
                        st.metric("Total Cost", f"₹{plan['Total Cost']:,.0f}")

        st.divider()

        st.subheader("⌨️ Ask by Text")

        user_query = st.text_input(
            "Enter your travel query",
            placeholder="Example: I want to travel to Paris under 50000 for female traveler"
        )

        if st.button("Ask Agent"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                plan, answer = ai_travel_agent(user_query, df)

                if plan is None:
                    st.error(answer)
                    speak_text(answer)
                else:
                    st.info(answer)
                    speak_text(answer)

                    c1, c2, c3, c4 = st.columns(4)

                    with c1:
                        st.metric("Destination", plan["Destination"])

                    with c2:
                        st.metric("Days", int(plan["Duration (days)"]))

                    with c3:
                        st.metric("Hotel", plan["Accommodation type"])

                    with c4:
                        st.metric("Total Cost", f"₹{plan['Total Cost']:,.0f}")

        st.divider()

        st.subheader("Manual Travel Planner")

        col1, col2, col3 = st.columns(3)

        with col1:
            destination = st.selectbox("Destination", sorted(df["Destination"].dropna().unique()))

        with col2:
            gender = st.selectbox("Gender", sorted(df["Traveler gender"].dropna().unique()))

        with col3:
            travelers = st.number_input("Number of Travelers", min_value=1, value=1)

        filtered = df[
            (df["Destination"] == destination) &
            (df["Traveler gender"] == gender)
        ]

        if filtered.empty:
            filtered = df[df["Destination"] == destination]

        if not filtered.empty:
            filtered = filtered.copy()
            filtered["Total Cost"] = filtered["Accommodation cost"] + filtered["Transportation cost"]

            best = filtered.sort_values("Total Cost").iloc[0]
            final_cost = best["Total Cost"] * travelers

            st.success(f"Best plan total cost for {travelers} traveler(s): ₹{final_cost:,.0f}")
            st.dataframe(filtered)

    with tab2:
        st.subheader("Male and Female Travelers Count")

        gender_count = df["Traveler gender"].value_counts().reset_index()
        gender_count.columns = ["Gender", "Count"]

        st.dataframe(gender_count)

        fig = px.bar(
            gender_count,
            x="Gender",
            y="Count",
            text="Count",
            title="Male vs Female Travelers"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Destination Wise Average Cost")

        df["Total Cost"] = df["Accommodation cost"] + df["Transportation cost"]

        cost_chart = df.groupby("Destination")["Total Cost"].mean().reset_index()

        fig2 = px.bar(
            cost_chart,
            x="Destination",
            y="Total Cost",
            title="Average Travel Cost by Destination"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Uploaded Dataset")
        st.dataframe(df)

else:
    st.warning("Please upload your travel dataset file.")