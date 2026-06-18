import streamlit as st
import speech_recognition as sr
import streamlit.components.v1 as components
import re

if "stop" not in st.session_state:
    st.session_state.stop = False

def speak(text):
    components.html(f"""
    <script>
    var msg = new SpeechSynthesisUtterance("{text}");
    msg.lang = "en-US";
    window.speechSynthesis.speak(msg);
    </script>
    """, height=0)

def listen():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        return r.recognize_google(audio).lower()
    except:
        return ""

def extract_calculation(text):
    pattern = r'(\d+)\s*(plus|add|minus|subtract|multiply|into|divide)\s*(\d+)'
    match = re.search(pattern, text)

    if match:
        num1 = float(match.group(1))
        operator = match.group(2)
        num2 = float(match.group(3))
        return num1, operator, num2

    return None, None, None

def calculate(a, b, op):
    if "plus" in op or "add" in op:
        return a + b
    elif "minus" in op or "subtract" in op:
        return a - b
    elif "multiply" in op or "into" in op:
        return a * b
    elif "divide" in op:
        return "Cannot divide by zero" if b == 0 else a / b

    return "Invalid operator"

st.set_page_config(page_title="AI Voice Calculator", page_icon="🎙️")

st.title("🎙️ AI Voice Calculator Agent")
st.write("Speak calculation in one line: **10 plus 20**")

col1, col2 = st.columns(2)

with col1:
    start_btn = st.button("🎤 Start Calculation")

with col2:
    stop_btn = st.button("🛑 Exit")

if stop_btn:
    st.session_state.stop = True
    st.error("🛑 Calculator Closed")
    st.stop()

if start_btn and not st.session_state.stop:

    speak("Speak your calculation. Example 10 plus 20")

    query = listen()

    if query in ["exit", "stop", "quit"]:
        st.warning("🛑 Calculator Closed By Voice")
        st.stop()

    st.write("### Voice Command")
    st.write(query)

    num1, operator, num2 = extract_calculation(query)

    if num1 is not None:
        result = calculate(num1, num2, operator)

        st.write("First Value:", num1)
        st.write("Operator:", operator)
        st.write("Second Value:", num2)

        st.success(f"✅ Result: {result}")

        speak(f"Your answer is {result}")

    else:
        st.error("❌ Calculation not understood")
        speak("Calculation not understood")