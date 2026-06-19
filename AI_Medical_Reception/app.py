import os
import pandas as pd
import streamlit as st
import speech_recognition as sr
import pyttsx3

from pypdf import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = "data"

st.set_page_config(
    page_title="AI Medical Receptionist",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 AI Medical Receptionist Agent")


@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def read_csv(path):
    df = pd.read_csv(path)
    text = f"\nFile Name: {os.path.basename(path)}\n"
    text += df.to_string(index=False)
    return text


def read_pdf(path):
    text = f"\nFile Name: {os.path.basename(path)}\n"
    reader = PdfReader(path)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def load_documents():
    documents = []

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        try:
            if file.endswith(".csv"):
                content = read_csv(path)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": file}
                    )
                )

            elif file.endswith(".pdf"):
                content = read_pdf(path)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={"source": file}
                    )
                )

        except Exception as e:
            st.error(f"Error reading {file}: {e}")

    return documents


@st.cache_resource
def create_vector_store():
    documents = load_documents()

    if len(documents) == 0:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)
    embeddings = load_embeddings()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


def search_context(question, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n"

    return context, docs


def generate_answer(question, context, tokenizer, model):
    prompt = f"""
You are an AI Medical Receptionist Agent.

Your job:
- Answer patient questions politely.
- Use only the hospital knowledge base.
- Give short and clear answers.
- If answer is not available, say: Please contact hospital reception.

Hospital Knowledge Base:
{context}

Patient Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def speak_text(text):
    engine = None

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    except RuntimeError:
        if engine:
            try:
                engine.stop()
            except:
                pass
        st.warning("Voice engine busy hai. Dobara Speak button click karo.")

    except Exception as e:
        st.error(f"Voice error: {e}")


def listen_voice():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak now")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

        question = recognizer.recognize_google(audio)
        return question

    except sr.UnknownValueError:
        return "Sorry, I could not understand your voice."

    except sr.RequestError:
        return "Speech recognition service error."

    except Exception as e:
        return f"Voice error: {e}"


if "exit_app" not in st.session_state:
    st.session_state.exit_app = False

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""


tokenizer, model = load_llm()
vector_store = create_vector_store()

if vector_store is None:
    st.warning("Please add CSV or PDF files inside data folder.")
    st.stop()

st.sidebar.markdown("### 🏥 AI Medical Dashboard")
menu = st.sidebar.radio(
    "",
    [
        "💬 Text Chat",
        "🔊 Text Input Voice Answer",
        "🎤 Voice Chat",
        "📅 Book Appointment",
        "📁 View Data"
    ]
)

if st.sidebar.button("❌ Exit Agent"):
    st.session_state.exit_app = True
if st.session_state.exit_app:
    st.error("AI Medical Receptionist Agent stopped.")
    st.stop()


if menu == "💬 Text Chat":
    st.subheader("💬 Text Medical Receptionist")

    question = st.text_input(
        "Ask hospital query:",
        placeholder="Example: Who is the cardiologist?"
    )

    if st.button("Ask"):
        if question.strip() == "":
            st.warning("Please enter your question.")
        else:
            context, docs = search_context(question, vector_store)
            answer = generate_answer(question, context, tokenizer, model)

            st.session_state.last_answer = answer

            st.subheader("🤖 Answer")
            st.write(answer)

            with st.expander("Retrieved LangChain Context"):
                for doc in docs:
                    st.write("Source:", doc.metadata.get("source"))
                    st.write(doc.page_content)
                    st.divider()


elif menu == "🔊 Text Input Voice Answer":
    st.subheader("🔊 Type Question → Agent Voice Answer")

    question = st.text_input(
        "Type your hospital query:",
        placeholder="Example: What is MRI cost?"
    )

    if st.button("Generate Answer"):
        if question.strip() == "":
            st.warning("Please enter your question.")
        else:
            context, docs = search_context(question, vector_store)
            answer = generate_answer(question, context, tokenizer, model)

            st.session_state.last_answer = answer

            st.subheader("🤖 Answer")
            st.write(answer)

            with st.expander("Retrieved LangChain Context"):
                for doc in docs:
                    st.write("Source:", doc.metadata.get("source"))
                    st.write(doc.page_content)
                    st.divider()

    if st.session_state.last_answer:
        if st.button("🔊 Speak Answer"):
            speak_text(st.session_state.last_answer)


elif menu == "🎤 Voice Chat":
    st.subheader("🎤 Voice Medical Receptionist")

    if st.button("🎙️ Speak Now"):
        question = listen_voice()

        st.subheader("🗣️ Your Question")
        st.write(question)

        if "sorry" not in question.lower() and "error" not in question.lower():
            context, docs = search_context(question, vector_store)
            answer = generate_answer(question, context, tokenizer, model)

            st.session_state.last_answer = answer

            st.subheader("🤖 Answer")
            st.write(answer)

            with st.expander("Retrieved LangChain Context"):
                for doc in docs:
                    st.write("Source:", doc.metadata.get("source"))
                    st.write(doc.page_content)
                    st.divider()

    if st.session_state.last_answer:
        if st.button("🔊 Speak Voice Answer"):
            speak_text(st.session_state.last_answer)


elif menu == "📁 View Data":
    st.subheader("📁 Hospital Data Files")

    files = os.listdir(DATA_DIR)

    if len(files) == 0:
        st.warning("No files found in data folder.")
    else:
        selected_file = st.selectbox("Select file", files)
        path = os.path.join(DATA_DIR, selected_file)

        if selected_file.endswith(".csv"):
            df = pd.read_csv(path)
            st.dataframe(df)

        elif selected_file.endswith(".pdf"):
            st.info("PDF file added in knowledge base.")
            st.write(selected_file)


elif menu == "📅 Book Appointment":
    st.subheader("📅 Book Appointment")

    patient_name = st.text_input("Patient Name")

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120
    )

    phone = st.text_input("Phone Number")

    department = st.text_input(
        "Department",
        placeholder="Example: Cardiology"
    )

    doctor_name = st.text_input(
        "Doctor Name",
        placeholder="Example: Dr Sharma"
    )

    appointment_date = st.date_input("Appointment Date")
    appointment_time = st.time_input("Appointment Time")

    reason = st.text_area("Reason / Symptoms")

    if st.button("📅 Book Appointment"):
        if (
            patient_name.strip() == "" or
            phone.strip() == "" or
            doctor_name.strip() == ""
        ):
            st.warning("Please fill Patient Name, Phone Number and Doctor Name.")

        else:
            appointment_id = "A" + pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

            new_appointment = pd.DataFrame({
                "AppointmentID": [appointment_id],
                "PatientName": [patient_name],
                "Age": [age],
                "Phone": [phone],
                "Department": [department],
                "Doctor": [doctor_name],
                "Date": [str(appointment_date)],
                "Time": [str(appointment_time)],
                "Reason": [reason],
                "Status": ["Confirmed"]
            })

            appointment_file = os.path.join(DATA_DIR, "appointments.csv")

            if os.path.exists(appointment_file):
                old_df = pd.read_csv(appointment_file)
                final_df = pd.concat(
                    [old_df, new_appointment],
                    ignore_index=True
                )
            else:
                final_df = new_appointment

            final_df.to_csv(appointment_file, index=False)

            st.success(f"✅ Appointment Booked Successfully! ID: {appointment_id}")

            st.dataframe(new_appointment)

            if st.button("🔊 Speak Confirmation"):
                speak_text(
                    f"Appointment booked successfully for {patient_name}"
                )