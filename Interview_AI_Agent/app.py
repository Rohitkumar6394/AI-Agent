import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    return generator, embedder


generator, embedder = load_models()

# -----------------------------
# Interview Questions
# -----------------------------
questions = [
    {
        "question": "Tell me about yourself.",
        "ideal": "Candidate should introduce education, skills, projects, and career goals clearly."
    },
    {
        "question": "What is Python?",
        "ideal": "Python is a high-level, interpreted programming language used for web development, AI, data science, automation, and scripting."
    },
    {
        "question": "What is Machine Learning?",
        "ideal": "Machine Learning is a branch of AI where systems learn patterns from data and make predictions or decisions without explicit programming."
    },
    {
        "question": "Explain your final year project.",
        "ideal": "Candidate should explain project title, problem, technology stack, features, working process, and outcome."
    },
    {
        "question": "Why should we hire you?",
        "ideal": "Candidate should explain skills, learning ability, projects, teamwork, and how they can contribute to the company."
    }
]

# -----------------------------
# Score Function
# -----------------------------
def calculate_score(user_answer, ideal_answer):
    user_embedding = embedder.encode(user_answer, convert_to_tensor=True)
    ideal_embedding = embedder.encode(ideal_answer, convert_to_tensor=True)

    similarity = util.cos_sim(user_embedding, ideal_embedding).item()
    score = round(similarity * 100, 2)

    if score < 0:
        score = 0

    return score


# -----------------------------
# Feedback Function
# -----------------------------
def generate_feedback(question, answer):
    prompt = f"""
    You are an interview evaluator.
    Question: {question}
    Candidate Answer: {answer}

    Give short feedback and improvement tips.
    """

    result = generator(prompt, max_length=150)
    return result[0]["generated_text"]


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Interview Agent", page_icon="🤖")

st.title("🤖 AI Interview Agent")
st.write("Practice interview with AI. No API key required.")

name = st.text_input("Enter your name")

if "current_question" not in st.session_state:
    st.session_state.current_question = 0

if "results" not in st.session_state:
    st.session_state.results = []

if name:
    index = st.session_state.current_question

    if index < len(questions):
        q = questions[index]

        st.subheader(f"Question {index + 1}")
        st.write(q["question"])

        answer = st.text_area("Your Answer")

        if st.button("Submit Answer"):
            if answer.strip() == "":
                st.warning("Please enter your answer.")
            else:
                score = calculate_score(answer, q["ideal"])
                feedback = generate_feedback(q["question"], answer)

                st.session_state.results.append({
                    "question": q["question"],
                    "answer": answer,
                    "score": score,
                    "feedback": feedback
                })

                st.session_state.current_question += 1
                st.rerun()

    else:
        st.success("Interview Completed!")

        total_score = sum(item["score"] for item in st.session_state.results)
        average_score = round(total_score / len(st.session_state.results), 2)

        st.subheader(f"Candidate: {name}")
        st.metric("Average Score", f"{average_score}%")

        for i, item in enumerate(st.session_state.results):
            st.write("---")
            st.write(f"### Question {i + 1}")
            st.write(item["question"])

            st.write("**Your Answer:**")
            st.write(item["answer"])

            st.write("**Score:**")
            st.write(f"{item['score']}%")

            st.write("**AI Feedback:**")
            st.write(item["feedback"])

        if st.button("Restart Interview"):
            st.session_state.current_question = 0
            st.session_state.results = []
            st.rerun()