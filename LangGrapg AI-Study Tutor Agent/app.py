import uuid
import torch
import streamlit as st
from typing import TypedDict

from transformers import AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, START, END


st.set_page_config(
    page_title="LangGraph AI Study Tutor Agent",
    page_icon="🎓",
    layout="centered"
)


class TutorState(TypedDict):
    question: str
    intent: str
    answer: str


@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


def clean_answer(text):
    text = text.strip()

    lines = text.split("\n")
    clean_lines = []
    seen = set()

    for line in lines:
        line = line.strip()

        if line and line not in seen:
            clean_lines.append(line)
            seen.add(line)

    final_text = "\n".join(clean_lines)

    if len(final_text) < 5:
        final_text = "Sorry, answer clear generate nahi hua. Question thoda simple karke poochho."

    return final_text


def generate_ai_answer(system_prompt, user_question):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.25,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return clean_answer(answer)


def classify_question(state: TutorState):
    question = state["question"].lower()

    quiz_words = [
        "quiz", "mcq", "test", "question banao", "questions banao"
    ]

    summary_words = [
        "summary", "summarize", "notes", "short notes", "revision"
    ]

    code_words = [
        "code", "program", "error", "bug", "python", "java",
        "javascript", "react", "function", "loop", "variable", "class"
    ]

    if any(word in question for word in quiz_words):
        intent = "quiz"

    elif any(word in question for word in summary_words):
        intent = "summary"

    elif any(word in question for word in code_words):
        intent = "code"

    else:
        intent = "explain"

    return {
        "question": state["question"],
        "intent": intent,
        "answer": ""
    }


def route_question(state: TutorState):
    return state["intent"]


def explain_node(state: TutorState):
    system_prompt = """
You are an AI Study Tutor.
Answer in simple Hinglish.
Explain the topic clearly.

Use this format:

1. Simple Definition
2. Easy Explanation
3. Real Life Example
4. Short Conclusion

Do not repeat words.
Do not give useless answer.
"""

    answer = generate_ai_answer(system_prompt, state["question"])

    return {
        "question": state["question"],
        "intent": state["intent"],
        "answer": answer
    }


def code_node(state: TutorState):
    system_prompt = """
You are an AI Coding Tutor.
Answer in simple Hinglish.
If user asks a coding concept, explain with example.
If user asks for code, give clean working code.

Use this format:

1. Meaning
2. Explanation
3. Code Example
4. Output
5. Conclusion

Do not repeat words.
Do not generate useless text.
"""

    answer = generate_ai_answer(system_prompt, state["question"])

    return {
        "question": state["question"],
        "intent": state["intent"],
        "answer": answer
    }


def quiz_node(state: TutorState):
    system_prompt = """
You are an AI Quiz Generator.
Create 5 simple MCQ questions.
Use simple Hinglish.
Each MCQ must have 4 options.
Also give correct answer.

Format:
Q1.
A)
B)
C)
D)
Correct Answer:
"""

    answer = generate_ai_answer(system_prompt, state["question"])

    return {
        "question": state["question"],
        "intent": state["intent"],
        "answer": answer
    }


def summary_node(state: TutorState):
    system_prompt = """
You are an AI Notes Maker.
Create short and clear notes.
Use simple Hinglish.
Use bullet points.
Keep notes useful for students.
Do not repeat words.
"""

    answer = generate_ai_answer(system_prompt, state["question"])

    return {
        "question": state["question"],
        "intent": state["intent"],
        "answer": answer
    }


graph = StateGraph(TutorState)

graph.add_node("classify_question", classify_question)
graph.add_node("explain_node", explain_node)
graph.add_node("code_node", code_node)
graph.add_node("quiz_node", quiz_node)
graph.add_node("summary_node", summary_node)

graph.add_edge(START, "classify_question")

graph.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "explain": "explain_node",
        "code": "code_node",
        "quiz": "quiz_node",
        "summary": "summary_node"
    }
)

graph.add_edge("explain_node", END)
graph.add_edge("code_node", END)
graph.add_edge("quiz_node", END)
graph.add_edge("summary_node", END)

study_agent = graph.compile()


st.markdown(
    """
    <h1 style="
        white-space: nowrap;
        font-size: 45px;
        text-align: center;
        margin-bottom: 20px;
    ">
    🎓 LangGraph AI Study Tutor Agent
    </h1>
    """,
    unsafe_allow_html=True
)
 

if "history" not in st.session_state:
    st.session_state.history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


st.subheader("Ask Your Study Question")

with st.form("question_form", clear_on_submit=True):
    user_question = st.text_input(
        "Enter your question",
        placeholder="Example: Python mein function kya hota hai?"
    )

    submit = st.form_submit_button("submit button")


if submit and user_question.strip() != "":
    with st.spinner("LangGraph Agent thinking..."):
        result = study_agent.invoke({
            "question": user_question,
            "intent": "",
            "answer": ""
        })

    st.session_state.history.append({
        "question": user_question,
        "intent": result["intent"],
        "answer": result["answer"]
    })


st.divider()

st.subheader("Agent Response")

if len(st.session_state.history) == 0:
    st.info("Abhi koi question nahi poocha gaya.")

for item in reversed(st.session_state.history):
    st.markdown("### You Asked")
    st.write(item["question"])

    st.markdown(f"**Detected Intent:** `{item['intent']}`")

    st.markdown("### AI Answer")
    st.write(item["answer"])

    st.divider()


if st.button("Clear Chat"):
    st.session_state.history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()