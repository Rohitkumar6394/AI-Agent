import uuid
import torch
import streamlit as st
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================
# Streamlit Page Config
# ==============================

st.set_page_config(
    page_title="LangGraph AI Chatbot ",
    page_icon="🤖",
    layout="centered"
)


# ==============================
# LangGraph State
# ==============================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==============================
# Load Local Chat Model
# ==============================

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()


# ==============================
# Generate Answer
# ==============================

def generate_answer(question, messages):
    chat_messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI teacher. "
                "Answer in simple Hinglish. "
                "Give clear explanation with example. "
                "Do not repeat words. "
                "Do not give useless answer."
            )
        }
    ]

    # last few messages memory
    for msg in messages[-6:-1]:
        if isinstance(msg, HumanMessage):
            chat_messages.append({
                "role": "user",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            chat_messages.append({
                "role": "assistant",
                "content": msg.content
            })

    chat_messages.append({
        "role": "user",
        "content": question
    })

    prompt = tokenizer.apply_chat_template(
        chat_messages,
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
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if not answer.strip():
        answer = "Sorry, answer generate nahi hua. Question thoda simple karke poochho."

    return answer.strip()


# ==============================
# LangGraph Node
# ==============================

def chatbot_node(state: ChatState):
    messages = state["messages"]
    user_question = messages[-1].content

    answer = generate_answer(user_question, messages)

    return {
        "messages": [AIMessage(content=answer)]
    }


# ==============================
# Build LangGraph
# ==============================

graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot_node)

graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

memory = MemorySaver()

chatbot_app = graph.compile(checkpointer=memory)


# ==============================
# Streamlit UI
# ==============================

st.title("🤖 LangGraph AI Chatbot  ")



# Session State
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ==============================
# Ask Question Input Upper
# ==============================

st.subheader("Ask Your Question")

with st.form("question_form", clear_on_submit=True):
    user_input = st.text_input(
        "Enter your question",
        placeholder="Example: Python mein function kya hota hai?"
    )

    submit_btn = st.form_submit_button("submit button")


if submit_btn and user_input.strip() != "":
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.spinner("AI thinking..."):
        result = chatbot_app.invoke(
            {
                "messages": [HumanMessage(content=user_input)]
            },
            config=config
        )

        ai_response = result["messages"][-1].content

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": ai_response
    })


st.divider()


# ==============================
# Display Chat History
# ==============================

st.subheader("Chat History")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])


# ==============================
# Clear Chat Button Bottom
# ==============================

st.divider()

if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()