from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

MODEL_NAME = "google/flan-t5-small"

print("Loading AI model... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("AI model loaded successfully.")


def ai_agent(question):
    if not question.strip():
        return "Please ask a question."

    prompt = f"""
    Answer the question in simple Hinglish language.

    Question: {question}

    Answer:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    if answer.strip() == "":
        return "Sorry, main iska answer clear nahi de pa raha hoon."

    return answer


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Docker</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            padding: 40px;
        }

        .container {
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 14px;
            box-shadow: 0px 5px 20px rgba(0,0,0,0.12);
        }

        h1 {
            text-align: center;
            color: #111827;
        }

        p {
            color: #555;
            text-align: center;
        }

        textarea {
            width: 100%;
            height: 100px;
            padding: 12px;
            font-size: 16px;
            border-radius: 8px;
            border: 1px solid #ccc;
            resize: none;
        }

        button {
            margin-top: 15px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
            background: #111827;
            color: white;
            border: none;
            border-radius: 8px;
            width: 100%;
        }

        .answer {
            margin-top: 25px;
            background: #eef2ff;
            padding: 18px;
            border-radius: 10px;
            font-size: 17px;
            line-height: 1.6;
        }

        .note {
            margin-top: 20px;
            background: #fff7ed;
            padding: 12px;
            border-radius: 8px;
            color: #7c2d12;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Agent Using Docker</h1>
        <p>Ask any question. This AI Agent runs inside Docker.</p>

        <form method="POST">
            <textarea name="question" placeholder="Example: Explain cloud computing in simple words..." required>{{ question or "" }}</textarea>
            <button type="submit">Ask AI Agent</button>
        </form>

        {% if answer %}
        <div class="answer">
            <strong>AI Agent:</strong><br>
            {{ answer }}
        </div>
        {% endif %}

        <div class="note">
            <b>Note:</b> Ye free offline Hugging Face model use karta hai. Large ChatGPT jaisa powerful nahi hoga, lekin random questions ka basic answer de sakta hai.
        </div>
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "")
        answer = ai_agent(question)

    return render_template_string(
        HTML_PAGE,
        answer=answer,
        question=question
    )


@app.route("/api/ask", methods=["POST"])
def ask_api():
    data = request.get_json()
    question = data.get("question", "")

    answer = ai_agent(question)

    return jsonify({
        "question": question,
        "answer": answer
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)