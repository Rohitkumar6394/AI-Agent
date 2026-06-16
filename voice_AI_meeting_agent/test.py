from gpt4all import GPT4All

model = GPT4All(
    model_name="mistral-7b-instruct.Q4_0.gguf",
    model_path="models",
    allow_download=False
)

print(model.generate("Hello"))