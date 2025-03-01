import torch
from transformers import BertTokenizer
from main import TinyTransformer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
model = TinyTransformer(vocab_size)
model.load_state_dict(torch.load("tiny_llm_plain_text_500.pth", map_location=torch.device("cpu")))
model.eval()

def generate_response(input_text):
    # Note: Use the same max_length as during generation (e.g., 10)
    tokens = tokenizer(input_text, padding="max_length", max_length=10, return_tensors="pt")
    input_ids = tokens["input_ids"]

    with torch.no_grad():
        output = model(input_ids)

    predicted_ids = output.argmax(dim=2)
    response_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return response_text

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(user_input)
    print("Bot:", response)
