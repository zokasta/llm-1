import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from main import TinyTransformer
import torch.nn as nn


def resize_positional_embedding(model, new_length):
    """
    Resize the model's positional embedding from its original size to new_length.
    If new_length is greater than the current size, append extra embeddings.
    """
    with torch.no_grad():
        old_length = model.pos_embedding.shape[1]
        if new_length > old_length:
            extra_length = new_length - old_length

            extra_embedding = (
                torch.randn(
                    1,
                    extra_length,
                    model.pos_embedding.shape[-1],
                    device=model.pos_embedding.device,
                )
                * 0.02
            )
            new_embedding = torch.cat(
                [model.pos_embedding.data, extra_embedding], dim=1
            )
            model.pos_embedding.data = new_embedding


def generate_response(input_text, temperature=1.0, generation_max_length=64):

    tokenizer = BertTokenizer.from_pretrained("mini_llm_model")
    vocab_size = len(tokenizer)

    orig_seq_length = 32
    model = TinyTransformer(
        vocab_size, d_model=128, seq_length=orig_seq_length, nhead=4
    )

    state_dict = torch.load(
        "mini_llm_model/pytorch_model.bin", map_location=torch.device("cpu")
    )
    model.load_state_dict(state_dict)

    resize_positional_embedding(model, generation_max_length)

    model.eval()

    formatted_input = f"[USER] {input_text} [SEP] [ASSISTANT]"
    tokens = tokenizer(
        formatted_input,
        padding="max_length",
        truncation=True,
        max_length=generation_max_length,
        return_tensors="pt",
    )
    generated = tokens["input_ids"]

    with torch.no_grad():
        while generated.shape[1] < generation_max_length:
            output = model(generated)
            logits = output[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            if next_token.item() == tokenizer.convert_tokens_to_ids("[SEP]"):
                break

    full_text = tokenizer.decode(generated[0], skip_special_tokens=True)

    response_text = full_text.split("[ASSISTANT]")[-1].strip()
    return response_text


if __name__ == "__main__":
    print("Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(
            user_input, temperature=0.8, generation_max_length=64
        )
        print("Bot:", response)
