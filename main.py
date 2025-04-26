import time
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


torch.backends.cudnn.benchmark = True


def load_dialogue_pairs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for entry in data:
        if "user" in entry and "ai" in entry:
            pairs.append(entry)
    return pairs


data_file = os.path.join("data", "data.json")
dialogue_pairs = load_dialogue_pairs(data_file)
print(f"Total dialogue pairs: {len(dialogue_pairs)}")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["[USER]", "[ASSISTANT]", "[SEP]"]}
)


class DialogueDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=32):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        user_text = pair["user"].strip()
        ai_text = pair["ai"].strip()

        text = f"[USER] {user_text} [SEP] [ASSISTANT] {ai_text} [SEP]"
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens_tensor = tokens["input_ids"].squeeze(0)
        labels = tokens_tensor.clone()
        return tokens_tensor, labels


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, seq_length=32, nhead=4):
        super(TinyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        self.attn = nn.MultiheadAttention(d_model, num_heads=nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        logits = self.fc(x)
        return logits


def main():
    dataset = DialogueDataset(dialogue_pairs, tokenizer, max_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

    vocab_size = len(tokenizer)
    model = TinyTransformer(vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_amp else None

    num_epochs = 10000
    loss_threshold = 1e-7
    total_train_start = time.time()
    total_epoch_time = 0.0
    total_tokens_processed = 0

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_times = []
        model.train()

        for inp, tgt in dataloader:
            batch_start = time.time()
            optimizer.zero_grad()
            inp, tgt = inp.to(device, non_blocking=True), tgt.to(
                device, non_blocking=True
            )
            total_tokens_processed += inp.numel()

            if use_amp:
                with torch.amp.autocast():
                    output = model(inp)
                    loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(inp)
                loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
                loss.backward()
                optimizer.step()

            batch_times.append(time.time() - batch_start)
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        total_epoch_time += epoch_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_loss = epoch_loss / len(dataloader)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Epoch Time: {epoch_time:.2f} sec, Avg Batch Time: {avg_batch_time:.2f} sec"
        )

        if avg_loss < loss_threshold:
            print(
                f"Avg loss {avg_loss:.6f} is below threshold {loss_threshold}, stopping early."
            )
            break

    total_training_time = time.time() - total_train_start
    avg_epoch_time = total_epoch_time / (epoch + 1)

    print(f"Total training time: {total_training_time:.2f} sec")
    print(f"Average epoch time: {avg_epoch_time:.2f} sec")
    print(f"Total tokens processed: {total_tokens_processed}")

    if not os.path.exists("mini_llm_model"):
        os.makedirs("mini_llm_model")
    torch.save(model.state_dict(), "mini_llm_model/pytorch_model.bin")
    tokenizer.save_pretrained("mini_llm_model")


if __name__ == "__main__":
    main()
