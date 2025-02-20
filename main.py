import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


with open("data.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


class SimpleTextDataset(Dataset):
    def __init__(self, lines):
        self.data = lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = tokenizer(
            text, padding="max_length", max_length=32, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return input_ids, input_ids


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128):
        super(TinyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x).permute(1, 0, 2)
        attn_output, _ = self.attn(x, x, x)
        output = self.fc(attn_output).permute(1, 0, 2)
        return output


if __name__ == "__main__":
    dataset = SimpleTextDataset(lines)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    vocab_size = tokenizer.vocab_size
    model = TinyTransformer(vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inp, tgt in dataloader:
            optimizer.zero_grad()
            output = model(inp)
            loss = criterion(output.reshape(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")

        if avg_loss == 0:
            print("Loss reached 0, stopping training early.")
            break

    torch.save(model.state_dict(), "tiny_llm_plain_text_100.pth")
