import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

"""
Definition + training of PingHead module
"""

# params
DATAPATH = "data/pinghead_corpus.jsonl"
BATCH_SIZE = 64
EPOCHS = 20
LR = 5e-4
SEQUENCE_LENGTH = 32
H_DIM = 4096 # set to student model hidden size when i find it out
GATE_DIM = 128
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42
OUTPUT_DIR = "checkpoints/pinghead"

class PingHead(nn.Module):
    def __init__(self, h_dim, gate_dim=128):
        super().__init__()
        self.h_dim = h_dim
        self.gate_dim = gate_dim
        self.gru = nn.GRUCell(
            input_size=h_dim,
            hidden_size=gate_dim,
        )
        self.linear = nn.Linear(
            in_features=gate_dim,
            out_features=1
        )
    
    def forward(self, x):
        """
        x is a BATCH of shape (B, L, H)
        """
        B, L, H = x.shape
        s = torch.zeros(B, self.gate_dim).to(DEVICE)
        outputs = []
        for t in range(L):
            s_t = self.gru(x[:, t, :], s)
            outputs.append(self.linear(s_t).squeeze(-1))
        return torch.stack(outputs, dim=1)


# load dataset, create splits
ds = load_dataset(
    "json",
    data_files=DATAPATH,
    split="train",
)
splits = ds.train_test_split(test_size=0.1, seed=RANDOM_SEED)
train_ds, eval_ds = splits["train"], splits["test"]
train_ds.set_format(type="torch", columns=["h_t", "token_div"])
eval_ds.set_format(type="torch", columns=["h_t", "token_div"])

# create dataloaders
train_dataloader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
)
eval_dataloader = DataLoader(
    eval_ds,
    batch_size=BATCH_SIZE,
)

model = PingHead(
    h_dim=H_DIM,
    gate_dim=GATE_DIM,
).to(DEVICE)
opt = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-2,
)
mse = nn.MSELoss(reduction="none")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        h_t = batch["h_t"].to(DEVICE)
        token_div = batch["token_div"].to(DEVICE)

        # forward pass
        pred = model(h_t)
        loss = mse(pred, token_div).mean()

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss / len(train_dataloader)}")

    # eval model
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating {epoch}/{EPOCHS}"):
            h_t = batch["h_t"].to(DEVICE)
            token_div = batch["token_div"].to(DEVICE)
            pred = model(h_t)
            val_loss += mse(pred, token_div).mean().item()
    print(f"Epoch {epoch}/{EPOCHS} - Validation Loss: {val_loss / len(eval_dataloader)}")

torch.save(model.state_dict(), OUTPUT_DIR)