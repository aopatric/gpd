import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pinghead import PingHead
import os
import numpy as np

"""
Definition + training of PingHead module
"""

# params
DATAPATH = "data/pinghead_corpus.jsonl"
BATCH_SIZE = 64
EPOCHS = 100
LR = 5e-5
SEQUENCE_LENGTH = 32
H_DIM = 2048
MLP_DIM = 32
DROPOUT = 0.2
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

os.makedirs("checkpoints/pinghead", exist_ok=True)
OUTPUT_DIR = "checkpoints/pinghead/pinghead.pt"

# load dataset, create splits
if not os.path.exists("data/pinghead_corpus_train"):
    ds = load_dataset(
        "json",
        data_files=DATAPATH,
        split="train",
    )
    splits = ds.train_test_split(test_size=0.1, seed=RANDOM_SEED)
    train_ds, eval_ds = splits["train"], splits["test"]
    train_ds.save_to_disk("data/pinghead_corpus_train")
    eval_ds.save_to_disk("data/pinghead_corpus_eval")
else:
    train_ds = load_from_disk("data/pinghead_corpus_train")
    eval_ds = load_from_disk("data/pinghead_corpus_eval")

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
    mlp_dim=MLP_DIM,
    dropout=DROPOUT,
).to(DEVICE)

decay, no_decay = [], []
for name, param in model.named_parameters():
    if "bias" in name or param.ndim < 2 in name:
        no_decay.append(param)
    else:
        decay.append(param)

opt = torch.optim.AdamW([
    {'params': decay,
    'weight_decay': 1e-1,},
    {'params': no_decay,
    'weight_decay': 0.0,},
], lr=LR)
mse = nn.MSELoss(reduction="none")

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, 
    mode="min", 
    factor=0.5,  # Reduce LR by a factor of 0.5
    patience=3,  # Number of epochs with no improvement before reducing LR
)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch}/{EPOCHS}"):
        h_t = batch["h_t"].to(DEVICE)
        token_div = batch["token_div"].to(DEVICE)

        # forward pass
        pred = model(h_t)
        mask = (token_div != 0).float()  # or an explicit attention_mask
        loss = (mse(pred, token_div) * mask).sum() / mask.sum()

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss / len(train_dataloader)}")

    # Eval model
    model.eval()
    eval_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in eval_dataloader:
            h_t   = batch["h_t"].to(DEVICE)          # (B, L, H)
            target= batch["token_div"].to(DEVICE)    # (B, L)
            pred  = model(h_t)                       # (B, L)

            # Compute evaluation loss
            mask = (target != 0).float()
            batch_loss = (mse(pred, target) * mask).sum() / mask.sum()
            eval_loss += batch_loss.item()

            all_preds.append(pred.cpu().numpy().ravel())
            all_targets.append(target.cpu().numpy().ravel())

    eval_loss /= len(eval_dataloader)
    print(f"Epoch {epoch}/{EPOCHS} - Eval Loss: {eval_loss}")

    # Step the scheduler
    scheduler.step(eval_loss)

    # Flatten predictions and targets for metrics
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Compute metrics
    mean_t   = all_targets.mean()
    std_t    = all_targets.std()
    rmse     = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae      = mean_absolute_error(all_targets, all_preds)
    r2       = r2_score(all_targets, all_preds)
    baseline_rmse = np.sqrt(((all_targets - mean_t) ** 2).mean())

    # Log summary
    print(f"\n— Epoch {epoch} Eval Metrics —")
    print(f"  Target JSD    mean: {mean_t:.4f},  std: {std_t:.4f}")
    print(f"  RMSE: {rmse:.4f},  MAE: {mae:.4f},  R²: {r2:.4f}")
    print(f"  Baseline RMSE (mean predictor): {baseline_rmse:.4f}\n")

torch.save(model.state_dict(), OUTPUT_DIR)