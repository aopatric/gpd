import json
import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from peft import PeftModel

# params
IN_FILE = "data/distill_corpus.jsonl"
OUT_FILE = "data/pinghead_corpus.jsonl"
STUDENT_MODEL = "checkpoints/lora/checkpoint-22000"
MODELNAME = "mtgv/MobileLLaMA-1.4B-Base"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
SEQUENCE_LENGTH = 32

tokenizer = AutoTokenizer.from_pretrained(MODELNAME)
base_model = AutoModelForCausalLM.from_pretrained(
    MODELNAME,
    torch_dtype=torch.float16,
    device_map="auto",
).to(DEVICE)

model = PeftModel.from_pretrained(
    base_model,
    STUDENT_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
).to(DEVICE)
model.eval()

# add pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

"""
Jensen-Shannon divergence for symmetry and boundedness
"""
def safe_div_torch(s_prob, t_prob, eps=1e-6):
    s_prob += eps
    t_prob += eps
    s_prob = s_prob / s_prob.sum(dim=-1, keepdim=True)
    t_prob = t_prob / t_prob.sum(dim=-1, keepdim=True)
    M = 0.5 * (s_prob + t_prob)
    kl_s = F.kl_div(s_prob.log(), M, reduction="none").sum(dim=-1)
    kl_t = F.kl_div(t_prob.log(), M, reduction="none").sum(dim=-1)
    return 0.5 * (kl_s + kl_t).clamp(min=0, max=1)

# process data
ds = load_dataset(
    "json",
    data_files={
        "train": IN_FILE,
    },
    split="train",
)

with open(OUT_FILE, "w") as f:
    for batch in tqdm(ds.iter(batch_size=BATCH_SIZE), desc="Processing dataset..."):
        teacher_logits = torch.tensor(
            batch["logits"],
            dtype=torch.float16,
        ).to(DEVICE)
        
        inputs = tokenizer(
            batch["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=SEQUENCE_LENGTH,
        ).to(DEVICE)

        with torch.no_grad():
            out = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        hiddens = out.hidden_states[-1]
        logits = out.logits
        
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        student_probs = F.softmax(logits.float(), dim=-1)
        div = safe_div_torch(student_probs, teacher_probs)

        for i in range(len(batch["prompt"])):
            record = {
                "h_t": hiddens[i].cpu().numpy().tolist(),
                "token_div": div[i].cpu().numpy().tolist(),
            }
            f.write(json.dumps(record) + "\n")