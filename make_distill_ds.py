from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import json
from tqdm import tqdm

# params
TEACHER_MODEL = "meta-llama/Llama-2-7b-hf"
DATASET = "Skylion007/openwebtext"
SPLIT = "train"
SAMPLES = 10000
OUTPUT_DIR = "data/distill_corpus.jsonl"
RANDOM_SEED = 42
BATCH_SIZE = 16
SEQUENCE_LENGTH = 32

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# load datset, grab random samples
dataset = load_dataset(DATASET, split=SPLIT, trust_remote_code=True)
dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(SAMPLES))

# load teacher model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    output_hidden_states=False,
    output_attentions=False
).to(DEVICE)
model.eval()

# dump generations
with open(OUTPUT_DIR, "w") as f:
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Generating sample dataset..."):
        batch = dataset[i:i + BATCH_SIZE]
        prompts = batch["text"]
        
        # turn into inputs and move to GPU
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=SEQUENCE_LENGTH)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        for name, tensor in inputs.items():
            assert tensor.device == next(model.parameters()).device, (
            f"{name} is on {tensor.device}, "
            f"but model is on {next(model.parameters()).device}"
            )


        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        logits = logits.cpu().tolist()
        for prompt, logit in zip(prompts, logits):
            record = {
                "prompt": prompt,
                "logits": logit
            }
            f.write(json.dumps(record) + "\n")

print(f"Dataset dumped to {OUTPUT_DIR}")