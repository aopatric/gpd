import os
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

# params
LOGIT_FILE = "data/distill_corpus.jsonl"
STUDENT_MODELNAME = "mtgv/MobileLLaMA-1.4B-Base"
TEACHER_MODELNAME = "meta-llama/Llama-2-7b-chat-hf"
CHECKPOINT_DIR = "checkpoints/lora"
BATCH_SIZE = 64
SEQUENCE_LENGTH = 32
EPOCHS = 20
LR = 2e-5
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LORA_RANK = 32
LORA_ALPHA = 4 * LORA_RANK
LORA_DROPOUT = 0.1

# load tokenizer and student model
tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODELNAME)
model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODELNAME,
    torch_dtype=torch.float16,
    device_map="auto",
).to(DEVICE)
model.resize_token_embeddings(len(tokenizer))
# sanity check
assert model.get_input_embeddings().weight.size(0) == len(tokenizer)


# add pad token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

# lora setup
lora_conf = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_conf)

# preprocessing helper
def preprocess(example):
    tok = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        padding="max_length",
    )
    teacher_logits = torch.tensor(
        example["logits"],
        dtype=torch.float16,
    )
    seq_len = len(tok["input_ids"])
    vocab_size = teacher_logits.numel() // seq_len
    teacher_logits = teacher_logits.view(seq_len, vocab_size)
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "teacher_logits": teacher_logits,
    }

# preprocess dataset, or load from cache
if not os.path.exists("data/distill_corpus_train"):
    # load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": LOGIT_FILE,
        },
        split="train",
    )

    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_raw = splits["train"]
    eval_raw = splits["test"]

    # preprocess datasets and cache
    train_data = train_raw.map(
        preprocess,
        remove_columns=["prompt", "logits"],
        num_proc=4,
        desc="Preprocessing train dataset",
    )
    eval_data = eval_raw.map(
        preprocess,
        remove_columns=["prompt", "logits"],
        num_proc=4,
        desc="Preprocessing eval dataset",
    )
    train_data.save_to_disk("data/distill_corpus_train")
    eval_data.save_to_disk("data/distill_corpus_eval")

else:
    train_data = load_from_disk("data/distill_corpus_train")
    eval_data = load_from_disk("data/distill_corpus_eval")

train_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "teacher_logits"],
)
eval_data.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "teacher_logits"],
)

# custom trainer for our loss function
class DistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_logits = inputs["teacher_logits"].to(DEVICE)

        outputs = model(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
        )
        student_logits = outputs.logits
        loss_fn = torch.nn.KLDivLoss(reduction="batchmean")
        student_log_probs = torch.log_softmax(student_logits, dim=-1)
        teacher_log_probs = torch.softmax(teacher_logits, dim=-1)
        loss = loss_fn(
            student_log_probs.view(-1, student_log_probs.size(-1)),
            teacher_log_probs.view(-1, teacher_log_probs.size(-1)),
        )
        return (loss, outputs) if return_outputs else loss

# training args
args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="linear",
    warmup_steps=500,
    logging_dir=os.path.join(CHECKPOINT_DIR, "logs"),
    logging_steps=50,
    save_steps=500,
    save_total_limit=5,
    fp16=True,
    label_names=["teacher_logits"],
    logging_strategy="steps",
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# custom collator
def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(x["input_ids"]) for x in batch]),
        "attention_mask": torch.stack([torch.tensor(x["attention_mask"]) for x in batch]),
        "teacher_logits": torch.stack([torch.tensor(x["teacher_logits"]) for x in batch]),
    }

trainer = DistillationTrainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    data_collator=collate_fn,
)

trainer.train()
trainer.save_model(CHECKPOINT_DIR)