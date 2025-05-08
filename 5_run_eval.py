import time
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pinghead import DualDecoder, PingHead
from tqdm import tqdm   

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_DIR = "data/eval_prompts.txt"
OUTPUT_DIR = "data/eval_results.jsonl"
TEACHER_NAME = "meta-llama/Llama-2-7b-chat-hf"
STUDENT_BASE_NAME = "mtgv/MobileLLaMA-1.4B-Base"
STUDENT_LORA_DIR = "checkpoints/lora/checkpoint-22500"
PINGHEAD_DIR = "checkpoints/pinghead/pinghead.pt"

def load_models(teacher_name, student_base_name, student_lora_dir, pinghead_dir):
    """
    Load the models from the specified directories.

    Returns:
        - tokenizer: The tokenizer for the teacher model.
        - teacher: The teacher model.
        - student_base: The base student model.
        - student: The student model with LoRA.
        - pinghead: The pinghead model.
    """


def main(
        prompt_dir,
        output_dir,
        teacher_name,
        student_base_name,
        student_lora_dir,
        pinghead_dir,
        delta=0.1,
        max_length=32,
        device=DEVICE
):
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=False)
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval().to(DEVICE)
    student_base = AutoModelForCausalLM.from_pretrained(
        student_base_name,
        torch_dtype=torch.float16,
        device_map="auto",
    ).to(DEVICE)
    student = PeftModel.from_pretrained(
        student_base,
        student_lora_dir,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval().to(DEVICE)
    pinghead = PingHead(h_dim=2048, mlp_dim=32).to(device)
    state = torch.load(pinghead_dir, map_location=DEVICE)
    pinghead.load_state_dict(state)
    pinghead = pinghead.to(device)
    pinghead.eval()
    dual_decoder = DualDecoder(
        tokenizer=tokenizer,
        student=student,
        pinghead=pinghead,
        teacher=teacher,
        delta=delta,
    ).to(device)
    dual_decoder.eval()
    print("Models loaded.")

if __name__ == "__main__":
    main(
        prompt_dir=PROMPT_DIR,
        output_dir=OUTPUT_DIR,
        teacher_name=TEACHER_NAME,
        student_base_name=STUDENT_BASE_NAME,
        student_lora_dir=STUDENT_LORA_DIR,
        pinghead_dir=PINGHEAD_DIR,
    )