import torch.nn as nn
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

class PingHead(nn.Module):
    def __init__(self, h_dim, mlp_dim=256, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(h_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, mlp_dim)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(mlp_dim, 1)
    
    def forward(self, x):
        B, L, H = x.shape
        flattened = x.view(B * L, H)
        x = self.fc1(flattened)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x.view(B, L)

class DualDecoder(nn.Module):
    def __init__(self, teacher, student, tokenizer, pinghead, delta=0.1):
        super().__init__()
        self.teacher = teacher.eval().to(DEVICE)
        self.student = student.eval().to(DEVICE)
        self.pinghead = pinghead.eval().to(DEVICE)

        self.tokenizer = tokenizer
        self.delta = delta
    
    @torch.no_grad()
    def generate(self, prompt, max_length=32):
        input_ids = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        generated = input_ids.clone()

        # start generation from teacher model
        out_t = self.teacher(input_ids=generated, return_dict=True)
        tok_t = out_t.logits[:, -1, :].softmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, tok_t], dim=1)

        # continue by proposing and accepting/rejecting student tokens
        for _ in range(max_length - 1):
            out_s = self.student(
                input_ids=generated,
                output_hidden_states=True,
                return_dict=True
            )
            logits_s = out_s.logits[:, -1, :]
            h_s = out_s.hidden_states[-1][:, -1, :]

            pred_div = self.pinghead(h_s).item()
            if pred_div > self.delta:
                # diverging too far, use a teacher token
                out = self.teacher(
                    input_ids=generated,
                    return_dict=True
                )
                next_tok = out.logits[:, -1, :].softmax(dim=-1, keepdim=True)
            else:
                # accept student token
                next_tok = logits_s.softmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)

            if next_tok.item() == self.tokenizer.eos_token_id:
                break
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]