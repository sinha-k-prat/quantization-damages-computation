"""Find a GSM8K problem the 4-bit quantized Qwen solves CORRECTLY, and show its worked solution.
Uses the saved checkpoint runs/qwen4bit/.
"""
import re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gsm8k_tag as G

tok = AutoTokenizer.from_pretrained("runs/qwen4bit")
q = AutoModelForCausalLM.from_pretrained("runs/qwen4bit", torch_dtype=torch.float32); q.eval()
ds = load_dataset("gsm8k", "main", split="test[:24]")

def extract(t):
    m = re.search(r'\\boxed\{\s*(-?\$?\d[\d,]*\.?\d*)', t) or re.search(r'####\s*(-?\$?\d[\d,]*\.?\d*)', t)
    nums = re.findall(r'-?\$?\d[\d,]*\.?\d*', t)
    p = m.group(1) if m else (nums[-1] if nums else None)
    return p.replace(",", "").replace("$", "").strip() if p else None

@torch.no_grad()
def solve(question):
    ids = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True, return_tensors="pt")
    g = q.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=320, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

found = 0
for i, d in enumerate(ds):
    gold = G.gold_answer(d["answer"]); out = solve(d["question"]); pred = extract(out)
    ok = (pred == gold)
    print(f"[{i:2d}] gold={gold:<6} 4bit_pred={str(pred):<6} {'CORRECT' if ok else 'wrong'}", flush=True)
    if ok and found < 2:
        found += 1
        print("\n" + "=" * 72)
        print("4-BIT QWEN SOLVED THIS CORRECTLY:")
        print("Q:", d["question"])
        print("\n4-bit model's worked solution:\n", out.strip()[:900])
        print(f"\n-> extracted answer {pred} == gold {gold}  ✓")
        print("=" * 72 + "\n")
    if found >= 2 and i >= 12:
        break
print("DONE", flush=True)
