"""Tangible demo of the Qwen result:
(1) show a REAL weight row: full-precision vs 4-bit quantized (the actual snapping)
(2) a worked GSM8K example: per-token CE fp vs 4-bit, computation vs copy tokens
(3) SAVE the quantized checkpoint to runs/qwen4bit/ so it persists
"""
import os, copy, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gsm8k_tag as G
from qwen_gptq import naive_quantize, is_target

MODEL = os.environ.get("QMODEL", "Qwen/Qwen2.5-0.5B-Instruct")
torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained(MODEL)
fp = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32); fp.eval()
q = copy.deepcopy(fp)

# capture ONE real weight row BEFORE quantizing (a middle MLP layer)
lin = q.model.layers[3].mlp.gate_proj
row_fp = lin.weight.data[0, :8].clone()
for n, m in q.named_modules():
    if is_target(n, m): naive_quantize(m, 4)
q.eval()
row_q = lin.weight.data[0, :8]
uniq = torch.unique(lin.weight.data[0]).numel()

print("=" * 72)
print("(1) A REAL weight row from Qwen  layers.3.mlp.gate_proj, row 0  (first 8 of 896)")
print("  full precision:", [round(float(x), 4) for x in row_fp])
print("  4-bit quantized:", [round(float(x), 4) for x in row_q])
print(f"  -> row of 896 weights now uses only {uniq} distinct values (4-bit = 16 levels)")

# (2) worked GSM8K example
print("\n" + "=" * 72)
print("(2) A worked GSM8K example: per-token loss, fp vs 4-bit")
ex = load_dataset("gsm8k", "main", split="test[7:8]")[0]
Q, A = ex["question"], ex["answer"]
print("Q:", Q[:110], "...")
print("gold:", G.gold_answer(A))
clean, toks = G.tag_tokens(Q, A, tok)
pids = tok(Q + "\n", add_special_tokens=False)["input_ids"]
sids = [t for _, _, t in toks]; labs = [l for _, l, _ in toks]; strs = [s for s, _, _ in toks]
ids = torch.tensor([pids + sids])
with torch.no_grad():
    lf = fp(ids).logits[0]; lq = q(ids).logits[0]
p0 = len(pids)
print(f"\n  {'token':<10}{'type':<12}{'fp CE':>8}{'4bit CE':>9}{'jump':>8}")
rows = []
for j in range(len(sids)):
    pos = p0 + j
    cf = float(F.cross_entropy(lf[pos-1:pos], ids[0, pos:pos+1]))
    cq = float(F.cross_entropy(lq[pos-1:pos], ids[0, pos:pos+1]))
    rows.append((strs[j].strip(), labs[j], cf, cq))
# show computation and copy tokens (the interesting ones)
for t, l, cf, cq in rows:
    if l in ("computation", "copy") and t:
        print(f"  {t[:9]:<10}{l:<12}{cf:>8.2f}{cq:>9.2f}{cq-cf:>+8.2f}")
comp = [(cf, cq) for _, l, cf, cq in rows if l == "computation"]
cop = [(cf, cq) for _, l, cf, cq in rows if l == "copy"]
if comp: print(f"\n  COMPUTATION tokens: mean fp {np.mean([c for c,_ in comp]):.2f} -> 4bit {np.mean([c for _,c in comp]):.2f}")
if cop: print(f"  COPY tokens:        mean fp {np.mean([c for c,_ in cop]):.2f} -> 4bit {np.mean([c for _,c in cop]):.2f}")

# (3) save the quantized checkpoint (fp16 to halve size)
print("\n" + "=" * 72)
os.makedirs("runs/qwen4bit", exist_ok=True)
q.half().save_pretrained("runs/qwen4bit"); tok.save_pretrained("runs/qwen4bit")
sz = sum(os.path.getsize(os.path.join("runs/qwen4bit", f)) for f in os.listdir("runs/qwen4bit")) / 1e6
print(f"(3) SAVED quantized model -> runs/qwen4bit/  ({sz:.0f} MB)")
print("DONE")
