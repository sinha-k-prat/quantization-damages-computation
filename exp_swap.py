"""(A) Does the model break when we swap instruction<->list (task-first)?  greedy exact-match.
(B) Is L0:SA's first scan 'evenly spread'?  inspect post-softmax attention at the working position.
"""
import json, math, numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen, apply_rope

ck = torch.load("runs/ckpt.pt", map_location="cpu"); vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
model = MiniQwen(cfg, len(vocab), K); model.load_state_dict(ck["control"]); model.eval()
data = [d for d in json.load(open("runs/probe_data.json")) if d["vector"] == "fetch"][:160]

def p_last(d): return [str(e) for e in d["list"]] + ["|", "Q"] + list(map(str, d["query"])) + ["="]
def p_first(d): return ["Q"] + list(map(str, d["query"])) + ["|"] + [str(e) for e in d["list"]] + ["="]

@torch.no_grad()
def greedy_answer(prompt_toks, maxn=26):
    ids = tok.enc(prompt_toks)
    for _ in range(maxn):
        nxt = int(model(torch.tensor([ids]))[0][0, -1].argmax())
        ids.append(nxt)
        if nxt == tok.eos: break
    gen = [tok.itos[i] for i in ids[len(prompt_toks):]]
    if "A" in gen:
        j = gen.index("A")
        if j + 1 < len(gen): return gen[j + 1]
    return None

def em(builder):
    ok = 0
    for d in data:
        a = greedy_answer(builder(d))
        ok += (a is not None and a == str(d["answer"]))
    return ok / len(data)

print("=== (A) does it break? greedy exact-match ===")
acc_last = em(p_last); print(f"task-last (trained format):  {acc_last:.2f}")
acc_first = em(p_first); print(f"task-first (swapped):        {acc_first:.2f}")
print("VERDICT:", "BREAKS (task-first fails) -> structurally task-last" if acc_first < 0.4 * max(acc_last, 1e-9)
      else "does NOT break -> inspect L0:SA spread below")

# ---- (B) L0:SA attention distribution at the working position ----
@torch.no_grad()
def l0_pattern(prompt_toks):
    idx = torch.tensor([tok.enc(prompt_toks)]); T = idx.shape[1]
    x = model.embed(idx); cos, sin = model.cos[:T], model.sin[:T]
    b = model.blocks[0]; h = b.n1(x); nh, nkv, hd = b.attn.nh, b.attn.nkv, b.attn.hd
    q, _ = b.attn.q(h); k, _ = b.attn.k(h)
    q = q.view(1, T, nh, hd).transpose(1, 2); k = k.view(1, T, nkv, hd).transpose(1, 2)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    k = k.repeat_interleave(nh // nkv, 1)
    sc = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
    mask = torch.triu(torch.ones(T, T), 1).bool()
    sc = sc.masked_fill(mask, -1e9)
    att = F.softmax(sc, -1)[0, :, -1, :]                 # [heads, T] from working pos
    return att.mean(0).numpy(), T                        # head-avg attention over keys

# aggregate: entropy vs uniform, and how much mass on the query tokens (recent) vs list
ents, unif_ents, recent_mass = [], [], []
sample_pat = None
for d in data[:120]:
    pt = p_last(d); att, T = l0_pattern(pt)
    ents.append(-(att * np.log(att + 1e-12)).sum())
    unif_ents.append(math.log(T))
    # query tokens are the last ~7 before '=': positions [T-8 : T-1]
    recent_mass.append(att[max(0, T - 8):T - 1].sum())
    if sample_pat is None: sample_pat = (pt, att)
Emean, Umean = np.mean(ents), np.mean(unif_ents)
print("\n=== (B) L0:SA first-scan spread (task-last, working position) ===")
print(f"attention entropy: {Emean:.2f} nats   (uniform would be {Umean:.2f})   ratio {Emean/Umean:.2f}")
print(f"mass on the recent QUERY tokens: {np.mean(recent_mass):.2f}   (of total 1.0)")
print("interpretation:", "evenly spread (near-uniform scan)" if Emean/Umean > 0.9
      else f"NOT even — concentrated ({np.mean(recent_mass):.0%} on the adjacent query tokens)")

pt, att = sample_pat
plt.figure(figsize=(12, 4))
plt.bar(range(len(att)), att, color="#3E7CB1")
plt.xticks(range(len(pt)), pt, rotation=45, ha="right", fontsize=8)
plt.axhline(1/len(att), ls="--", c="#d95f2b", label="uniform")
plt.ylabel("L0:SA attention from '='"); plt.title("L0:SA first-scan: where the working position looks (one example)")
plt.legend(); plt.tight_layout(); plt.savefig("runs/exp_swap.png", dpi=120)
print("saved runs/exp_swap.png")
