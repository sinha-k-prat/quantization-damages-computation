"""The payoff: did the new 'sign' skill REUSE the existing 'content' (>V comparison) circuit?
(1) Which trits flipped to install sign (orig vs fine-tuned)? How many, where.
(2) Circuit overlap: are the sign-skill's load-bearing weights ENRICHED on the content-skill's
    load-bearing weights (vs chance)? And did the FLIPPED trits target the content circuit?
Enrichment > 1 => the model reused the comparison machinery to learn sign.
"""
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen

ckn = torch.load("runs_ternary/ckpt_neg.pt", map_location="cpu")
vocab, cfg, K = ckn["vocab"], ckn["cfg"], ckn["K"]; tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary(); m.load_state_dict(ckn["model"]); m.eval()
orig_trits = torch.load("runs_ternary/orig_trits.pt")

def cur_trits(L):
    s = L.weight.abs().mean(1, keepdim=True).clamp(min=1e-8)
    return torch.clamp(torch.round(L.weight / s), -1, 1)

# (1) flipped trits per layer
print("=== (1) trits flipped to install 'sign' (orig ternary -> fine-tuned) ===")
flipped = {}; tot_flip = tot = 0
for L in m.quant_layers():
    ot = orig_trits[L.tag]; nt = cur_trits(L).detach()
    f = (ot != nt)
    flipped[L.tag] = f; tot_flip += int(f.sum()); tot += f.numel()
top = sorted(flipped.items(), key=lambda kv: int(kv[1].sum()), reverse=True)[:6]
for tag, f in top:
    print(f"  {tag:<9} {int(f.sum()):>6} flips ({100*int(f.sum())/f.numel():.1f}% of layer)")
print(f"  TOTAL: {tot_flip}/{tot} trits flipped ({100*tot_flip/tot:.2f}%)")

# jacobian-based circuit: accumulate |grad| of a skill's CE per weight
def circuit(target_skill, gen, n=40):
    acc = {L.tag: torch.zeros_like(L.weight) for L in m.quant_layers()}
    got = 0
    for i in range(n*4):
        ex = gen(np.random.RandomState(6000+i))
        if not ex: continue
        toks, pl, sk = D._example_tokens(ex); ids = torch.tensor([tok.enc(toks)])
        losses = []
        m.zero_grad(); logits = m(ids)[0][0]
        for j in range(pl, len(toks)):
            if sk[j] == target_skill:
                losses.append(F.cross_entropy(logits[j-1:j], ids[0, j:j+1]))
        if not losses: continue
        torch.stack(losses).sum().backward()
        for L in m.quant_layers():
            if L.weight.grad is not None: acc[L.tag] += L.weight.grad.abs()
        got += 1
        if got >= n: break
    return acc

def content_ex(rng):
    for _ in range(50):
        els = D.make_list(rng, rng.randint(6, 9))
        r = D.SOLVERS["T5"](rng, els)                          # "first number greater than V" (content)
        if r: q, a, s = r; return dict(list=els, qtype="T5", level=3, query=q, answer=a, steps=s)
    return None

print("\ncomputing content-circuit and sign-circuit Jacobians ...", flush=True)
Cacc = circuit("content", content_ex)
Sacc = circuit("sign", lambda rng: N.make_neg_example(rng))

def topmask(acc, frac=0.02):
    return {t: (v >= torch.quantile(v.flatten(), 1-frac)) for t, v in acc.items()}
C = topmask(Cacc); S = topmask(Sacc)

def enrich(A, B):
    inter = sum(int((A[t] & B[t]).sum()) for t in A)
    a = sum(int(A[t].sum()) for t in A); n = sum(A[t].numel() for t in A)
    b = sum(int(B[t].sum()) for t in B)
    exp = a * b / n                                            # expected overlap by chance
    return inter, exp, inter/max(exp, 1e-9)

print("\n=== (2) circuit overlap (enrichment >1 = reuse) ===")
i1, e1, r1 = enrich(S, C)
print(f"sign-circuit ∩ content-circuit: {i1} weights (chance {e1:.0f})  ENRICHMENT {r1:.1f}x")
Fmask = flipped
i2, e2, r2 = enrich(Fmask, C)
print(f"flipped-trits ∩ content-circuit: {i2} (chance {e2:.0f})  ENRICHMENT {r2:.1f}x")
print("\ninterpretation:")
print(f"  sign reuses the comparison circuit: {'YES' if r1>1.5 else 'weak/no'} ({r1:.1f}x)")
print(f"  the skill was installed by editing that circuit: {'YES' if r2>1.5 else 'weak/no'} ({r2:.1f}x)")

# figure
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
tags = [L.tag for L in m.quant_layers()]
ax[0].bar(range(len(tags)), [100*int(flipped[t].sum())/flipped[t].numel() for t in tags], color="#6B5CA5")
ax[0].set_xticks(range(len(tags))); ax[0].set_xticklabels(tags, rotation=90, fontsize=6)
ax[0].set_ylabel("% trits flipped"); ax[0].set_title("(1) where sign was installed\n(trits flipped per layer)")
ax[1].bar(["sign∩content","flips∩content"], [r1, r2], color=["#3E8E5A","#6B5CA5"])
ax[1].axhline(1, ls="--", c="#000", label="chance (1x)"); ax[1].set_ylabel("enrichment vs chance")
ax[1].set_title("(2) did sign REUSE the comparison circuit?"); ax[1].legend()
plt.tight_layout(); plt.savefig("runs_ternary/exp_neg_overlap.png", dpi=120)
print("\nsaved runs_ternary/exp_neg_overlap.png")
