"""Where do the minimum-symbolic-edit flips live? On the lam=10 run (~0.05% flips, 100%/100% acc):
(1) flips per layer/component; (2) enrichment of the flips on the CONTENT (comparison) circuit and
on the SIGN circuit (Jacobian top-2%); (3) are flips concentrated vs the unconstrained lam=0 run?
"""
import os, numpy as np, torch, torch.nn.functional as F
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen

torch.set_num_threads(2)
LAM = os.environ.get("LAM_DIR", "lam_10")
base = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
ckA = torch.load(f"runs_mse/{LAM}/ckpt.pt", map_location="cpu")
cfg, K = ckA["cfg"], ckA["K"]; vocab = ckA["vocab"]; tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary(); m.load_state_dict(ckA["model"]); m.eval()

def trit(W):
    s = W.abs().mean(1, keepdim=True).clamp(min=1e-8); return torch.clamp(torch.round(W / s), -1, 1)
orig = {t: trit(w) for t, w in ((L.tag, base["target"][f"blocks.{L.tag[1]}.{'attn' if L.tag.split('.')[1] in ('q','k','v','o') else 'mlp'}.{L.tag.split('.')[1]}.weight"]) for L in m.quant_layers())} if False else None
# simpler: rebuild orig trits by loading the base target into a scratch model
m0 = MiniQwen(cfg, len(base["vocab"]), K); m0.enable_ternary(); m0.load_state_dict(base["target"])
orig = {L0.tag: trit(L0.weight.data) for L0 in m0.quant_layers()}
del m0

flips = {}
tot_f = tot = 0
for L in m.quant_layers():
    f = (trit(L.weight.data) != orig[L.tag]); flips[L.tag] = f
    tot_f += int(f.sum()); tot += f.numel()
print(f"=== ({LAM}) flip localization: {tot_f} flips total ({100*tot_f/tot:.3f}%) ===")
top = sorted(flips.items(), key=lambda kv: int(kv[1].sum()), reverse=True)[:8]
for tag, f in top:
    print(f"  {tag:<9} {int(f.sum()):>6} flips  ({100*int(f.sum())/f.numel():.3f}% of layer)")
byclass = {}
for tag, f in flips.items():
    c = tag.split(".")[1]; byclass[c] = byclass.get(c, 0) + int(f.sum())
print("  by class:", {k: v for k, v in sorted(byclass.items(), key=lambda kv: -kv[1])})

def circuit(target_skill, gen, n=30):
    acc = {L.tag: torch.zeros_like(L.weight) for L in m.quant_layers()}
    got = 0
    for i in range(n * 5):
        ex = gen(np.random.RandomState(6000 + i))
        if not ex: continue
        toks, pl, sk = D._example_tokens(ex); ids = torch.tensor([tok.enc(toks)])
        m.zero_grad(); logits = m(ids)[0][0]
        losses = [F.cross_entropy(logits[j-1:j], ids[0, j:j+1]) for j in range(pl, len(toks)) if sk[j] == target_skill]
        if not losses: continue
        torch.stack(losses).sum().backward()
        for L in m.quant_layers():
            if L.weight.grad is not None: acc[L.tag] += L.weight.grad.abs()
        got += 1
        if got >= n: break
    return acc

def content_ex(rng):
    for _ in range(50):
        els = D.make_list(rng, rng.randint(6, 9)); r = D.SOLVERS["T5"](rng, els)
        if r: q, a, s = r; return dict(list=els, qtype="T5", level=3, query=q, answer=a, steps=s)
    return None

print("\ncomputing content & sign circuits (Jacobian top-2%) ...", flush=True)
C = circuit("content", content_ex); S = circuit("sign", lambda r: N.make_neg_example(r))
def topmask(acc, frac=0.02):
    return {t: (v >= torch.quantile(v.flatten(), 1 - frac)) for t, v in acc.items()}
Cm, Sm = topmask(C), topmask(S)

def enrich(A, B):
    inter = sum(int((A[t] & B[t]).sum()) for t in A)
    a = sum(int(A[t].sum()) for t in A); b = sum(int(B[t].sum()) for t in B); n = sum(A[t].numel() for t in A)
    return inter, a * b / n
i1, e1 = enrich(flips, Cm); i2, e2 = enrich(flips, Sm); i3, e3 = enrich(Sm, Cm)
print(f"\n=== enrichment (observed / chance) ===")
print(f"flips ∩ CONTENT circuit : {i1:>6} / {e1:.0f}   = {i1/max(e1,1e-9):.1f}x")
print(f"flips ∩ SIGN circuit    : {i2:>6} / {e2:.0f}   = {i2/max(e2,1e-9):.1f}x")
print(f"SIGN ∩ CONTENT circuits : {i3:>6} / {e3:.0f}   = {i3/max(e3,1e-9):.1f}x")
