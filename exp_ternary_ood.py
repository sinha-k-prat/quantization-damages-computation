"""Ternary OOD test: the trained 1.58-bit model is perfect IN-distribution — does it break on
COMPUTATION when it must EXTRAPOLATE (longer lists)? Per-skill gap (ternary - control), in-dist vs OOD.
"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen
from retrieval_train import batchify, split_loss

ck = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]; tok = D.Tokenizer(vocab)
control = MiniQwen(cfg, len(vocab), K); control.load_state_dict(ck["control"]); control.eval()
tern = MiniQwen(cfg, len(vocab), K); tern.enable_ternary(); tern.load_state_dict(ck["target"]); tern.eval()
SKILLS = ["read", "semantic", "filter", "index", "content", "relative"]

def evalset(kind):
    if kind == "indist":
        ex = [D.make_example(np.random.RandomState(7000 + i), (i % 5) + 1) for i in range(500)]
    else:
        ex = [D.make_ood_example(np.random.RandomState(9000 + i), 13, 20) for i in range(500)]
    return batchify([e for e in ex if e], tok, cfg["block"])

@torch.no_grad()
def per_skill(model, X, S, lv):
    _, ps, _ = split_loss(model(X)[0], X, S, lv); return ps

res = {}
for kind in ["indist", "ood"]:
    X, S, lv = evalset(kind)
    pc, pt = per_skill(control, X, S, lv), per_skill(tern, X, S, lv)
    res[kind] = {s: (pc[s], pt[s]) for s in SKILLS if pc.get(s) is not None and pt.get(s) is not None}

print("=== TERNARY (1.58-bit) per-skill: control vs ternary, in-dist vs OOD ===")
print(f"{'skill':<10}{'type':<9}{'IN gap':>9}{'IN ratio':>10}{'OOD gap':>10}{'OOD ratio':>11}")
for s in SKILLS:
    ci, ti = res["indist"].get(s, (None, None)); co, to = res["ood"].get(s, (None, None))
    if ci is None or co is None: continue
    typ = "lookup" if s in ("read", "semantic", "filter") else "COMPUTE"
    print(f"{s:<10}{typ:<9}{ti-ci:>+9.4f}{ti/max(ci,1e-9):>9.2f}x{to-co:>+10.4f}{to/max(co,1e-9):>10.2f}x")
lk = [s for s in SKILLS if s in ("read", "semantic", "filter")]
cp = [s for s in SKILLS if s in ("index", "content", "relative")]
def mgap(kind, group): return np.mean([res[kind][s][1] - res[kind][s][0] for s in group if s in res[kind]])
print(f"\nmean lookup gap:  in-dist {mgap('indist',lk):+.4f}   OOD {mgap('ood',lk):+.4f}")
print(f"mean COMPUTE gap: in-dist {mgap('indist',cp):+.4f}   OOD {mgap('ood',cp):+.4f}")

# figure
fig, ax = plt.subplots(figsize=(10, 5.5)); x = np.arange(len(SKILLS)); w = 0.38
gin = [res["indist"][s][1] - res["indist"][s][0] if s in res["indist"] else 0 for s in SKILLS]
goo = [res["ood"][s][1] - res["ood"][s][0] if s in res["ood"] else 0 for s in SKILLS]
ax.bar(x - w/2, gin, w, label="in-distribution", color="#888")
ax.bar(x + w/2, goo, w, label="OOD (longer lists)", color="#d62728")
ax.axhline(0, color="#000", lw=.8); ax.set_xticks(x); ax.set_xticklabels(SKILLS)
for i, s in enumerate(SKILLS):
    ax.text(i, -0.002, "compute" if s in cp else "lookup", ha="center", va="top", fontsize=7, color="#555")
ax.set_ylabel("ternary − control CE gap"); ax.legend()
ax.set_title("Ternary (1.58-bit): perfect in-distribution, but does computation break OOD?\n(prediction: lookup free both; compute ~0 in-dist, positive OOD)")
plt.tight_layout(); plt.savefig("runs_ternary/exp_ternary_ood.png", dpi=120)
print("\nsaved runs_ternary/exp_ternary_ood.png")
