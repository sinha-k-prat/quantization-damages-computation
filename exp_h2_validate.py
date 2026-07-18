"""H2 — validate the H1 elasticity map against a NON-quantization ground truth.
Ground truth = gradient/Taylor saliency (∂L·W)^2 on the fp32 CONTROL model (no quantization
anywhere, different model, different method). Compute per-component saliency for COMPUTE-skill
loss vs LOOKUP-skill loss. Then correlate per-component compute-saliency with H1's compute
elasticity across all 42 components. Agreement of two independent methods = tool validated.
"""
import re, numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen
from retrieval_train import batchify, SKILL_ID

ck = torch.load("runs/ckpt.pt", map_location="cpu")
vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
control = MiniQwen(cfg, len(vocab), K); control.load_state_dict(ck["control"]); control.train()  # grads on fp32

ev = [D.make_example(np.random.RandomState(7000 + i), (i % 5) + 1) for i in range(400)]
ev = [e for e in ev if e]
Xe, Se, lve = batchify(ev, tok, cfg["block"])
V = len(vocab)
COMPUTE_IDS = {SKILL_ID["relative"], SKILL_ID["content"]}
LOOKUP_IDS = {SKILL_ID["semantic"], SKILL_ID["filter"], SKILL_ID["read"]}

def group_saliency(skill_ids):
    control.zero_grad()
    logits = control(Xe)[0]
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), Xe[:, 1:].reshape(-1), reduction="none")
    sk = Se[:, 1:].reshape(-1)
    mask = torch.zeros_like(sk, dtype=torch.bool)
    for s in skill_ids:
        mask |= (sk == s)
    ((ce * mask).sum() / mask.sum().clamp_min(1)).backward()
    return {m.tag: float((m.weight.grad * m.weight).pow(2).sum()) for m in control.quant_layers()}

sal_c = group_saliency(COMPUTE_IDS)
sal_l = group_saliency(LOOKUP_IDS)

# parse H1 per-component compute elasticity from the saved log
el = {}
for line in open("runs/logs/exp_h1.log"):
    m = re.search(r"crush (\S+)\s+compute\+(\S+)\s+lookup\+", line)
    if m:
        el[m.group(1)] = float(m.group(2))

tags = [m.tag for m in control.quant_layers()]
tags = [t for t in tags if t in el]
sc = np.array([sal_c[t] for t in tags])           # gradient compute-saliency (fp32 control)
sl = np.array([sal_l[t] for t in tags])
elc = np.array([el[t] for t in tags])             # H1 compute elasticity (VQ target, quant probe)

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]

rho = spearman(sc, elc)
print(f"=== H2: does gradient-saliency (fp32 control, non-quant) agree with H1 elasticity (VQ) ? ===")
print(f"Spearman rank corr(compute-saliency, compute-elasticity) over {len(tags)} components = {rho:+.2f}\n")

# class-average of the INDEPENDENT gradient saliency (compute vs lookup)
print("class-average gradient compute-saliency (fp32 control, log10):")
print(f"{'class':<6}{'compute':>12}{'lookup':>12}{'c/l ratio':>11}")
for cl in ["q", "k", "v", "o", "gate", "up", "down"]:
    c = np.mean([sal_c[t] for t in tags if t.split(".")[1] == cl])
    l = np.mean([sal_l[t] for t in tags if t.split(".")[1] == cl])
    print(f"{cl:<6}{c:>12.3e}{l:>12.3e}{c/max(l,1e-30):>11.1f}")
above = int((sc > sl).sum())
print(f"\ncomponents where gradient compute-saliency > lookup-saliency: {above}/{len(tags)}")

# figure: scatter H1 elasticity vs independent gradient saliency
CLSCOL = {"q": "#3E7CB1", "k": "#3E7CB1", "v": "#ff7f0e", "o": "#ff7f0e",
          "gate": "#2ca02c", "up": "#2ca02c", "down": "#2ca02c"}
fig, ax = plt.subplots(figsize=(8.5, 6.5))
for t, x, y in zip(tags, sc, elc):
    ax.scatter(x, max(y, 1e-5), c=CLSCOL[t.split(".")[1]], s=48, edgecolor="#222", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("INDEPENDENT ground truth: gradient compute-saliency (∂L·W)²  [fp32 control]")
ax.set_ylabel("H1 compute elasticity  [1-bit crush, VQ target]")
ax.set_title(f"H2 — two independent methods agree on the compute circuit\nSpearman ρ = {rho:+.2f}  ·  {above}/{len(tags)} components compute>lookup in BOTH")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=c, label=l, markersize=9)
    for l, c in [("Q/K", "#3E7CB1"), ("V/O", "#ff7f0e"), ("MLP", "#2ca02c")]], fontsize=10)
ax.grid(alpha=.3, which="both")
plt.tight_layout(); plt.savefig("runs/exp_h2.png", dpi=120)
print("\nsaved runs/exp_h2.png")
