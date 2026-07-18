"""Exp E-a — is IMPLICIT (recall-and-bind) identification more quant-fragile than
EXPLICIT (identity/position) identification?
  EXPLICIT id token: `pos N` in T6/T7 — find a NAMED object, read its position.
  IMPLICIT id token: the shape-selected object in T4 ("1st flat = book") — must RECALL
                     which objects are flat, then bind to the load-bearing token.
Measure control & target CE at the identification-result token, in-dist vs OOD length.
Prediction (binding hypothesis): implicit gap >> explicit gap.
"""
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen

ck = torch.load("runs/ckpt.pt", map_location="cpu")
vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
control = MiniQwen(cfg, len(vocab), K); control.load_state_dict(ck["control"]); control.eval()
target = MiniQwen(cfg, len(vocab), K); target.enable_quant(); target.load_state_dict(ck["target"]); target.eval()

@torch.no_grad()
def ce_at(ex, marker):
    """CE (control, target) at the token right after the first `marker` token in the target region."""
    toks, pl, sk = D._example_tokens(ex)
    j = next((i for i in range(pl, len(toks)) if toks[i] == marker), None)
    if j is None or j + 1 >= len(toks):
        return None
    idt = j + 1
    ids = torch.tensor([tok.enc(toks)])
    cc = float(F.cross_entropy(control(ids)[0][0, idt - 1:idt], ids[0, idt:idt + 1]))
    tt = float(F.cross_entropy(target(ids)[0][0, idt - 1:idt], ids[0, idt:idt + 1]))
    return cc, tt

def cohort(qts, marker, lo, hi, n, seed):
    rng = np.random.RandomState(seed); out = []; tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        els = D.make_list(rng, rng.randint(lo, hi))
        r = D.SOLVERS[qts[rng.randint(len(qts))]](rng, els)
        if not r: continue
        q, ans, steps = r
        v = ce_at(dict(list=els, qtype="x", level=5, query=q, answer=ans, steps=steps), marker)
        if v: out.append(v)
    return np.array(out)

RUNS = {
    "EXPLICIT id (pos of named anchor)": (["T6", "T7"], "pos"),
    "IMPLICIT id (shape-selected object)": (["T4"], "="),
}
print("=== Exp E-a: explicit(identity) vs implicit(recall+bind) identification ===\n")
res = {}
for name, (qts, marker) in RUNS.items():
    for tag, (lo, hi) in [("in-dist (len 8-12)", (8, 13)), ("OOD (len 13-20)", (13, 21))]:
        a = cohort(qts, marker, lo, hi, 400, 42)
        cc, tt = a[:, 0].mean(), a[:, 1].mean()
        res[(name, tag)] = (cc, tt, tt - cc)
        print(f"{name:<38} {tag:<20} control {cc:6.3f}  target {tt:6.3f}  gap {tt-cc:+.4f}  n={len(a)}")
    print()

# bar chart: gap by cohort x regime
labels = ["EXPLICIT", "IMPLICIT"]; regimes = ["in-dist (len 8-12)", "OOD (len 13-20)"]
names = list(RUNS.keys())
fig, ax = plt.subplots(figsize=(9, 5.5)); x = np.arange(2); w = 0.36
for i, rg in enumerate(regimes):
    vals = [res[(n, rg)][2] for n in names]
    ax.bar(x + (i - .5) * w, vals, w, label=rg)
ax.axhline(0, color="#000", lw=.8); ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("quant gap at identification token (target − control CE)")
ax.set_title("Exp E-a — quantization damages RECALL-AND-BIND identification\nmore than IDENTITY/position identification")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("runs/exp_ea.png", dpi=120)
print("saved runs/exp_ea.png")
