"""Exp E-b — recall as the fragile ingredient (matched OBJECT-readout, removes E-a's
token-type confound). Graded selection complexity, all emitting an object:
  T1 '8th from left'   -> position only          (no filter, no recall)
  T3 '3rd object'      -> filter + position       (surface type, NO recalled knowledge)
  T4 '1st round object'-> recall(shape)+filter+pos (needs implicit knowledge)
Measure quant gap at the identification-result object token (the token before 'A').
Prediction: gap escalates, with the big jump at T3->T4 (adding recalled-knowledge binding).
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
def gap_at_id(ex):
    """CE (control, target) at the identification object token = the token before 'A'."""
    toks, pl, sk = D._example_tokens(ex)
    a = next((i for i in range(pl, len(toks)) if toks[i] == "A"), None)
    if a is None or a - 2 < pl:
        return None
    idt = a - 2                              # identified object: '... obj ; A obj' -> a-2
    if toks[idt] not in D.SHAPE:             # ensure it's an object readout
        return None
    ids = torch.tensor([tok.enc(toks)])
    cc = float(F.cross_entropy(control(ids)[0][0, idt - 1:idt], ids[0, idt:idt + 1]))
    tt = float(F.cross_entropy(target(ids)[0][0, idt - 1:idt], ids[0, idt:idt + 1]))
    return cc, tt

def cohort(qt, lo, hi, n, seed):
    rng = np.random.RandomState(seed); out = []; tries = 0
    while len(out) < n and tries < 40000:
        tries += 1
        els = D.make_list(rng, rng.randint(lo, hi))
        r = D.SOLVERS[qt](rng, els)
        if not r: continue
        q, ans, steps = r
        v = gap_at_id(dict(list=els, qtype="x", level=3, query=q, answer=ans, steps=steps))
        if v: out.append(v)
    return np.array(out)

# each query tested at ITS OWN native training length (+matched OOD = train-max .. +4)
# T1=L1(3-5), T3=L2(5-6), T4=L3(6-7)
COH = [("T1\nposition", "T1", (3, 6), (6, 10)),
       ("T3\nfilter+pos", "T3", (5, 7), (7, 11)),
       ("T4\nRECALL+bind", "T4", (6, 8), (8, 12))]
print("=== Exp E-b: recall as the fragile ingredient (each at NATIVE length) ===\n")
G = {}
for lbl, qt, ind, ood in COH:
    for rlab, (lo, hi) in [("in-dist", ind), ("OOD", ood)]:
        a = cohort(qt, lo, hi, 400, 7)
        if len(a):
            cc, tt = a[:, 0].mean(), a[:, 1].mean(); G[(qt, rlab)] = tt - cc
            print(f"{qt} {rlab:<8} len{lo}-{hi-1:<3} control {cc:6.3f} target {tt:6.3f} gap {tt-cc:+.4f} n={len(a)}")
    print()
regimes = [("in-dist", None), ("OOD", None)]

fig, ax = plt.subplots(figsize=(9.5, 5.5)); x = np.arange(len(COH)); w = 0.36
for i, (rlab, _) in enumerate(regimes):
    ax.bar(x + (i - .5) * w, [G.get((qt, rlab), float("nan")) for _, qt, _, _ in COH], w, label=rlab)
ax.axhline(0, color="#000", lw=.8); ax.set_xticks(x); ax.set_xticklabels([l for l, _, _, _ in COH])
ax.set_ylabel("quant gap at object-identification token (target − control CE)")
ax.set_title("Exp E-b (native length) — recall is NOT the fragile ingredient\nOOD gap tracks how far the COMPUTED QUANTITY extrapolates (T1 counts to unseen positions);\nrecall+bind (T4) is the MOST robust — refutes the recall reframe")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("runs/exp_eb.png", dpi=120)
print("saved runs/exp_eb.png")
