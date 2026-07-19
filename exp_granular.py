"""Granular precision-recovery: restore precision to a fraction of ONE fragile matrix
by ROW (output dims) / COLUMN (input dims) / SVD-SUBSPACE (rotated spans), ranked by
quantization-error magnitude, and measure per-SKILL and per-LEVEL CE vs. fraction restored.
Fast-saturating curve => the matrix's effect on that skill is localized/low-rank.
"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen
from retrieval_train import batchify, split_loss

ck = torch.load("runs/ckpt.pt", map_location="cpu"); vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_quant(); m.load_state_dict(ck["target"]); m.eval()

# eval set spanning skills x levels (in-dist, for the complexity axis)
ev = [D.make_example(np.random.RandomState(7000+i), (i % 5)+1) for i in range(700)]
ev = [e for e in ev if e]
Xe, Se, lve = batchify(ev, tok, cfg["block"])
# OOD eval (longer lists) — where the fragile penalty is real and recoverable
oo = [D.make_ood_example(np.random.RandomState(9000+i), 13, 20) for i in range(400)]
oo = [e for e in oo if e]
Xo, So, lvo = batchify(oo, tok, cfg["block"])

# pick the fragile matrix: highest relative+content CE rise when crushed to 1-bit, among value/output
def q_of(w, cb): return torch.gather(cb, 1, (w[:, :, None]-cb[None if False else slice(None), None, :]).abs().argmin(-1)) if False else None
comps = {L.tag: L for L in m.quant_layers()}
@torch.no_grad()
def per_skill_now():
    _, ps, mat = split_loss(m(Xe)[0], Xe, Se, lve); return ps, mat
base_ps, base_mat = per_skill_now()
FRAG = ["relative", "content"]
TARGET = "L1.v"   # top value-path component from the elasticity map (H1)
comp = comps[TARGET]
W = comp.weight.data.clone(); CB = comp.codebook.data.clone()
idx = (W[:, :, None]-CB[:, None, :]).abs().argmin(-1); Wq = torch.gather(CB, 1, idx)
E = (W - Wq)                                        # quantization error
U, S, Vt = torch.linalg.svd(E, full_matrices=False)
row_ord = torch.argsort(E.norm(dim=1), descending=True)   # rows by error norm (outputs)
col_ord = torch.argsort(E.norm(dim=0), descending=True)   # cols by error norm (inputs)
Rrank = S.numel()

@torch.no_grad()
def measure(Wmix):
    comp.weight.data = Wmix; comp.quantize = False       # use the mixed weight directly
    _, ps, mat = split_loss(m(Xe)[0], Xe, Se, lve)       # in-dist per-level
    _, pso, _ = split_loss(m(Xo)[0], Xo, So, lvo)        # OOD (recoverable penalty)
    comp.weight.data = W; comp.quantize = True            # restore
    return ps, mat, pso

def restore_rows(f):
    Wm = Wq.clone(); k = int(np.ceil(f*W.shape[0])); r = row_ord[:k]; Wm[r] = W[r]; return Wm
def restore_cols(f):
    Wm = Wq.clone(); k = int(np.ceil(f*W.shape[1])); c = col_ord[:k]; Wm[:, c] = W[:, c]; return Wm
def restore_svd(f):
    r = max(1, int(np.ceil(f*Rrank))); return Wq + (U[:, :r]*S[:r]) @ Vt[:r]

fracs = [0.0, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0]
modes = {"row (outputs)": restore_rows, "col (inputs)": restore_cols, "svd (subspace)": restore_svd}
# baseline at f=0 = fully quantized (Wq). record fragile-skill CE and per-level.
print(f"target matrix {TARGET}  shape {tuple(W.shape)}  rank {Rrank}")
print(f"baseline fully-quantized CE: relative {base_ps['relative']:.4f}  content {base_ps['content']:.4f}\n")
print(f"baseline OOD fully-quantized CE: relative {split_loss(m(Xo)[0],Xo,So,lvo)[1]['relative']:.4f}\n")
curves = {mode: {"frag": [], "ood": [], "bylevel": {L: [] for L in range(1, 6)}} for mode in modes}
for mode, fn in modes.items():
    for f in fracs:
        Wm = Wq.clone() if f == 0 else fn(f)
        ps, mat, pso = measure(Wm)
        curves[mode]["frag"].append(np.mean([ps[s] for s in FRAG]))
        curves[mode]["ood"].append(np.mean([pso[s] for s in FRAG]))
        for L in range(1, 6):
            vals = [mat[L][s] for s in FRAG if mat[L].get(s) is not None]
            curves[mode]["bylevel"][L].append(np.mean(vals) if vals else np.nan)
    print(f"{mode:<16} in-dist:", " ".join(f"{v:.3f}" for v in curves[mode]["frag"]),
          " | OOD:", " ".join(f"{v:.3f}" for v in curves[mode]["ood"]))

# fraction to reach 90% of total change (localization metric), per mode & per level
def frac90(ys):
    ys = np.array(ys); tot = ys[-1]-ys[0]
    if abs(tot) < 1e-6: return 1.0
    for f, y in zip(fracs, ys):
        if abs(y-ys[0]) >= 0.9*abs(tot): return f
    return 1.0
print("\nfraction restored to reach 90% of the matrix's effect (lower = more localized):")
print(f"{'mode':<16}{'overall':>9}" + "".join(f"  L{L}" for L in range(1, 6)))
for mode in modes:
    fl = [frac90(curves[mode]["bylevel"][L]) for L in range(1, 6)]
    print(f"{mode:<16}{frac90(curves[mode]['frag']):>9.2f}" + "".join(f"{x:>4.2f}" for x in fl))

# figure
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
col = {"row (outputs)": "#3E7CB1", "col (inputs)": "#e07b3c", "svd (subspace)": "#3E8E5A"}
for mode in modes:
    ax[0].plot(fracs, curves[mode]["ood"], "o-", c=col[mode], label=mode)
ax[0].set_xlabel(f"fraction of {TARGET} restored to full precision"); ax[0].set_ylabel("OOD fragile-skill CE (relative+content)")
ax[0].set_title(f"OOD recovery — how localized is {TARGET}'s fragile computation?\n(steeper drop = fewer weight dims carry it)")
ax[0].legend(); ax[0].grid(alpha=.3)
for mode in modes:
    ax[1].plot(fracs, curves[mode]["frag"], "s--", c=col[mode], alpha=.7, label=mode)
ax[1].set_xlabel(f"fraction of {TARGET} restored"); ax[1].set_ylabel("in-dist fragile-skill CE")
ax[1].set_title("In-distribution (regularization regime)\nrestoring precision slightly HURTS — no penalty to recover")
ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("runs/exp_granular.png", dpi=120)
print("\nsaved runs/exp_granular.png")
