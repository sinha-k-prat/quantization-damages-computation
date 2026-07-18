"""Exp 1b — quant gap at the `pos N` token vs list length.
Prediction (from 1a's regularizer/brittle-OOD flip): the gap should rise
monotonically with length and cross zero near the training max (list len 12).
Negative (target better) inside training range, positive (target worse) OOD.
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
POS_TOK = "pos"
QTS = ["T6", "T7", "T8", "T9", "T10"]          # all query types that emit a `pos N`
TRAIN_MAX = 12

@torch.no_grad()
def pos_ce(ex):
    """control & target CE at the first `pos N` number token, or None."""
    toks, pl, sk = D._example_tokens(ex)
    first_pos = next((i for i in range(pl, len(toks)) if toks[i] == POS_TOK), None)
    if first_pos is None or first_pos + 1 >= len(toks):
        return None
    numi = first_pos + 1
    ids = torch.tensor([tok.enc(toks)])
    cc = float(F.cross_entropy(control(ids)[0][0, numi - 1:numi], ids[0, numi:numi + 1]))
    tt = float(F.cross_entropy(target(ids)[0][0, numi - 1:numi], ids[0, numi:numi + 1]))
    return cc, tt

lengths = list(range(5, 25))
res = {}
for L in lengths:
    rng = np.random.RandomState(500 + L); cc_l, tt_l = [], []; tries = 0
    while len(cc_l) < 250 and tries < 6000:
        tries += 1
        els = D.make_list(rng, L)
        r = D.SOLVERS[QTS[rng.randint(len(QTS))]](rng, els)
        if not r: continue
        q, ans, steps = r
        pc = pos_ce(dict(list=els, qtype="x", level=5, query=q, answer=ans, steps=steps))
        if pc: cc_l.append(pc[0]); tt_l.append(pc[1])
    if cc_l:
        res[L] = (np.mean(cc_l), np.mean(tt_l), len(cc_l))

print(f"{'len':>4}{'control CE':>12}{'target CE':>12}{'gap(t-c)':>11}{'n':>6}   {'range'}")
for L in lengths:
    if L in res:
        c, t, n = res[L]
        print(f"{L:>4}{c:>12.4f}{t:>12.4f}{t-c:>+11.4f}{n:>6}   {'TRAIN' if L<=TRAIN_MAX else 'OOD'}")

xs = [L for L in lengths if L in res]
gap = [res[L][1] - res[L][0] for L in xs]
cc = [res[L][0] for L in xs]; tt = [res[L][1] for L in xs]
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].axvspan(4.5, TRAIN_MAX + .5, color="#2ca02c", alpha=.08, label="training range")
ax[0].axhline(0, color="#000", lw=.8); ax[0].axvline(TRAIN_MAX + .5, color="#C9A227", lw=2, ls="--", label="train max (12)")
ax[0].plot(xs, gap, "o-", color="#d62728")
ax[0].set_xlabel("list length"); ax[0].set_ylabel("quant gap at `pos N`  (target − control CE)")
ax[0].set_title("gap grows with length, crosses 0 near train max"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].axvspan(4.5, TRAIN_MAX + .5, color="#2ca02c", alpha=.08)
ax[1].axvline(TRAIN_MAX + .5, color="#C9A227", lw=2, ls="--")
ax[1].plot(xs, cc, "o-", color="#444", label="control (fp32)")
ax[1].plot(xs, tt, "s-", color="#2ca02c", label="target (VQ)")
ax[1].set_xlabel("list length"); ax[1].set_ylabel("CE at `pos N`")
ax[1].set_title("both degrade OOD — target degrades faster"); ax[1].legend(); ax[1].grid(alpha=.3)
fig.suptitle("Exp 1b — positional-computation brittleness vs list length (quantization = regularizer in-dist, brittle OOD)", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig("runs/exp1b.png", dpi=120)
print("\nsaved runs/exp1b.png")
