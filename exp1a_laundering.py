"""Experiment 1a — positional laundering.
Bucket the per-token quant gap (target CE - control CE) in 'relative' traces by
position relative to the first `pos N` number token. Prediction: gap is large on
tokens up to & including `pos N` (fragile geometric computation) and collapses to
~0 afterward (position is now a robust content symbol).
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
POS = tok.stoi["pos"]

COHORTS = {
    "easy T6/T7 (L4)":  (["T6", "T7"], 6, 10, 4),
    "hard T8-T10 (L5)": (["T8", "T9", "T10"], 9, 13, 5),
    "OOD longer lists": (["T8", "T9", "T10"], 13, 21, 6),
}

def make_cohort(qts, lo, hi, level, n, seed):
    out = []; rng = np.random.RandomState(seed)
    while len(out) < n:
        els = D.make_list(rng, rng.randint(lo, hi))
        qt = qts[rng.randint(len(qts))]
        r = D.SOLVERS[qt](rng, els)
        if r:
            q, ans, steps = r
            out.append(dict(list=els, qtype=qt, level=level, query=q, answer=ans, steps=steps))
    return out

POS_TOK = "pos"

@torch.no_grad()
def per_token(ex):
    toks, pl, sk = D._example_tokens(ex)
    ids = torch.tensor([tok.enc(toks)])
    ce_c = F.cross_entropy(control(ids)[0][0, :-1], ids[0, 1:], reduction="none")
    ce_t = F.cross_entropy(target(ids)[0][0, :-1], ids[0, 1:], reduction="none")
    first_pos = next((i for i in range(pl, len(toks)) if toks[i] == POS_TOK), None)
    if first_pos is None:
        return None
    numi = first_pos + 1                      # anchor's position-number token
    rows = []
    for t in range(pl, len(toks)):
        rows.append((t - numi, float(ce_c[t - 1]), float(ce_t[t - 1]), toks[t]))
    return rows

def bucket_of(off):
    return "AT pos N" if off == 0 else ("pre (<posN)" if off < 0 else "post (>posN)")

fig, axes = plt.subplots(1, len(COHORTS), figsize=(16, 4.6), sharey=True)
print("=== Experiment 1a: quant gap around first `pos N`, by cohort ===\n")
for ci, (cname, (qts, lo, hi, lvl)) in enumerate(COHORTS.items()):
    exs = make_cohort(qts, lo, hi, lvl, 400, 100 + ci)
    by_off = {}; B = {"pre (<posN)": ([], []), "AT pos N": ([], []), "post (>posN)": ([], [])}
    for ex in exs:
        r = per_token(ex)
        if not r: continue
        for off, cc, tt, tk in r:
            by_off.setdefault(off, ([], []))[0].append(cc); by_off[off][1].append(tt)
            b = bucket_of(off); B[b][0].append(cc); B[b][1].append(tt)
    print(f"[{cname}]  ({len(exs)} examples)")
    print(f"  {'bucket':<14}{'control CE':>11}{'target CE':>11}{'gap(t-c)':>10}{'n':>7}")
    for b, (cs, ts) in B.items():
        print(f"  {b:<14}{np.mean(cs):>11.4f}{np.mean(ts):>11.4f}{np.mean(ts)-np.mean(cs):>+10.4f}{len(cs):>7}")
    print()
    offs = sorted(o for o in by_off if -5 <= o <= 8)
    gap = [np.mean(by_off[o][1]) - np.mean(by_off[o][0]) for o in offs]
    ax = axes[ci]
    ax.axvline(0, color="#C9A227", lw=2); ax.axhline(0, color="#000", lw=.8)
    ax.plot(offs, gap, "o-", color="#d62728")
    ax.set_title(cname, fontsize=10); ax.set_xlabel("offset from `pos N`"); ax.grid(alpha=.3)
axes[0].set_ylabel("quant gap (target − control CE)")
fig.suptitle("Exp 1a — does the quant penalty concentrate at/before `pos N` and collapse after? (gold line = pos N written)",
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig("runs/exp1a.png", dpi=120)
print("saved runs/exp1a.png")
