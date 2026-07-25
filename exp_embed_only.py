"""Embedding-graft test: does the sign skill live in the EMBEDDING TABLE?
A: 100% original ternary network + fine-tuned embedding (all rows)  -> sign 0.986 / orig 1.000
B: original network + ONLY the 34 new-token embedding rows          -> sign 0.429 / orig 1.000
Conclusion: skill = embedding-space programming of frozen circuits; zero weight edits needed.
"""
import numpy as np, torch
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen
base = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
ck = torch.load("runs_mse/lam_30/ckpt.pt", map_location="cpu")
cfg, K, vocab = ck["cfg"], ck["K"], ck["vocab"]; tok = D.Tokenizer(vocab)
n_old = base["target"]["embed.weight"].shape[0]
def build(variant):
    m = MiniQwen(cfg, len(vocab), K); m.enable_ternary(); nsd = m.state_dict()
    with torch.no_grad():
        for k, v in base["target"].items():
            if k in nsd and nsd[k].shape == v.shape: nsd[k].copy_(v)
        nsd["embed.weight"][:n_old].copy_(base["target"]["embed.weight"])
        if variant == "A": nsd["embed.weight"].copy_(ck["model"]["embed.weight"])
        elif variant == "B": nsd["embed.weight"][n_old:].copy_(ck["model"]["embed.weight"][n_old:])
    m.load_state_dict(nsd); m.eval(); return m
def build_eval(gen, n, seed0):
    exs = []
    for i in range(n*3):
        e = gen(np.random.RandomState(seed0+i))
        if e: exs.append(e)
        if len(exs) >= n: break
    rows = [(tok.enc(D._example_tokens(e)[0]), D._example_tokens(e)[0].index("A")) for e in exs]
    T = max(len(r[0]) for r in rows); X = np.full((len(rows), T), tok.pad, np.int64)
    for i, (ids, a) in enumerate(rows): X[i, :len(ids)] = ids
    return torch.from_numpy(X), torch.tensor([a for _, a in rows])
Xs, As = build_eval(lambda r: N.make_neg_example(r), 70, 9100)
Xo, Ao = build_eval(lambda r: D.make_example(r, r.randint(1, 6)), 70, 8600)
@torch.no_grad()
def acc(m, X, A):
    lg = m(X)[0]; return float((lg[torch.arange(len(A)), A].argmax(-1) == X[torch.arange(len(A)), A+1]).float().mean())
for v, desc in [("A", "orig network + full FT embedding"), ("B", "orig network + only 34 new-token rows")]:
    m = build(v); print(f"{v}: {desc}: sign {acc(m,Xs,As):.3f}  orig {acc(m,Xo,Ao):.3f}")
