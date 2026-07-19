"""Residual-capture harness + linear probe, on the single-vector minimal pairs.
Captures the residual at every (layer x {post-attn, post-MLP}) site at the working
position (the '=' token), then probes:
  FETCH vector  -> can we decode which ENTITY the query fetches?  (6-class)
  POS   vector  -> can we decode the target's LOCATION (early/late)? (binary)
Answers: where is the operator argument represented, post-attn vs post-MLP, and is
it a different site from the addressing/position representation.
"""
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import retrieval_data as D
from retrieval_model import MiniQwen

ck = torch.load("runs/ckpt.pt", map_location="cpu"); vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
model = MiniQwen(cfg, len(vocab), K); model.load_state_dict(ck["control"]); model.eval()   # fp32 control
data = json.load(open("runs/probe_data.json"))
OBJ = list(D.OBJECTS); OBJ_ID = {o: i for i, o in enumerate(OBJ)}

@torch.no_grad()
def capture(prompt_toks):
    """residual at every (layer, post-attn/post-mlp) site, at the last (working) position."""
    idx = torch.tensor([tok.enc(prompt_toks)]); T = idx.shape[1]
    x = model.embed(idx); cos, sin = model.cos[:T], model.sin[:T]; sites = {}
    for L, b in enumerate(model.blocks):
        a, _ = b.attn(b.n1(x), cos, sin); x = x + a; sites[f"L{L}.att"] = x[0, -1].numpy().copy()
        m, _ = b.mlp(b.n2(x)); x = x + m; sites[f"L{L}.mlp"] = x[0, -1].numpy().copy()
    return sites

def prompt_of(d):
    lst = [str(e) for e in d["list"]]
    return lst + ["|", "Q"] + list(map(str, d["query"])) + ["="]

# capture all
feat = {d["pair"] + "_" + d.get("region", d["X"]) + str(id(d)): None for d in data}
rows = []
for d in data:
    sites = capture(prompt_of(d))
    rows.append((d, sites))
    if len(rows) % 400 == 0: print("captured", len(rows), flush=True)

SITES = list(rows[0][1].keys())
def matrix(examples): return {s: np.stack([sit[s] for _, sit in examples]) for s in SITES}

fetch = [(d, s) for d, s in rows if d["vector"] == "fetch"]
pos = [(d, s) for d, s in rows if d["vector"] == "pos"]

def probe(examples, label_fn, splitkey):
    """pair-split logistic probe per site; returns test accuracy per site."""
    pairs = sorted({d["pair"] for d, _ in examples})
    tr = set(pairs[:int(len(pairs) * 0.7)])
    Xtr = [(d, s) for d, s in examples if d["pair"] in tr]
    Xte = [(d, s) for d, s in examples if d["pair"] not in tr]
    acc = {}
    for site in SITES:
        A = np.stack([s[site] for _, s in Xtr]); ya = [label_fn(d) for d, _ in Xtr]
        B = np.stack([s[site] for _, s in Xte]); yb = [label_fn(d) for d, _ in Xte]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(A, ya)
        acc[site] = clf.score(B, yb)
    return acc

acc_fetch = probe(fetch, lambda d: OBJ_ID[d["X"]], "pair")           # which entity (6-class, chance 1/6)
acc_pos = probe(pos, lambda d: d["region"], "pair")                   # early/late (binary, chance 1/2)

print("\n=== linear probe accuracy per site (fp32 control) ===")
print(f"{'site':<9}{'fetch-entity':>14}{'target-position':>17}")
for s in SITES:
    print(f"{s:<9}{acc_fetch[s]:>14.2f}{acc_pos[s]:>17.2f}")
print(f"chance: fetch {1/len(OBJ):.2f}  position 0.50")

# plot
x = np.arange(len(SITES))
plt.figure(figsize=(12, 5.5))
plt.plot(x, [acc_fetch[s] for s in SITES], "o-", c="#3E7CB1", label="FETCH vector: decode which entity")
plt.plot(x, [acc_pos[s] for s in SITES], "s-", c="#d95f2b", label="POSITION vector: decode target location")
plt.axhline(1/len(OBJ), ls=":", c="#3E7CB1", alpha=.6); plt.axhline(0.5, ls=":", c="#d95f2b", alpha=.6)
for i, s in enumerate(SITES):
    if s.endswith(".mlp"): plt.axvspan(i-0.5, i+0.5, color="#eee", zorder=0)
plt.xticks(x, SITES, rotation=45, ha="right"); plt.ylim(0, 1.02)
plt.ylabel("probe test accuracy (pair-split)")
plt.title("Where is each operator vector decoded? (shaded = post-MLP sites)\nfetch-entity vs target-position, across the residual stream")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("runs/exp_probe.png", dpi=120)
print("\nsaved runs/exp_probe.png")
