"""Does L0:MLP build the fetch bias? Ablate L0:MLP's write at the working position,
re-probe where fetch-entity and target-position become decodable.
  position collapses at L1:SA under ablation -> L0:MLP builds the seek-query (MLP decides)
  position survives                          -> L1:SA attention did it directly (Q-composition)
Site naming: L{n}:SA = post-attention residual, L{n}:MLP = post-MLP residual.
"""
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import retrieval_data as D
from retrieval_model import MiniQwen

ck = torch.load("runs/ckpt.pt", map_location="cpu"); vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
model = MiniQwen(cfg, len(vocab), K); model.load_state_dict(ck["control"]); model.eval()
data = json.load(open("runs/probe_data.json")); OBJ = list(D.OBJECTS); OID = {o: i for i, o in enumerate(OBJ)}

@torch.no_grad()
def capture(prompt_toks, ablate=None):
    """residual at every site at working (last) position; ablate zeros a sub-block's write there."""
    idx = torch.tensor([tok.enc(prompt_toks)]); T = idx.shape[1]
    x = model.embed(idx); cos, sin = model.cos[:T], model.sin[:T]; sites = {}
    for L, b in enumerate(model.blocks):
        a, _ = b.attn(b.n1(x), cos, sin)
        if ablate == f"L{L}:SA": a = a.clone(); a[0, -1] = 0
        x = x + a; sites[f"L{L}:SA"] = x[0, -1].numpy().copy()
        m, _ = b.mlp(b.n2(x))
        if ablate == f"L{L}:MLP": m = m.clone(); m[0, -1] = 0
        x = x + m; sites[f"L{L}:MLP"] = x[0, -1].numpy().copy()
    return sites

def prompt_of(d):
    return [str(e) for e in d["list"]] + ["|", "Q"] + list(map(str, d["query"])) + ["="]

def run_probe(ablate):
    cap = [(d, capture(prompt_of(d), ablate)) for d in data]
    SITES = list(cap[0][1].keys())
    def probe(examples, label_fn):
        pairs = sorted({d["pair"] for d, _ in examples}); tr = set(pairs[:int(len(pairs) * .7)])
        A = [(d, s) for d, s in examples if d["pair"] in tr]; B = [(d, s) for d, s in examples if d["pair"] not in tr]
        acc = {}
        for site in SITES:
            clf = LogisticRegression(max_iter=2000).fit(np.stack([s[site] for _, s in A]), [label_fn(d) for d, _ in A])
            acc[site] = clf.score(np.stack([s[site] for _, s in B]), [label_fn(d) for d, _ in B])
        return acc
    fetch = [(d, s) for d, s in cap if d["vector"] == "fetch"]; pos = [(d, s) for d, s in cap if d["vector"] == "pos"]
    return SITES, probe(fetch, lambda d: OID[d["X"]]), probe(pos, lambda d: d["region"])

print("baseline ...", flush=True)
SITES, f_base, p_base = run_probe(None)
print("ablate L0:MLP ...", flush=True)
_, f_abl, p_abl = run_probe("L0:MLP")

print(f"\n{'site':<9}{'pos base':>10}{'pos L0:MLP-abl':>16}{'fetch base':>12}{'fetch abl':>11}")
for s in SITES:
    print(f"{s:<9}{p_base[s]:>10.2f}{p_abl[s]:>16.2f}{f_base[s]:>12.2f}{f_abl[s]:>11.2f}")

l1 = "L1:SA"
verdict = ("L0:MLP BUILDS the seek-query (MLP decides the bias)" if p_base[l1] - p_abl[l1] > 0.15
           else "position SURVIVES -> L1:SA attention did it directly (Q-composition, not MLP)")
print(f"\nL1:SA position: baseline {p_base[l1]:.2f} -> L0:MLP-ablated {p_abl[l1]:.2f}  (Δ {p_base[l1]-p_abl[l1]:+.2f})")
print("VERDICT:", verdict)

x = np.arange(len(SITES))
plt.figure(figsize=(12, 5.5))
plt.plot(x, [p_base[s] for s in SITES], "s-", c="#d95f2b", label="position — baseline")
plt.plot(x, [p_abl[s] for s in SITES], "s--", c="#d95f2b", alpha=.6, label="position — L0:MLP ablated")
plt.plot(x, [f_base[s] for s in SITES], "o-", c="#3E7CB1", label="fetch — baseline")
plt.plot(x, [f_abl[s] for s in SITES], "o--", c="#3E7CB1", alpha=.6, label="fetch — L0:MLP ablated")
plt.axhline(0.5, ls=":", c="#d95f2b", alpha=.4); plt.axhline(1/len(OBJ), ls=":", c="#3E7CB1", alpha=.4)
plt.xticks(x, SITES, rotation=45, ha="right"); plt.ylim(0, 1.02); plt.ylabel("probe accuracy")
plt.title("Does L0:MLP build the fetch bias?\nif position at L1:SA drops when L0:MLP is ablated → MLP decides; if not → attention does")
plt.legend(fontsize=9); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("runs/exp_ablate.png", dpi=120)
print("saved runs/exp_ablate.png")
