"""C2 — retroactive-fetch control. Probe the target's location (early/late) from the residual at
(a) the LAST LIST token (before the query is seen) vs (b) the '=' working position (after the query).
Retroactive fetch => decodable at '=' but chance at the list position (the model can't know which
entity is the target until it reads the query).
"""
import json, numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import retrieval_data as D
from retrieval_model import MiniQwen

ck = torch.load("runs/ckpt.pt", map_location="cpu"); vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
model = MiniQwen(cfg, len(vocab), K); model.load_state_dict(ck["control"]); model.eval()
data = [d for d in json.load(open("runs/probe_data.json")) if d["vector"] == "pos"]

@torch.no_grad()
def capture_at(prompt_toks, pos):
    idx = torch.tensor([tok.enc(prompt_toks)]); T = idx.shape[1]
    x = model.embed(idx); cos, sin = model.cos[:T], model.sin[:T]; sites = {}
    for L, b in enumerate(model.blocks):
        a, _ = b.attn(b.n1(x), cos, sin); x = x + a; sites[f"L{L}:SA"] = x[0, pos].numpy().copy()
        m, _ = b.mlp(b.n2(x)); x = x + m; sites[f"L{L}:MLP"] = x[0, pos].numpy().copy()
    return sites

def prompt_of(d): return [str(e) for e in d["list"]] + ["|", "Q"] + list(map(str, d["query"])) + ["="]

rows = []
for d in data:
    pt = prompt_of(d)
    last_list = len(d["list"]) - 1                 # last list token, BEFORE '| Q query ='
    eq = len(pt) - 1                                # the '=' working position, AFTER the query
    rows.append((d, capture_at(pt, last_list), capture_at(pt, eq)))
SITES = list(rows[0][1].keys())

def probe(which):
    pairs = sorted({d["pair"] for d, _, _ in rows}); tr = set(pairs[:int(len(pairs)*.7)])
    A = [(d, r) for d, l, e in rows for r in [l if which == "list" else e] if d["pair"] in tr]
    B = [(d, r) for d, l, e in rows for r in [l if which == "list" else e] if d["pair"] not in tr]
    acc = {}
    for s in SITES:
        clf = LogisticRegression(max_iter=2000).fit(np.stack([r[s] for _, r in A]), [d["region"] for d, _ in A])
        acc[s] = clf.score(np.stack([r[s] for _, r in B]), [d["region"] for d, _ in B])
    return acc

acc_list = probe("list"); acc_eq = probe("eq")
print("=== C2: decode target location at LIST position vs '=' working position ===")
print(f"{'site':<9}{'@last-list':>12}{'@=':>8}")
for s in SITES:
    print(f"{s:<9}{acc_list[s]:>12.2f}{acc_eq[s]:>8.2f}")
ml, me = max(acc_list.values()), max(acc_eq.values())
print(f"\nbest @last-list {ml:.2f}   best @= {me:.2f}   (chance 0.50)")
verdict = ("RETROACTIVE confirmed: target location unknown while reading, built at '='"
           if ml <= 0.60 and me >= 0.85 else
           "NOT clean: target location partly decodable at the list position too")
print("VERDICT:", verdict)

x = np.arange(len(SITES))
plt.figure(figsize=(11, 5))
plt.plot(x, [acc_list[s] for s in SITES], "s-", c="#888", label="@ last list token (before query)")
plt.plot(x, [acc_eq[s] for s in SITES], "o-", c="#d95f2b", label="@ '=' (after query)")
plt.axhline(0.5, ls=":", c="#333", label="chance"); plt.ylim(0.4, 1.0)
plt.xticks(x, SITES, rotation=45, ha="right"); plt.ylabel("target-location probe accuracy")
plt.title("C2 — is the fetch retroactive?\ntarget location is unknown while reading the list, decodable only at the working position")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("runs/exp_c2.png", dpi=120)
print("saved runs/exp_c2.png")
