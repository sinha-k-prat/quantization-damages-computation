"""Q1: are the 284 flips at lam=30 a SUBSET of the flip sets of every less-constrained run?
For each lam in {0,0.3,1,3,10}, report |flips(30) ∩ flips(lam)| as % of 284.
High % everywhere -> the 284 are semantically necessary (every recipe finds them).
Low/inconsistent -> multiple solutions; the optimizer's path matters.
Also: adjacent-lambda nesting (is flips(lam_hi) ⊆ flips(lam_lo) down the chain) + Jaccard.
"""
import json, os, torch
import retrieval_data as D
from retrieval_model import MiniQwen

torch.set_num_threads(int(os.environ.get("THREADS", 2)))
base = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
cfg, K = base["cfg"], base["K"]

def trit(W):
    s = W.abs().mean(1, keepdim=True).clamp(min=1e-8)
    return torch.clamp(torch.round(W / s), -1, 1)

m0 = MiniQwen(cfg, len(base["vocab"]), K); m0.enable_ternary(); m0.load_state_dict(base["target"])
orig = {L.tag: trit(L.weight.data) for L in m0.quant_layers()}
TAGS = [L.tag for L in m0.quant_layers()]
del m0

def flipmask(lam):
    ck = torch.load(f"runs_mse/lam_{lam}/ckpt.pt", map_location="cpu")
    m = MiniQwen(ck["cfg"], len(ck["vocab"]), ck["K"]); m.enable_ternary(); m.load_state_dict(ck["model"])
    out = {L.tag: (trit(L.weight.data) != orig[L.tag]) for L in m.quant_layers()}
    del m; return out

LAMS = ["0", "0.3", "1", "3", "10", "30"]
masks = {}
for lam in LAMS:
    masks[lam] = flipmask(lam)
    n = sum(int(v.sum()) for v in masks[lam].values())
    print(f"loaded lam={lam:<4} flips={n}", flush=True)

ref = masks["30"]; nref = sum(int(v.sum()) for v in ref.values())
print(f"\n=== Q1: how much of the lam=30 set ({nref} flips) appears in each run? ===")
print(f"{'lam':>5}{'|set|':>9}{'∩ with 284':>12}{'% of 284':>10}")
res = {}
for lam in LAMS[:-1]:
    inter = sum(int((ref[t] & masks[lam][t]).sum()) for t in TAGS)
    n = sum(int(v.sum()) for v in masks[lam].values())
    res[lam] = round(100 * inter / nref, 1)
    print(f"{lam:>5}{n:>9}{inter:>12}{res[lam]:>9.1f}%")

print("\n=== adjacent-lambda nesting: % of the SMALLER set contained in the next-larger set ===")
for hi, lo in [("30", "10"), ("10", "3"), ("3", "1"), ("1", "0.3"), ("0.3", "0")]:
    nh = sum(int(v.sum()) for v in masks[hi].values())
    inter = sum(int((masks[hi][t] & masks[lo][t]).sum()) for t in TAGS)
    print(f"  flips({hi}) in flips({lo}): {inter}/{nh} = {100*inter/nh:.1f}%")

json.dump(dict(pct_of_284_in=res, n_ref=nref), open("runs_mse/overlap.json", "w"))
print("\nsaved runs_mse/overlap.json")
