"""Q2: find the TRUE minimal flip set. From the lam=30 model (284 flips, sign-acc 1.0):
  CONTROL: revert ALL flips -> if skill survives, it lives in continuous params (scales/biases/
           norms/embeddings), not in the symbols — report loudly.
  LOO:     revert each flip alone -> individually-necessary flips.
  PRUNE:   greedy ddmin-style group reverting (most-harmless-first) until no flip can be removed
           while search-set sign-acc stays 1.0. Verify final set on held-out sign + orig sets.
  STRUCTURE: is the minimal set a single column / row / span? (the 'miracle' check)
Method note: ternary weights are MATERIALIZED (quantize=False, W := s_ft * trit) so each revert is
an exact single-symbol edit (w[i,j] := s_ft_i * orig_trit) with no scale-coupling artifacts.
"""
import json, os, numpy as np, torch
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen

torch.set_num_threads(int(os.environ.get("THREADS", 2)))
base = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
ck = torch.load("runs_mse/lam_30/ckpt.pt", map_location="cpu")
cfg, K, vocab = ck["cfg"], ck["K"], ck["vocab"]; tok = D.Tokenizer(vocab)

def trit(W):
    s = W.abs().mean(1, keepdim=True).clamp(min=1e-8)
    return torch.clamp(torch.round(W / s), -1, 1)

m0 = MiniQwen(cfg, len(base["vocab"]), K); m0.enable_ternary(); m0.load_state_dict(base["target"])
orig = {L.tag: trit(L.weight.data) for L in m0.quant_layers()}; del m0
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary(); m.load_state_dict(ck["model"]); m.eval()

# materialize ternary -> plain weights (forward identical, single-value edits exact)
layers = {}
flips = []                                        # (tag, i, j, ft_val, orig_val)
for L in m.quant_layers():
    s = L.weight.data.abs().mean(1, keepdim=True).clamp(min=1e-8)
    t = torch.clamp(torch.round(L.weight.data / s), -1, 1)
    L.weight.data = s * t; L.quantize = False; layers[L.tag] = L
    f = torch.nonzero(t != orig[L.tag])
    for i, j in f.tolist():
        flips.append((L.tag, i, j, float(s[i, 0] * t[i, j]), float(s[i, 0] * orig[L.tag][i, j])))
print(f"{len(flips)} flips materialized", flush=True)

# batched eval sets
def build(gen, n, seed0):
    exs = []
    for i in range(n * 3):
        e = gen(np.random.RandomState(seed0 + i))
        if e: exs.append(e)
        if len(exs) >= n: break
    rows = []
    for e in exs:
        toks, pl, sk = D._example_tokens(e)
        a = toks.index("A"); rows.append((tok.enc(toks), a))
    T = max(len(r[0]) for r in rows)
    X = np.full((len(rows), T), tok.pad, np.int64)
    for i, (ids, a) in enumerate(rows): X[i, :len(ids)] = ids
    return torch.from_numpy(X), torch.tensor([a for _, a in rows])

Xs, As = build(lambda r: N.make_neg_example(r), 48, 5000)          # sign search set
Xv, Av = build(lambda r: N.make_neg_example(r), 70, 9100)          # sign verify (held out)
Xo, Ao = build(lambda r: D.make_example(r, r.randint(1, 6)), 70, 8600)  # orig verify

@torch.no_grad()
def acc(X, A):
    lg = m(X)[0]
    pred = lg[torch.arange(len(A)), A].argmax(-1)
    gold = X[torch.arange(len(A)), A + 1]
    return float((pred == gold).float().mean())

def set_flip(k, to_orig):
    tag, i, j, ftv, ov = flips[k]
    layers[tag].weight.data[i, j] = ov if to_orig else ftv

print(f"baseline (all {len(flips)} flips active): search {acc(Xs, As):.3f}  verify {acc(Xv, Av):.3f}", flush=True)
for k in range(len(flips)): set_flip(k, True)
zero_acc = acc(Xs, As)
print(f"CONTROL zero-flip (all reverted): sign-acc {zero_acc:.3f}  <-- if high, skill is in continuous params!", flush=True)
for k in range(len(flips)): set_flip(k, False)

# LOO
loo = []
for k in range(len(flips)):
    set_flip(k, True); loo.append(acc(Xs, As)); set_flip(k, False)
    if k % 60 == 0: print(f"  LOO {k}/{len(flips)}", flush=True)
nec = [k for k, a in enumerate(loo) if a < 1.0]
print(f"LOO: {len(nec)} individually-necessary flips (acc<1.0 when reverted alone); "
      f"worst {min(loo):.3f}", flush=True)

# greedy ddmin prune: try reverting groups (harmless-first order), shrink group size on failure
order = sorted(range(len(flips)), key=lambda k: -loo[k])           # most harmless first
reverted = set()
size = 64
while size >= 1:
    prog = False; idx = 0
    cand = [k for k in order if k not in reverted]
    while idx < len(cand):
        group = cand[idx:idx + size]
        for k in group: set_flip(k, True)
        if acc(Xs, As) >= 1.0:
            reverted.update(group); prog = True
        else:
            for k in group: set_flip(k, False)
        idx += size
        cand = [k for k in cand if k not in reverted] if prog else cand
    size //= 2
kept = [k for k in range(len(flips)) if k not in reverted]
# second local pass
for k in list(kept):
    set_flip(k, True)
    if acc(Xs, As) >= 1.0: kept.remove(k); reverted.add(k)
    else: set_flip(k, False)

sa, va, oa = acc(Xs, As), acc(Xv, Av), acc(Xo, Ao)
print(f"\n=== MINIMAL SET: {len(kept)} flips (from {len(flips)}) ===")
print(f"search sign-acc {sa:.3f} | HELD-OUT sign-acc {va:.3f} | orig-acc {oa:.3f}")
det = [dict(tag=flips[k][0], row=flips[k][1], col=flips[k][2],
            d=f"{'+' if flips[k][3]>flips[k][4] else '-'}") for k in kept]
bytag = {}
for d in det: bytag.setdefault(d["tag"], []).append((d["row"], d["col"]))
print("\nstructure of the minimal set:")
for tag, rc in sorted(bytag.items(), key=lambda kv: -len(kv[1])):
    rows = sorted(set(r for r, _ in rc)); cols = sorted(set(c for _, c in rc))
    print(f"  {tag:<9} {len(rc):>3} flips | {len(rows)} distinct rows, {len(cols)} distinct cols"
          + (f" | ROWS={rows}" if len(rows) <= 4 else "") + (f" COLS={cols}" if len(cols) <= 4 else ""))
json.dump(dict(n_total=len(flips), zero_flip_acc=zero_acc, n_loo_necessary=len(nec),
               n_minimal=len(kept), heldout_sign=va, orig=oa, minimal=det),
          open("runs_mse/minimal_set.json", "w"))
print("\nsaved runs_mse/minimal_set.json")
