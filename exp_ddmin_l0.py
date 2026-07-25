"""ddmin within Layer 0: prune the 2,896 skill-carrying flips (all L0 + L5.gate r49) to the
minimal sufficient core, holding search-set sign-acc at the baseline. Reports the core's size,
per-matrix breakdown, and held-out verification.
"""
import json, numpy as np, torch
import retrieval_data as D, retrieval_data_sign as S
from retrieval_model import MiniQwen

torch.set_num_threads(4)
base = torch.load("runs_ternary/ckpt.pt", map_location="cpu")["target"]
ck = torch.load("runs_mse3/lam_30/ckpt.pt", map_location="cpu")
cfg, K, vocab = ck["cfg"], ck["K"], ck["vocab"]; tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary()
for L in m.quant_layers(): L.freeze_scale()
m.load_state_dict(ck["model"]); m.eval()

def key_of(tag):
    p = "attn" if tag.split(".")[1] in ("q", "k", "v", "o") else "mlp"
    return f"blocks.{tag[1]}.{p}.{tag.split('.')[1]}.weight"

layers = {}; core = []                                        # core = L0 flips + L5.gate r49
for L in m.quant_layers():
    s0 = L.s0
    t_new = torch.clamp(torch.round(L.weight.data / s0), -1, 1)
    wb = base[key_of(L.tag)]
    t_old = torch.clamp(torch.round(wb / wb.abs().mean(1, keepdim=True).clamp(min=1e-8)), -1, 1)
    L.weight.data = s0 * t_new; L.quantize = False; layers[L.tag] = L
    for i, j in torch.nonzero(t_new != t_old).tolist():
        rec = (L.tag, i, j, float(s0[i, 0] * t_new[i, j]), float(s0[i, 0] * t_old[i, j]))
        if L.tag.startswith("L0.") or (L.tag == "L5.gate" and i == 49):
            core.append(rec)
        else:
            L.weight.data[i, j] = rec[4]                      # revert non-core permanently (V2 baseline)
print(f"core flips: {len(core)} (all L0 + L5.gate r49); non-core reverted", flush=True)

# batched search eval (48 sign examples) + held-out sets
def build(gen, n, seed0, sign=True):
    exs = []
    for i in range(n * 3):
        e = gen(np.random.RandomState(seed0 + i))
        if e: exs.append(e)
        if len(exs) >= n: break
    rows = []
    for e in exs:
        toks = S.sign_tokens(e)[0] if sign else D._example_tokens(e)[0]
        rows.append((tok.enc(toks), toks.index("A")))
    T = max(len(r[0]) for r in rows)
    X = np.full((len(rows), T), tok.pad, np.int64)
    for i, (ids, a) in enumerate(rows): X[i, :len(ids)] = ids
    return torch.from_numpy(X), torch.tensor([a for _, a in rows])

Xs, As = build(lambda r: S.make_sign_example(r), 48, 5000)
Xv, Av = build(lambda r: S.make_sign_example(r), 70, 9100)
Xo, Ao = build(lambda r: D.make_example(r, r.randint(1, 6)), 70, 8600, sign=False)

@torch.no_grad()
def sacc(X, A):
    lg = m(X)[0]; n = torch.arange(len(A))
    return float(((lg[n, A].argmax(-1) == X[n, A + 1]) & (lg[n, A + 1].argmax(-1) == X[n, A + 2])).float().mean())

@torch.no_grad()
def oacc():
    lg = m(Xo)[0]; n = torch.arange(len(Ao))
    return float((lg[n, Ao].argmax(-1) == Xo[n, Ao + 1]).float().mean())

def setf(k, to_orig):
    tag, i, j, ftv, ov = core[k]
    layers[tag].weight.data[i, j] = ov if to_orig else ftv

acc0 = sacc(Xs, As)
print(f"baseline (core kept): search {acc0:.3f}  heldout {sacc(Xv, Av):.3f}  orig {oacc():.3f}", flush=True)

# LOO ordering
loo = []
for k in range(len(core)):
    setf(k, True); loo.append(sacc(Xs, As)); setf(k, False)
    if k % 400 == 0: print(f"  LOO {k}/{len(core)}", flush=True)
print(f"LOO done: {sum(1 for a in loo if a < acc0)} individually-hurting flips", flush=True)

order = sorted(range(len(core)), key=lambda k: -loo[k])       # most harmless first
reverted = set(); size = 512
while size >= 1:
    cand = [k for k in order if k not in reverted]; idx = 0
    while idx < len(cand):
        grp = cand[idx:idx + size]
        for k in grp: setf(k, True)
        if sacc(Xs, As) >= acc0:
            reverted.update(grp); cand = [k for k in cand if k not in reverted]
        else:
            for k in grp: setf(k, False); idx += size
    print(f"  size {size}: kept {len(core) - len(reverted)}", flush=True)
    size //= 2
for k in [k for k in range(len(core)) if k not in reverted]:  # local pass
    setf(k, True)
    if sacc(Xs, As) >= acc0: reverted.add(k)
    else: setf(k, False)
kept = [k for k in range(len(core)) if k not in reverted]

va, oa = sacc(Xv, Av), oacc()
print(f"\n=== MINIMAL CORE: {len(kept)} flips (from {len(core)}) ===")
print(f"search {sacc(Xs, As):.3f}  HELD-OUT sign {va:.3f}  orig {oa:.3f}")
bym = {}
for k in kept: bym[core[k][0]] = bym.get(core[k][0], 0) + 1
print("per-matrix breakdown of the core:")
for t, n in sorted(bym.items(), key=lambda kv: -kv[1]): print(f"  {t:<9} {n}")
json.dump(dict(n_core_start=len(core), n_minimal=len(kept), heldout_sign=va, orig=oa,
               by_matrix=bym, flips=[core[k][:3] for k in kept]),
          open("runs_mse3/ddmin.json", "w"))
print("saved runs_mse3/ddmin.json")
