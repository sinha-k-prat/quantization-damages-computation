"""Sparsity-constrained skill install: add L1 penalty on (W - W_orig) so the model learns the
'sign' skill by flipping as FEW trits as possible (forced to reuse the existing circuit).
Sweep lambda; for each report sign-accuracy AND % trits flipped. Goal: beat the 20.1% from the
unconstrained fine-tune while keeping high accuracy.
"""
import copy, os, numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen

torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
STEPS = 700; BATCH = 48
ck = torch.load("runs_ternary/ckpt.pt", map_location="cpu"); cfg, K = ck["cfg"], ck["K"]
vneg = N.build_vocab_neg(ck["vocab"]); tok = D.Tokenizer(vneg)

m = MiniQwen(cfg, len(vneg), K); m.enable_ternary(); sd = ck["target"]; nsd = m.state_dict()
with torch.no_grad():
    for k, v in sd.items():
        if k in nsd and nsd[k].shape == v.shape: nsd[k].copy_(v)
    nsd["embed.weight"][:sd["embed.weight"].shape[0]].copy_(sd["embed.weight"])
m.load_state_dict(nsd)
init_state = copy.deepcopy(m.state_dict())
W_orig = {L.tag: L.weight.data.clone() for L in m.quant_layers()}
def trit(W):
    s = W.abs().mean(1, keepdim=True).clamp(min=1e-8); return torch.clamp(torch.round(W / s), -1, 1)
orig_trits = {L.tag: trit(W_orig[L.tag]) for L in m.quant_layers()}

def batch(rng, n):
    exs = [N.make_neg_example(rng) if rng.rand() < 0.5 else D.make_example(rng, rng.randint(1, 6)) for _ in range(n)]
    exs = [e for e in exs if e]; rows = []
    for e in exs:
        toks, pl, sk = D._example_tokens(e); ids = tok.enc(toks)[:cfg["block"]]
        rows.append((ids, [0]*min(pl, len(ids)) + [1]*(len(ids)-min(pl, len(ids)))))
    T = max(len(r[0]) for r in rows); X = np.full((len(rows), T), tok.pad, np.int64); Mk = np.zeros((len(rows), T), np.int64)
    for i, (ids, mk) in enumerate(rows): X[i, :len(ids)] = ids; Mk[i, :len(mk)] = mk
    return torch.from_numpy(X), torch.from_numpy(Mk)

def masked_ce(lg, X, Mk):
    V = lg.size(-1); ce = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
    mm = Mk[:, 1:].reshape(-1).float(); return (ce*mm).sum()/mm.sum().clamp_min(1)

@torch.no_grad()
def sign_acc(nn=80):
    m.eval(); ok = 0
    for i in range(nn):
        e = N.make_neg_example(np.random.RandomState(5000+i))
        if not e: continue
        toks, pl, sk = D._example_tokens(e); a = toks.index("A"); ids = torch.tensor([tok.enc(toks)])
        ok += int(m(ids)[0][0, a].argmax()) == ids[0, a+1].item()
    m.train(); return ok/nn

@torch.no_grad()
def flip_pct():
    tf = tot = 0
    for L in m.quant_layers():
        f = (trit(L.weight) != orig_trits[L.tag]); tf += int(f.sum()); tot += f.numel()
    return 100*tf/tot

def l1_edit():
    return sum((L.weight - W_orig[L.tag]).abs().sum() for L in m.quant_layers())

lambdas = [0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
res = []
print(f"{'lambda':>8}{'sign-acc':>10}{'% flips':>9}")
for lam in lambdas:
    m.load_state_dict(init_state); m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    rng = np.random.RandomState(0)
    for step in range(STEPS):
        X, Mk = batch(rng, BATCH); opt.zero_grad()
        lg, vq = m(X); loss = masked_ce(lg, X, Mk) + vq + lam*l1_edit()
        loss.backward(); opt.step()
    acc, fp = sign_acc(), flip_pct(); res.append((lam, acc, fp))
    print(f"{lam:>8.0e}{acc:>10.3f}{fp:>8.2f}%", flush=True)

print("\n=== summary: last-time (unconstrained) was 100% acc @ 20.11% flips ===")
best = min([r for r in res if r[1] >= 0.9], key=lambda r: r[2], default=None)
if best:
    print(f"BEST sparse install: lambda={best[0]:.0e}  sign-acc {best[1]:.0%}  flips {best[2]:.2f}%  "
          f"({20.11/max(best[2],1e-9):.0f}x fewer flips than last time)")

fig, ax = plt.subplots(figsize=(8, 5.5))
fp = [r[2] for r in res]; ac = [r[1] for r in res]
sc = ax.scatter(fp, ac, c=range(len(res)), cmap="viridis", s=90, zorder=3)
for lam, a, f in res: ax.annotate(f"λ={lam:.0e}", (f, a), fontsize=8, xytext=(4, 4), textcoords="offset points")
ax.axvline(20.11, ls="--", c="#d62728", label="last time (unconstrained): 20.1% flips")
ax.set_xlabel("% of trits flipped (fewer = more surgical)"); ax.set_ylabel("sign-skill accuracy")
ax.set_title("L1 edit-penalty: install the new skill with far fewer flips\n(top-left = high accuracy, few flips = surgical reuse)")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("runs_ternary/exp_neg_l1.png", dpi=120)
print("saved runs_ternary/exp_neg_l1.png")
