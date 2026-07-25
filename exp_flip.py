"""Jacobian-guided ternary sign-flipping. On a correctly-answered example, rank the ternary
weights of one layer by |∂CE/∂W| (the Jacobian), then flip the signs of the top-k (in the
loss-increasing direction) and watch the answer break. Compare to random sign-flips.
Shows the correct computation lives in a few load-bearing trits, findable via the gradient
(instead of the impossible 3^N brute force).
"""
import numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen

ck = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]; tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary(); m.load_state_dict(ck["target"]); m.eval()

# find a correct-but-not-overconfident example (moderate CE -> informative Jacobian + thin margin).
# OOD (longer lists) stresses the model, so it's right but on the edge.
rng = np.random.RandomState(3)
for _ in range(400):
    ex = D.make_ood_example(rng, 13, 20)
    if not ex: continue
    toks, pl, sk = D._example_tokens(ex); ids = torch.tensor([tok.enc(toks)])
    if "A" not in toks: continue
    a = toks.index("A"); ans_pos = a + 1
    with torch.no_grad():
        lg = m(ids)[0][0, ans_pos - 1]
        pred = int(lg.argmax()); c0 = float(F.cross_entropy(lg[None], ids[0, ans_pos:ans_pos + 1]))
    if pred == ids[0, ans_pos].item() and 0.15 < c0 < 3.0:     # correct, but with margin to break
        break
print(f"correct-on-the-edge example: '{' '.join(map(str,ex['query']))}' -> {ex['answer']}  (baseline CE {c0:.3f})")

# pick the layer whose weights the answer-CE is most sensitive to (largest gradient norm)
m.zero_grad(); logits = m(ids)[0]
F.cross_entropy(logits[0, ans_pos - 1:ans_pos], ids[0, ans_pos:ans_pos + 1]).backward()
best = max(m.quant_layers(), key=lambda L: float(L.weight.grad.norm()))
LAYER = best.tag; lin = best
print(f"most answer-sensitive layer (max |grad|): {LAYER}")
# Jacobian of the answer-token CE w.r.t. this layer's weights (STE -> grad flows to W = grad of effective weight)
m.zero_grad(); logits = m(ids)[0]
ce = F.cross_entropy(logits[0, ans_pos - 1:ans_pos], ids[0, ans_pos:ans_pos + 1])
ce.backward()
g = lin.weight.grad.detach()                                  # ∂CE/∂W
s = lin.weight.abs().mean(1, keepdim=True).clamp(min=1e-8)
t = torch.clamp(torch.round(lin.weight / s), -1, 1)           # ternary trits
Wq = (s * t).detach()
dCE_flip = (-2 * s * t * g).detach()                          # ΔCE from flipping t->-t (loss-increase)
nz = (t != 0)                                                 # only +-1 weights can flip
flat_dce = dCE_flip.clone(); flat_dce[~nz] = -1e9             # ignore zeros
order = torch.argsort(flat_dce.flatten(), descending=True)    # most loss-increasing flips first
n_flippable = int(nz.sum())
print(f"layer {LAYER}: {lin.weight.numel()} weights, {n_flippable} are +-1 (flippable), rest are 0")
print(f"baseline answer-CE: {float(ce):.4f}")

@torch.no_grad()
def ce_after_flip(idx_flat):
    Wf = Wq.clone().flatten(); Wf[idx_flat] = -Wf[idx_flat]; Wf = Wf.view_as(Wq)
    lin.quantize = False; old = lin.weight.data; lin.weight.data = Wf
    lg = m(ids)[0]
    c = float(F.cross_entropy(lg[0, ans_pos - 1:ans_pos], ids[0, ans_pos:ans_pos + 1]))
    ok = int(lg[0, ans_pos - 1].argmax()) == ids[0, ans_pos].item()
    lin.weight.data = old; lin.quantize = True
    return c, ok

ks = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200, 400]
guided, rnd = [], []
rs = np.random.RandomState(0); flippable_idx = torch.nonzero(nz.flatten()).flatten().numpy()
gbreak = rbreak = None
for k in ks:
    gc, gok = ce_after_flip(order[:k]) if k else (float(ce), True)
    guided.append(gc)
    if gbreak is None and not gok: gbreak = k
    ridx = torch.tensor(rs.choice(flippable_idx, min(k, len(flippable_idx)), replace=False)) if k else torch.tensor([], dtype=torch.long)
    rc, rok = ce_after_flip(ridx) if k else (float(ce), True)
    rnd.append(rc)
    if rbreak is None and not rok: rbreak = k

print(f"\n{'#flips':>7}{'guided CE':>12}{'random CE':>12}")
for k, gg, rr in zip(ks, guided, rnd):
    print(f"{k:>7}{gg:>12.3f}{rr:>12.3f}")
print(f"\nanswer FLIPS to wrong at:  guided {gbreak} flips   vs   random {rbreak} flips")
print(f"(out of {n_flippable} flippable trits — the Jacobian finds the load-bearing few)")

plt.figure(figsize=(9, 5))
plt.plot(ks, guided, "o-", c="#d62728", label="Jacobian-guided flips")
plt.plot(ks, rnd, "s-", c="#888", label="random flips")
if gbreak: plt.axvline(gbreak, ls="--", c="#d62728", alpha=.6, label=f"guided breaks @ {gbreak}")
plt.xscale("symlog"); plt.xlabel("# ternary sign-flips in layer "+LAYER); plt.ylabel("answer-token CE")
plt.title(f"A few Jacobian-guided sign-flips break the answer; random flips barely dent it\n(correct 1.58-bit example, layer {LAYER})")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("runs_ternary/exp_flip.png", dpi=120)
print("saved runs_ternary/exp_flip.png")
