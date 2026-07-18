"""H1 — bit-width elasticity map: is a component's tolerance-to-coarsening a functional
fingerprint (discrete/lookup = robust, analog/computation = fragile)?
For each of 42 components, crush ONLY it to k=2 (1-bit), keep the rest at trained k=4,
measure per-skill CE increase (elasticity). Validate: do the crush-sensitive components
line up with the computation skills (relative/content) & value path we found independently?
Confound check: is elasticity just weight magnitude?
"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen
from retrieval_train import batchify, split_loss

SKILLS = ["index", "filter", "semantic", "relative", "content", "read"]
COMPUTE = ["relative", "content"]; LOOKUP = ["semantic", "filter", "read"]
ck = torch.load("runs/ckpt.pt", map_location="cpu")
vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_quant(); m.load_state_dict(ck["target"]); m.eval()

ev = [D.make_example(np.random.RandomState(7000 + i), (i % 5) + 1) for i in range(600)]
ev = [e for e in ev if e]
Xe, Se, lve = batchify(ev, tok, cfg["block"])

def codebook_k(w, kk):
    qs = torch.linspace(0, 1, kk, device=w.device)
    return torch.quantile(w.float(), qs, dim=1).t().contiguous().to(w.dtype)

@torch.no_grad()
def per_skill():
    _, ps, _ = split_loss(m(Xe)[0], Xe, Se, lve)
    return ps

base = per_skill()
comps = m.quant_layers()
rows = []
for i, c in enumerate(comps):
    old = c.codebook.data.clone()
    c.codebook.data = codebook_k(c.weight, 2)          # crush THIS one to 1-bit
    ps = per_skill()
    c.codebook.data = old                              # restore
    el = {s: ps[s] - base[s] for s in SKILLS}
    comp_el = np.mean([el[s] for s in COMPUTE]); look_el = np.mean([el[s] for s in LOOKUP])
    wnorm = float(c.weight.std())
    rows.append(dict(tag=c.tag, cls=c.tag.split(".")[1], comp=comp_el, look=look_el,
                     total=np.mean([el[s] for s in SKILLS]), wnorm=wnorm, **el))
    print(f"[{i+1:2d}/42] crush {c.tag:<9} compute+{comp_el:+.4f}  lookup+{look_el:+.4f}", flush=True)

# --- analysis ---
print("\n=== class-average elasticity (CE rise when crushed to 1-bit) ===")
print(f"{'class':<6}{'compute skills':>16}{'lookup skills':>15}{'ratio c/l':>11}")
for cl in ["q", "k", "v", "o", "gate", "up", "down"]:
    r = [x for x in rows if x["cls"] == cl]
    c = np.mean([x["comp"] for x in r]); l = np.mean([x["look"] for x in r])
    print(f"{cl:<6}{c:>16.4f}{l:>15.4f}{c/max(l,1e-6):>11.1f}")

comp_all = np.array([x["comp"] for x in rows]); look_all = np.array([x["look"] for x in rows])
above = int((comp_all > look_all).sum())
print(f"\ncomponents where crushing hurts COMPUTE > LOOKUP: {above}/42")
# confound: elasticity vs weight magnitude
wn = np.array([x["wnorm"] for x in rows]); tot = np.array([x["total"] for x in rows])
r_mag = np.corrcoef(wn, tot)[0, 1]
print(f"corr(elasticity, weight-magnitude) = {r_mag:+.2f}  (low => elasticity is NOT just magnitude)")

# --- figures ---
CLSCOL = {"q": "#3E7CB1", "k": "#3E7CB1", "v": "#ff7f0e", "o": "#ff7f0e",
          "gate": "#2ca02c", "up": "#2ca02c", "down": "#2ca02c"}
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
# (A) scatter: compute vs lookup elasticity per component
for x in rows:
    ax[0].scatter(x["look"], x["comp"], c=CLSCOL[x["cls"]], s=45, edgecolor="#222", zorder=3)
lim = max(comp_all.max(), look_all.max()) * 1.1
ax[0].plot([0, lim], [0, lim], "--", color="#888", label="equal")
ax[0].set_xlabel("elasticity on LOOKUP skills (CE rise when crushed)")
ax[0].set_ylabel("elasticity on COMPUTE skills")
ax[0].set_title(f"Crushing hurts computation more than lookup\n({above}/42 components above the line)")
from matplotlib.lines import Line2D
ax[0].legend(handles=[Line2D([0],[0],marker='o',color='w',markerfacecolor=c,label=l,markersize=9)
    for l,c in [("Q/K (routing)","#3E7CB1"),("V/O (payload)","#ff7f0e"),("MLP (compute)","#2ca02c")]]+
    [Line2D([0],[0],ls='--',color='#888',label='equal')], fontsize=9)
ax[0].grid(alpha=.3)
# (B) per-component total elasticity ranked, colored by class
order = sorted(rows, key=lambda x: x["comp"], reverse=True)
ax[1].barh(range(len(order)), [x["comp"] for x in order],
           color=[CLSCOL[x["cls"]] for x in order], edgecolor="#222")
ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels([x["tag"] for x in order], fontsize=7)
ax[1].invert_yaxis(); ax[1].set_xlabel("compute-skill elasticity (CE rise, crushed to 1-bit)")
ax[1].set_title("Which components must stay precise for COMPUTATION")
plt.tight_layout(); plt.savefig("runs/exp_h1.png", dpi=115)
print("\nsaved runs/exp_h1.png")
