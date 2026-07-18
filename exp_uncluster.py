"""Exp 3a + 2a — un-cluster sweeps (the STE-alive setup's unfair advantage).
3a: leave-one-out un-quantize each of 42 components, measure per-skill CE recovery
    -> component x skill recovery matrix (where does each skill live?).
2a: un-quantize by WEIGHT CLASS (Q/K vs V/O vs MLP), per-skill recovery
    -> double dissociation (relative<-QK geometry, content<-MLP arithmetic?).
recovery[s] = CE_target_allquant[s] - CE_with_uncluster[s]   (>0 = quant here hurt skill s)
"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import retrieval_data as D
from retrieval_model import MiniQwen
from retrieval_train import batchify, split_loss, SKILL_ID

SKILLS = ["index", "filter", "semantic", "relative", "content", "read"]
ck = torch.load("runs/ckpt.pt", map_location="cpu")
vocab, cfg, K = ck["vocab"], ck["cfg"], ck["K"]
tok = D.Tokenizer(vocab)
target = MiniQwen(cfg, len(vocab), K); target.enable_quant(); target.load_state_dict(ck["target"]); target.eval()

# fixed eval set spanning all levels/skills (same recipe as trainer eval, bigger)
ev = [D.make_example(np.random.RandomState(7000 + i), (i % 5) + 1) for i in range(600)]
ev = [e for e in ev if e]
Xe, Se, lve = batchify(ev, tok, cfg["block"])

@torch.no_grad()
def per_skill(pred):
    target.set_quant(pred)
    _, ps, _ = split_loss(target(Xe)[0], Xe, Se, lve)
    return ps

base = per_skill(lambda t: True)                 # all quantized (target as trained)
tags = [m.tag for m in target.quant_layers()]    # 42 component tags
print("baseline (all-quant) per-skill CE:", {k: round(v, 4) for k, v in base.items()})

# ---- 3a: leave-one-out ----
rec = {}                                          # rec[tag][skill] = recovery
for i, tg in enumerate(tags):
    ps = per_skill(lambda t, tg=tg: t != tg)      # un-cluster ONLY this component
    rec[tg] = {s: base[s] - ps[s] for s in SKILLS}
    print(f"[{i+1:2d}/42] un-cluster {tg:<9} | relative {rec[tg]['relative']:+.4f}  content {rec[tg]['content']:+.4f}", flush=True)

# heatmap component x skill
M = np.array([[rec[tg][s] for s in SKILLS] for tg in tags])
fig, ax = plt.subplots(figsize=(7, 12))
vmax = np.abs(M).max()
im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(SKILLS))); ax.set_xticklabels(SKILLS, rotation=45, ha="right")
ax.set_yticks(range(len(tags))); ax.set_yticklabels(tags, fontsize=7)
ax.set_title("Exp 3a — leave-one-out un-cluster\nper-skill CE recovery (red = this component's\nquantization hurt that skill)", fontsize=10)
fig.colorbar(im, ax=ax, shrink=.5, label="recovery (CE_base − CE_uncluster)")
plt.tight_layout(); plt.savefig("runs/exp3a.png", dpi=110)

print("\n=== 3a: top components whose un-clustering RECOVERS each fragile skill ===")
for s in ["relative", "content", "index"]:
    top = sorted(tags, key=lambda tg: rec[tg][s], reverse=True)[:5]
    print(f"  {s:<9}:", ", ".join(f"{tg}({rec[tg][s]:+.3f})" for tg in top))

# ---- 2a: weight-class ablation ----
CLASSES = {"Q/K (routing)": ("q", "k"), "V/O (payload)": ("v", "o"),
           "MLP (compute)": ("gate", "up", "down")}
print("\n=== 2a: un-cluster by WEIGHT CLASS — per-skill recovery ===")
print(f"{'class':<16}" + "".join(f"{s[:5]:>9}" for s in SKILLS))
crec = {}
for cname, cls in CLASSES.items():
    ps = per_skill(lambda t, cls=cls: not t.split(".")[1] in cls)   # un-cluster this class only
    crec[cname] = {s: base[s] - ps[s] for s in SKILLS}
    print(f"{cname:<16}" + "".join(f"{crec[cname][s]:>+9.4f}" for s in SKILLS))

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(SKILLS)); w = 0.26
for i, (cname, _) in enumerate(CLASSES.items()):
    ax.bar(x + (i - 1) * w, [crec[cname][s] for s in SKILLS], w, label=cname)
ax.axhline(0, color="#000", lw=.8); ax.set_xticks(x); ax.set_xticklabels(SKILLS)
ax.set_ylabel("per-skill CE recovery when this class is un-clustered")
ax.set_title("Exp 2a — which weight class carries each skill's quant penalty?\n(tall bar = that class's quantization is what hurt that skill)")
ax.legend(); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("runs/exp2a.png", dpi=120)
print("\nsaved runs/exp3a.png, runs/exp2a.png")
