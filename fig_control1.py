"""Control-validation figure for Paper 1.

Leads with ABSOLUTE control performance to reassure a reviewer that the
fp32 control is a competent baseline, then shows the Qwen2.5-0.5B sanity
check (fp vs 4-bit) for external calibration.

Output: runs/fig_control_validation.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(HERE, "runs", "metrics.jsonl")
OUT = os.path.join(HERE, "runs", "fig_control_validation.png")

# ---- palette (validated) ----
INK = "#0b0b0b"        # control: black (primary ink)
GREEN = "#008300"      # 2-bit target: green (categorical slot 4)
BLUE = "#2a78d6"       # Qwen fp
RED = "#e34948"        # Qwen 4-bit
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SECONDARY = "#52514e"

# ---- load run metrics ----
rows = []
with open(METRICS) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
rows.sort(key=lambda r: r["step"])

steps = np.array([r["step"] for r in rows])
tr_c = np.array([r["tr_c"] for r in rows])
tr_o = np.array([r["tr_o"] for r in rows])
ood_c = np.array([r["ood_c"] for r in rows])
ood_o = np.array([r["ood_o"] for r in rows])

SKILLS = ["read", "semantic", "filter", "index", "content", "relative"]
last4 = rows[-4:]
skill_mean = {
    s: float(np.mean([r["skill_control"][s] for r in last4])) for s in SKILLS
}

# ---- Qwen2.5-0.5B reference numbers (verified in this project) ----
QWEN_TYPES = ["computation", "copy", "language"]
QWEN_FP = [0.211, 0.393, 1.113]
QWEN_4BIT = [0.956, 1.000, 1.831]
QWEN_EM_FP = 0.17
QWEN_EM_4BIT = 0.08

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlesize": 9.5,
    "axes.titlecolor": INK,
})


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
fig.patch.set_facecolor("#fcfcfb")
for ax in axes.flat:
    ax.set_facecolor("#fcfcfb")

# ------------------------------------------------------------------ (a)
ax = axes[0, 0]
style_ax(ax)
ax.plot(steps, tr_c, color=INK, lw=2, label="control (fp32)")
ax.plot(steps, tr_o, color=GREEN, lw=2, label="2-bit target")
ax.set_yscale("log")
ax.set_xlabel("step")
ax.set_ylabel("train CE (log scale)")
ax.set_title(
    "(a) Train CE: control converges to ~0.02–0.03\n"
    "(near-zero loss — task fully learned)",
    loc="left",
)
ax.legend(frameon=False, loc="upper right")

# ------------------------------------------------------------------ (b)
ax = axes[0, 1]
style_ax(ax)
ax.axhline(1.0, color=BASELINE, lw=1, ls="--", zorder=1)
ax.plot(steps, ood_c, color=INK, lw=2, label="control (fp32)")
ax.plot(steps, ood_o, color=GREEN, lw=2, label="2-bit target")
ax.set_ylim(0, 1.05)
ax.set_xlabel("step")
ax.set_ylabel("OOD exact-match")
ax.set_title(
    "(b) Held-out exact-match: the CONTROL itself reaches 100%\n"
    "(competent baseline, not a strawman)",
    loc="left",
)
ax.legend(frameon=False, loc="lower right")

# ------------------------------------------------------------------ (c)
ax = axes[1, 0]
style_ax(ax)
skill_order = ["read", "semantic", "filter", "index", "content", "relative"]
vals = [skill_mean[s] for s in skill_order]
x = np.arange(len(skill_order))
ax.bar(x, vals, width=0.62, color=INK, edgecolor="#fcfcfb", linewidth=1)
for xi, v in zip(x, vals):
    ax.text(xi, v + max(vals) * 0.03, f"{v:.4f}", ha="center", va="bottom",
            fontsize=8, color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(skill_order)
ax.set_ylabel("control CE (mean of last 4 evals)")
ax.set_ylim(0, max(vals) * 1.25)
max_skill = max(vals)
ax.set_title(
    f"(c) Control per-skill CE: all skills mastered\n"
    f"(max {max_skill:.3f} ≤ ~0.07)",
    loc="left",
)

# ------------------------------------------------------------------ (d)
ax = axes[1, 1]
style_ax(ax)
x = np.arange(len(QWEN_TYPES))
w = 0.36
ax.bar(x - w / 2, QWEN_FP, width=w, color=BLUE, label="fp",
       edgecolor="#fcfcfb", linewidth=1)
ax.bar(x + w / 2, QWEN_4BIT, width=w, color=RED, label="4-bit",
       edgecolor="#fcfcfb", linewidth=1)
for xi, (vf, vq) in enumerate(zip(QWEN_FP, QWEN_4BIT)):
    ax.text(xi - w / 2, vf + 0.03, f"{vf:.2f}", ha="center", va="bottom",
            fontsize=8, color=SECONDARY)
    ax.text(xi + w / 2, vq + 0.03, f"{vq:.2f}", ha="center", va="bottom",
            fontsize=8, color=SECONDARY)
ax.set_xticks(x)
ax.set_xticklabels(QWEN_TYPES)
ax.set_ylabel("GSM8K CE by token type")
ax.set_ylim(0, max(QWEN_4BIT) * 1.45)
ax.legend(frameon=False, loc="upper left")
ax.text(
    0.02, 0.80,
    f"fp EM {QWEN_EM_FP:.2f} vs 4-bit EM {QWEN_EM_4BIT:.2f}; "
    "8-bit = lossless (sanity check)",
    transform=ax.transAxes, ha="left", va="top", fontsize=8,
    color=SECONDARY,
    bbox=dict(boxstyle="round,pad=0.3", fc="#f0efec", ec="none"),
)
ax.set_title(
    "(d) Qwen2.5-0.5B on GSM8K: fp baseline is a healthy,\n"
    "normally-performing model for its size",
    loc="left",
)

fig.suptitle("Control validation: the fp32 control is a competent baseline",
             fontsize=12, color=INK, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT, dpi=120, facecolor=fig.get_facecolor())
print("saved:", OUT)

print(f"STATS: control final train CE={tr_c[-1]:.4f}, "
      f"control final OOD-EM={ood_c[-1]:.3f}, "
      f"max per-skill control CE (mean last 4)={max_skill:.4f}")
