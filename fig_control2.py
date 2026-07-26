"""Control-validation figure for Paper 2.

(a) substrate 1 (fp-embedding ternary): train CE, fp control vs ternary target
(b) substrate 2 (fully ternary incl vocab): same
(c) OOD exact-match vs step, both substrates (4 curves)
(d) old-skill retention (orig_acc) across the 6 lambda-sweep installation runs
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# -- palette (fixed assignment: control=black, target=purple; substrate 2 gets
#    lighter steps + dashed linestyle as secondary encoding) ------------------
C_CONTROL = "#1a1a1a"
C_TARGET = "#7a4bd6"
C_CONTROL2 = "#8a8a8a"
C_TARGET2 = "#b491ea"
INK = "#333333"
MUTED = "#666666"
GRID = dict(color="#dddddd", linewidth=0.7)


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["step"])
    return rows


def series(rows, key):
    return [r["step"] for r in rows], [r[key] for r in rows]


sub1 = load_jsonl(os.path.join(HERE, "runs_ternary", "metrics.jsonl"))
sub2 = load_jsonl(os.path.join(HERE, "runs_ternary2", "metrics.jsonl"))

sweep = []  # (label, orig_acc)
for d in ("runs_mse3", "runs_mse4"):
    for lam in (0, 3, 30):
        with open(os.path.join(HERE, d, f"lam_{lam}", "final.json")) as f:
            j = json.load(f)
        tag = "mse3" if d == "runs_mse3" else "mse4"
        sweep.append((f"{tag}\n$\\lambda$={lam}", j["orig_acc"]))

fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))
fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.07, right=0.98,
                    top=0.885, bottom=0.08)
(ax_a, ax_b), (ax_c, ax_d) = axes

for ax in (ax_a, ax_b, ax_c, ax_d):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb")
    ax.grid(True, axis="y", **GRID)
    ax.set_axisbelow(True)

# -- (a) substrate 1 train CE -------------------------------------------------
s, y = series(sub1, "tr_c")
ax_a.semilogy(s, y, color=C_CONTROL, lw=2, label="fp control")
s, y = series(sub1, "tr_o")
ax_a.semilogy(s, y, color=C_TARGET, lw=2, label="ternary target")
ax_a.set_title("(a) Substrate 1 (fp-embedding ternary): train CE\n"
               "ternary matches a competent fp control — not a weak one",
               fontsize=10, color=INK, loc="left")
ax_a.set_xlabel("step", fontsize=9, color=MUTED)
ax_a.set_ylabel("train CE (log)", fontsize=9, color=MUTED)
ax_a.legend(frameon=False, fontsize=9)

# -- (b) substrate 2 train CE -------------------------------------------------
s, y = series(sub2, "tr_c")
ax_b.semilogy(s, y, color=C_CONTROL, lw=2, label="fp control")
s, y = series(sub2, "tr_o")
ax_b.semilogy(s, y, color=C_TARGET, lw=2, label="ternary target")
ax_b.set_title("(b) Substrate 2 (fully ternary, incl. vocab): train CE\n"
               "parity with the fp control again",
               fontsize=10, color=INK, loc="left")
ax_b.set_xlabel("step", fontsize=9, color=MUTED)
ax_b.set_ylabel("train CE (log)", fontsize=9, color=MUTED)
ax_b.legend(frameon=False, fontsize=9)

# -- (c) OOD exact match, both substrates ------------------------------------
s, y = series(sub1, "ood_c")
ax_c.plot(s, y, color=C_CONTROL, lw=2, label="substrate 1: fp control")
s, y = series(sub1, "ood_o")
ax_c.plot(s, y, color=C_TARGET, lw=2, label="substrate 1: ternary target")
s, y = series(sub2, "ood_c")
ax_c.plot(s, y, color=C_CONTROL2, lw=2, ls="--",
          label="substrate 2: fp control")
s, y = series(sub2, "ood_o")
ax_c.plot(s, y, color=C_TARGET2, lw=2, ls="--",
          label="substrate 2: ternary target")
ax_c.set_ylim(0, 1.05)
ax_c.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
ax_c.set_title("(c) OOD exact-match: controls and targets all reach\n"
               "perfect held-out generalization",
               fontsize=10, color=INK, loc="left")
ax_c.set_xlabel("step", fontsize=9, color=MUTED)
ax_c.set_ylabel("OOD exact-match", fontsize=9, color=MUTED)
ax_c.legend(frameon=False, fontsize=8, loc="lower right")

# -- (d) old-skill retention --------------------------------------------------
labels = [t for t, _ in sweep]
vals = [v for _, v in sweep]
x = range(len(vals))
ax_d.bar(x, vals, width=0.62, color=C_TARGET, edgecolor="none")
ax_d.axhline(1.0, color=C_CONTROL, lw=1.0, ls="--")
for i, v in enumerate(vals):
    ax_d.text(i, v + 0.0015, f"{v:.3f}", ha="center", va="bottom",
              fontsize=8.5, color=INK)
ax_d.set_xticks(list(x))
ax_d.set_xticklabels(labels, fontsize=8.5)
ax_d.set_ylim(0.9, 1.01)
ax_d.set_title("(d) Old-skill retention during skill installation:\n"
               "every run preserves existing capabilities (orig-acc $\\approx$ 1.0)\n"
               "— flips were not bought by sacrificing the base model",
               fontsize=10, color=INK, loc="left")
ax_d.set_ylabel("orig-acc after installation", fontsize=9, color=MUTED)

fig.suptitle("Control validation: competent controls, perfect OOD, no base-model damage",
             fontsize=12, color=INK, x=0.07, y=0.975, ha="left")

out = os.path.join(HERE, "runs_ternary", "fig_control_validation2.png")
fig.savefig(out, dpi=120)
print("saved", out)

# stats for the caller
print("min orig_acc:", min(vals), "max orig_acc:", max(vals))
print("final OOD-EM control sub1:", sub1[-1]["ood_c"],
      "sub2:", sub2[-1]["ood_c"])
