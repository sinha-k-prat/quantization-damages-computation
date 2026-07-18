"""Watch runs/metrics.jsonl; regenerate curves.png whenever a new eval lands.
Also writes curves.html (meta-refresh 15s) so the plots page is self-updating.
"""
import json, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.environ.get("OUT", "runs")
MET = os.path.join(OUT, "metrics.jsonl")
skills = ["index", "filter", "semantic", "relative", "content", "read"]
col = dict(index="#1f77b4", filter="#ff7f0e", semantic="#2ca02c",
           relative="#d62728", content="#9467bd", read="#8c564b")


def render():
    R = [json.loads(l) for l in open(MET)]
    if not R:
        return 0
    step = [r["step"] for r in R]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Retrieval — fp32 control vs VQ k=4 (2-bit) · through step %d" % step[-1],
                 fontsize=13, fontweight="bold")

    a = ax[0, 0]
    a.plot(step, [r["tr_c"] for r in R], "o-", c="#444", label="control (fp32)")
    a.plot(step, [r["tr_o"] for r in R], "s-", c="#2ca02c", label="target (VQ)")
    a.set_yscale("log"); a.set_title("train CE (log)"); a.set_xlabel("step"); a.legend(); a.grid(alpha=.3)
    for x, lab in [(1200, "L2"), (2400, "L3"), (3600, "L4"), (4800, "L5")]:
        if step[-1] >= x - 200:
            a.axvline(x, ls=":", c="#bbb", lw=.8)
            a.text(x, a.get_ylim()[1], lab, fontsize=7, color="#999", va="top")

    a = ax[0, 1]
    a.plot(step, [r["ood_c"] for r in R], "o-", c="#444", label="control")
    a.plot(step, [r["ood_o"] for r in R], "s-", c="#2ca02c", label="target")
    a.set_ylim(0, 1.03); a.set_title("OOD exact-match (unseen longer lists)")
    a.set_xlabel("step"); a.legend(); a.grid(alpha=.3)

    a = ax[1, 0]
    for s in skills:
        y = [r["skill_target"].get(s) if (r["skill_target"].get(s) not in (None,)
             and r["skill_target"].get(s) == r["skill_target"].get(s)) else None for r in R]
        a.plot(step, y, "-", c=col[s], label=s)
    a.set_yscale("log"); a.set_title("per-skill eval CE — TARGET (learning order)")
    a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=.3)

    a = ax[1, 1]
    for s in skills:
        y = []
        for r in R:
            t = r["skill_target"].get(s); c = r["skill_control"].get(s)
            y.append(t - c if (t == t and c == c and t is not None and c is not None) else None)
        a.plot(step, y, "-", c=col[s], label=s)
    a.axhline(0, c="#000", lw=.8)
    a.set_title("quant gap per skill (target − control CE)  >0 = quant hurts")
    a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(OUT, "curves.png"), dpi=110)
    plt.close(fig)
    return step[-1]


HTML = """<meta http-equiv=refresh content=15><title>retrieval curves</title>
<body style='background:#fff;font:13px -apple-system,sans-serif;text-align:center'>
<h3>Retrieval curves — through step {step} (auto-refresh 15s)</h3>
<img src='curves.png?t={t}' style='max-width:100%;width:1100px'>
</body>"""


def main():
    open(os.path.join(OUT, "curves.html"), "w").write(HTML.format(step="…", t=0))
    last_n = -1
    while True:
        try:
            n = sum(1 for _ in open(MET)) if os.path.exists(MET) else 0
        except Exception:
            n = last_n
        if n != last_n and n > 0:
            s = render()
            open(os.path.join(OUT, "curves.html"), "w").write(HTML.format(step=s, t=n))
            print("re-plotted through step", s, flush=True)
            last_n = n
        time.sleep(10)


if __name__ == "__main__":
    main()
