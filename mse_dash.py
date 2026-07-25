"""Live dashboard for the Minimum-Symbolic-Edit lambda sweep.
Aggregates runs_mse/lam_*/metrics.jsonl -> runs_mse/index.html (auto-refresh 5s).
"""
import glob, json, os, time

OUT = __import__("os").environ.get("MSE_OUT", "runs_mse"); BASE_FLIPS = 20.11   # unconstrained fine-tune baseline (100% acc @ 20.11% flips)

def read(run):
    try: return [json.loads(l) for l in open(os.path.join(run, "metrics.jsonl"))]
    except Exception: return []

def spark(rows, key, ymax, color, w=180, h=36):
    if not rows: return ""
    xs = [r["step"] for r in rows]; ys = [min(r[key], ymax)/ymax for r in rows]
    x1 = max(xs[-1], 1)
    pts = " ".join(f"{4+(x/x1)*(w-8):.0f},{h-4-y*(h-8):.0f}" for x, y in zip(xs, ys))
    return f"<svg width={w} height={h}><polyline points='{pts}' fill=none stroke='{color}' stroke-width=2/></svg>"

def scatter(runs, w=520, h=300):
    o = [f"<svg viewBox='0 0 {w} {h}' style='max-width:{w}px;width:100%'>",
         f"<rect x=45 y=10 width={w-60} height={h-45} fill=none stroke='#ccc'/>"]
    xmax = 25.0
    bx = 45 + (BASE_FLIPS/xmax)*(w-60)
    o.append(f"<line x1={bx:.0f} y1=10 x2={bx:.0f} y2={h-35} stroke='#d62728' stroke-dasharray='5,4'/>")
    o.append(f"<text x={bx-4:.0f} y=24 font-size=10 fill='#d62728' text-anchor='end'>unconstrained 20.1%</text>")
    for r in runs:
        if not r["rows"]: continue
        last = r["rows"][-1]
        x = 45 + (min(last["flip"], xmax)/xmax)*(w-60); y = (h-35) - last["sign"]*(h-45)
        o.append(f"<circle cx={x:.0f} cy={y:.0f} r=7 fill='#6B5CA5' stroke='#222'/>")
        o.append(f"<text x={x+9:.0f} y={y+4:.0f} font-size=11>λ={r['lam']}</text>")
    o.append(f"<text x=6 y=16 font-size=10>sign-acc 1.0</text><text x=6 y={h-38} font-size=10>0.0</text>")
    o.append(f"<text x={w//2} y={h-6} font-size=11 text-anchor='middle'>% of trits flipped (fewer = more surgical)</text></svg>")
    return "".join(o)

def render():
    runs = []
    for d in sorted(glob.glob(os.path.join(OUT, "lam_*"))):
        runs.append(dict(lam=d.split("lam_")[-1], rows=read(d),
                         done=os.path.exists(os.path.join(d, "final.json"))))
    rows_html = ""
    for r in runs:
        last = r["rows"][-1] if r["rows"] else dict(step=0, sign=0, orig=0, flip=0, edit=0)
        status = "✅ done" if r["done"] else ("▶ running" if r["rows"] else "⏳ queued")
        rows_html += (f"<tr><td><b>{r['lam']}</b></td><td>{status}</td><td>{last['step']}</td>"
                      f"<td>{last['sign']:.3f}</td><td>{last['orig']:.3f}</td><td>{last['flip']:.2f}%</td>"
                      f"<td>{spark(r['rows'],'sign',1.0,'#3E7CB1')}</td>"
                      f"<td>{spark(r['rows'],'flip',25.0,'#6B5CA5')}</td></tr>")
    html = f"""<meta http-equiv=refresh content=5><meta charset=utf-8><title>Minimum Symbolic Edit — live</title>
<div style='font:14px -apple-system,sans-serif;max-width:960px;margin:18px auto'>
<h2>Minimum Symbolic Edit training — λ<sub>edit</sub> sweep (live)</h2>
<p>L = task + quant + λ<sub>e</sub>·KL(orig-symbol ∥ p) &nbsp;·&nbsp; every symbol change costs 1 &nbsp;·&nbsp;
reference: <b style='color:#d62728'>prior unconstrained run = 100% acc @ 20.11% flips</b> (inflated by a since-fixed vocab bug — the clean baseline is the λ=0 row below)</p>
<table border=1 cellspacing=0 cellpadding=6 style='border-collapse:collapse;font-size:13px'>
<tr style='background:#f3efe6'><th>λ_edit</th><th>status</th><th>step</th><th>sign-acc</th><th>orig-acc</th><th>% flips</th><th>sign-acc curve</th><th>flip% curve</th></tr>
{rows_html}</table>
<h3>accuracy vs edits (goal: top-left)</h3>
{scatter(runs)}
<p style='color:#888;font-size:12px'>auto-refresh 5s · metrics every 50 steps · runs execute 2-at-a-time (CPU)</p></div>"""
    open(os.path.join(OUT, "index.html"), "w").write(html)

os.makedirs(OUT, exist_ok=True)
while True:
    try: render()
    except Exception as e: print("render err", e, flush=True)
    time.sleep(5)
