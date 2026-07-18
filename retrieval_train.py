"""Paired lockstep training on the retrieval curriculum, with per-SKILL x per-COMPLEXITY
loss tracking (the tabular matrix, logged over time), plus training loss and OOD
exact-match. Control (fp32) and target (VQ k=4) trained in lockstep on identical batches.
Live dashboard writes the loss matrix at each eval timestamp.
"""
import copy, json, math, os, time
import numpy as np
import torch, torch.nn.functional as F
import retrieval_data as D
from retrieval_model import MiniQwen, config_M

K = int(os.environ.get("K", 4))
STEPS = int(os.environ.get("STEPS", 20000))
BATCH = int(os.environ.get("BATCH", 48)); LR, WD = 3e-4, 0.01
EVAL_EVERY = 500; SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", "runs")
SKILL_ID = {"input": 0, "index": 1, "filter": 2, "semantic": 3, "relative": 4, "content": 5, "read": 6}
ID_SKILL = {v: k for k, v in SKILL_ID.items()}
SCORED = [1, 2, 3, 4, 5, 6]                              # skills that are target tokens


def make_ood(rng):
    """OOD = level-5 query types on LONGER lists (13-20) than trained (<=12)."""
    for _ in range(200):
        n = rng.randint(13, 21); els = D.make_list(rng, n)
        qt = ["T8", "T9", "T10"][rng.randint(3)]
        out = D.SOLVERS[qt](rng, els)
        if out:
            q, ans, steps = out
            return dict(list=els, qtype=qt, level=6, query=q, answer=ans, steps=steps)
    return None


def batchify(exs, tok, block):
    rows = []
    for ex in exs:
        toks, pl, sk = D._example_tokens(ex)
        ids = tok.enc(toks)[:block]
        sid = [SKILL_ID[s] for s in sk][:block]
        rows.append((ids, sid, ex["level"]))
    T = max(len(r[0]) for r in rows)
    X = np.full((len(rows), T), tok.pad, np.int64)
    S = np.zeros((len(rows), T), np.int64); lv = np.zeros(len(rows), np.int64)
    for i, (ids, sid, level) in enumerate(rows):
        X[i, :len(ids)] = ids; S[i, :len(sid)] = sid; lv[i] = level
    return torch.from_numpy(X), torch.from_numpy(S), torch.from_numpy(lv)


def split_loss(logits, X, Sk, lv):
    """Per-token CE, aggregated into overall, per-skill, and per-(level,skill) matrix."""
    V = logits.size(-1)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
    sk = Sk[:, 1:].reshape(-1); lvt = lv[:, None].expand_as(Sk)[:, 1:].reshape(-1)
    m = sk > 0
    total = (ce * m).sum() / m.sum().clamp_min(1)
    per_skill = {}
    for s in SCORED:
        ms = sk == s
        per_skill[ID_SKILL[s]] = float((ce * ms).sum() / ms.sum().clamp_min(1)) if ms.any() else float("nan")
    matrix = {}                                          # matrix[level][skill] = mean CE
    for L in range(1, 7):
        matrix[L] = {}
        for s in SCORED:
            mm = (sk == s) & (lvt == L)
            matrix[L][ID_SKILL[s]] = float((ce * mm).sum() / mm.sum().clamp_min(1)) if mm.any() else None
    return total, per_skill, matrix


@torch.no_grad()
def exact_match(model, exs, tok, block, device):
    model.eval(); ok = 0
    eos = tok.eos
    for ex in exs:
        toks, pl, _ = D._example_tokens(ex)
        ids = tok.enc(toks[:pl])                          # prompt only
        # teacher-force the trace, then check the final answer token(s) after 'A'
        full = tok.enc(toks)
        x = torch.tensor([full[:-1]], device=device)
        pred = model(x)[0].argmax(-1)[0]
        # answer position: token after 'A'
        a_pos = toks.index("A")
        gen = int(pred[a_pos])                            # predicts toks[a_pos+1] = answer
        ok += (gen == full[a_pos + 1])
    model.train(); return ok / len(exs)


def dash(hist, meta, matrix_o, matrix_c):
    def cell(mx, L, s):
        v = mx.get(L, {}).get(s)
        return f"{v:.2f}" if v is not None and v == v else "·"
    skills = ["index", "filter", "semantic", "relative", "content", "read"]
    def table(mx, title):
        rows = "".join(
            f"<tr><td>L{L}</td>" + "".join(f"<td>{cell(mx,L,s)}</td>" for s in skills) + "</tr>"
            for L in range(1, 7))
        return (f"<h4>{title}</h4><table style='border-collapse:collapse;font:11px monospace'>"
                f"<tr><th>lvl</th>{''.join(f'<th>{s[:4]}</th>' for s in skills)}</tr>{rows}</table>")
    def col(k): return [(h['step'], h[k]) for h in hist if k in h and h[k] == h[k]]
    def svg(series, ylog=True, ymin=None, ymax=None, w=460, h=150):
        xs=[p[0] for s in series for p in s['pts']] or [0]; ys=[p[1] for s in series for p in s['pts']] or [0]
        x0,x1=min(xs),max(xs)+1e-9; lo=ymin if ymin is not None else min(ys); hi=ymax if ymax is not None else max(ys)+1e-9
        tx=lambda x:44+(x-x0)/(x1-x0)*(w-54)
        def ty(y):
            if ylog: y,l2,h2=math.log10(max(y,1e-3)),math.log10(max(lo,1e-3)),math.log10(max(hi,1e-3)); return h-20-(y-l2)/(h2-l2+1e-9)*(h-32)
            return h-20-(y-lo)/(hi-lo+1e-9)*(h-32)
        o=[f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px'><rect x=44 y=8 width={w-54} height={h-30} fill=none stroke=#ccc/>"]
        for sname in series:
            o.append(f"<polyline points='{' '.join(f'{tx(x):.0f},{ty(y):.0f}' for x,y in sname['pts'])}' fill=none stroke='{sname['c']}' stroke-width=2/>")
        for i,sname in enumerate(series): o.append(f"<text x=52 y={20+i*13} fill='{sname['c']}' font-size=10>{sname['name']}</text>")
        return "".join(o)+f"<text x=2 y={h-20} font-size=9>{lo:.2g}</text><text x=2 y=12 font-size=9>{hi:.2g}</text></svg>"
    trainc = svg([{"name":"control","c":"#444","pts":col("tr_c")},{"name":"target(VQ)","c":"#2ca02c","pts":col("tr_o")}])
    oodc = svg([{"name":"control","c":"#444","pts":col("ood_c")},{"name":"target","c":"#2ca02c","pts":col("ood_o")}], ylog=False, ymin=0, ymax=1)
    html = f"""<meta http-equiv=refresh content=5><title>retrieval VQ</title>
<div style='font:13px -apple-system,sans-serif;max-width:1000px;margin:16px auto'>
<h2>Retrieval — per-skill × complexity loss · control(fp32) vs target(VQ k={K})</h2>
<p>step {meta['step']}/{STEPS} · {meta['sps']:.2f}s/st · {meta['elapsed']/60:.0f}min · curriculum up to L{meta['maxlv']}</p>
<div style='display:flex;gap:30px;flex-wrap:wrap'>
<div>{table(matrix_o,'TARGET (VQ) loss matrix')}</div>
<div>{table(matrix_c,'CONTROL (fp32) loss matrix')}</div>
</div>
<div style='display:flex;gap:30px;flex-wrap:wrap;margin-top:10px'>
<div><h4>train loss</h4>{trainc}</div>
<div><h4>OOD exact-match (longer lists)</h4>{oodc}</div>
</div>
<p style='color:#888;font-size:11px'>cells = mean CE per (level,skill); · = skill absent at that level · auto-refresh 5s</p></div>"""
    os.makedirs(OUT, exist_ok=True); open(os.path.join(OUT, "index.html"), "w").write(html)


def main():
    torch.set_num_threads(4); os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("building vocab...", flush=True)
    vocab = D.build_vocab(0, 3000); tok = D.Tokenizer(vocab)
    cfg = config_M()
    control = MiniQwen(cfg, len(vocab), K)
    target = copy.deepcopy(control); target.enable_quant()
    print(f"vocab {len(vocab)} · {sum(p.numel() for p in control.parameters())/1e6:.2f}M params", flush=True)
    optc = torch.optim.AdamW(control.parameters(), lr=LR, weight_decay=WD)
    opto = torch.optim.AdamW(target.parameters(), lr=LR, weight_decay=WD)
    rng = np.random.RandomState(SEED)
    ood_eval = [make_ood(np.random.RandomState(9000 + i)) for i in range(200)]
    ood_eval = [e for e in ood_eval if e]
    hist, t0 = [], time.time()

    for step in range(STEPS + 1):
        maxlv = min(5, 1 + int(step / STEPS / 0.6 * 5))       # curriculum ramp -> all 5 by 60%
        exs = [D.make_example(rng, rng.randint(1, maxlv + 1)) for _ in range(BATCH)]
        exs = [e for e in exs if e]
        X, Sk, lv = batchify(exs, tok, cfg["block"])
        # control
        optc.zero_grad(); lc, _, _ = split_loss(control(X)[0], X, Sk, lv); lc.backward(); optc.step()
        # target
        opto.zero_grad(); lo_logits, vq = target(X); lo, _, _ = split_loss(lo_logits, X, Sk, lv)
        (lo + vq).backward(); opto.step()

        if step % EVAL_EVERY == 0:
            ev = [D.make_example(np.random.RandomState(7000 + i), (i % 5) + 1) for i in range(160)]
            ev = [e for e in ev if e]
            Xe, Se, lve = batchify(ev, tok, cfg["block"])
            with torch.no_grad():
                tc, psc, mc = split_loss(control(Xe)[0], Xe, Se, lve)
                to, pso, mo = split_loss(target(Xe)[0], Xe, Se, lve)
            oc = exact_match(control, ood_eval[:120], tok, cfg["block"], "cpu")
            oo = exact_match(target, ood_eval[:120], tok, cfg["block"], "cpu")
            rec = {"step": step, "tr_c": lc.item(), "tr_o": lo.item(),
                   "ev_c": float(tc), "ev_o": float(to), "ood_c": oc, "ood_o": oo}
            hist.append(rec)
            sps = (time.time() - t0) / max(step, 1)
            dash(hist, dict(step=step, sps=sps, elapsed=time.time() - t0, maxlv=maxlv), mo, mc)
            with open(os.path.join(OUT, "metrics.jsonl"), "a") as f:
                f.write(json.dumps({**rec, "skill_control": psc, "skill_target": pso,
                                    "matrix_target": mo, "matrix_control": mc}) + "\n")
            print(f"step {step:5d} L{maxlv} | train c {lc.item():.3f} o {lo.item():.3f} "
                  f"| OOD-EM c {oc:.3f} o {oo:.3f}", flush=True)
    torch.save({"control": control.state_dict(), "target": target.state_dict(),
                "vocab": vocab, "cfg": cfg, "K": K}, os.path.join(OUT, "ckpt.pt"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
