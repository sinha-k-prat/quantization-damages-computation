"""FULLY-TERNARY variant (mse4): same one-token leak-proof experiment, but the BASE model was
trained with ternary embeddings (tied encoder+decoder vocab) and ternary biases as well — only
RMSNorm weights are continuous. Phase 1 trains the latent of the single '-' embedding row (its
FORWARD value is ternary via STE); phase 2 trains trits of the 42 matrices only.
PHASE 1: train ONLY the single '-' embedding row (256 params); everything else frozen.
         One shared vector cannot memorize per-number answers -> measures how far pure
         vocabulary gets on a COMPOSITIONAL skill.
PHASE 2: freeze ALL continuous params (embeddings, biases, norms, row-scales); train trits only,
         with the symbolic edit penalty. Flip count = true symbolic cost of the binding circuit.
Sign-acc requires BOTH answer tokens ('-' then MAG) correct.
Env: LAM, STEPS1(600), STEPS2(700), PHASE1_ONLY, OUT.
"""
import json, os, numpy as np, torch, torch.nn.functional as F
import retrieval_data as D, retrieval_data_sign as S
from retrieval_model import MiniQwen

torch.set_num_threads(int(os.environ.get("THREADS", 2)))
LAM = float(os.environ.get("LAM", 3.0)); TAU = 0.2
S1, S2 = int(os.environ.get("STEPS1", 600)), int(os.environ.get("STEPS2", 700))
BATCH = 48; P1 = os.environ.get("P1", "runs_mse4/phase1.pt")
OUT = os.environ.get("OUT", f"runs_mse4/lam_{LAM:g}"); os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

BASE = os.environ.get("BASE", "runs_ternary2/ckpt.pt")
ck = torch.load(BASE, map_location="cpu"); cfg, K = ck["cfg"], ck["K"]
vocab = S.build_vocab_sign(ck["vocab"]); tok = D.Tokenizer(vocab); n_old = len(ck["vocab"])
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary(); m.enable_ternary_embed()
sd = ck["target"]; nsd = m.state_dict()
with torch.no_grad():
    for k, v in sd.items():
        if k in nsd and nsd[k].shape == v.shape: nsd[k].copy_(v)
    nsd["embed.weight"][:n_old].copy_(sd["embed.weight"])
m.load_state_dict(nsd)

def toks_of(e):
    return S.sign_tokens(e) if e.get("qtype") == "Tsign" else D._example_tokens(e)[:2]

def batch(rng, n):
    exs = [S.make_sign_example(rng) if rng.rand() < 0.5 else D.make_example(rng, rng.randint(1, 6)) for _ in range(n)]
    exs = [e for e in exs if e]; rows = []
    for e in exs:
        toks, pl = toks_of(e); ids = tok.enc(toks)[:cfg["block"]]
        rows.append((ids, [0]*min(pl, len(ids)) + [1]*(len(ids)-min(pl, len(ids)))))
    T = max(len(r[0]) for r in rows)
    X = np.full((len(rows), T), tok.pad, np.int64); Mk = np.zeros((len(rows), T), np.int64)
    for i, (ids, mk) in enumerate(rows): X[i, :len(ids)] = ids; Mk[i, :len(mk)] = mk
    return torch.from_numpy(X), torch.from_numpy(Mk)

def masked_ce(lg, X, Mk):
    V = lg.size(-1)
    ce = F.cross_entropy(lg[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
    mm = Mk[:, 1:].reshape(-1).float(); return (ce*mm).sum()/mm.sum().clamp_min(1)

@torch.no_grad()
def sign_acc(nn=70):
    m.eval(); ok = cnt = 0
    for i in range(nn):
        e = S.make_sign_example(np.random.RandomState(5000+i))
        toks, pl = S.sign_tokens(e); a = toks.index("A")
        ids = torch.tensor([tok.enc(toks)]); lg = m(ids)[0][0]
        ok += (int(lg[a].argmax()) == ids[0, a+1].item() and int(lg[a+1].argmax()) == ids[0, a+2].item())
        cnt += 1
    m.train(); return ok/cnt

@torch.no_grad()
def orig_acc(nn=70):
    m.eval(); ok = cnt = 0
    for i in range(nn):
        e = D.make_example(np.random.RandomState(8600+i), (i % 5)+1)
        if not e: continue
        toks, pl, sk = D._example_tokens(e); a = toks.index("A"); ids = torch.tensor([tok.enc(toks)])
        ok += int(m(ids)[0][0, a].argmax()) == ids[0, a+1].item(); cnt += 1
    m.train(); return ok/cnt

# ---------- PHASE 1: only the '-' embedding row ----------
if not os.path.exists(P1):
    print("PHASE 1: training ONLY the single '-' embedding row ...", flush=True)
    for p in m.parameters(): p.requires_grad = False
    m.embed.weight.requires_grad = True
    mask = torch.zeros_like(m.embed.weight); mask[n_old:] = 1.0        # just the '-' row
    m.embed.weight.register_hook(lambda g: g * mask)
    opt = torch.optim.AdamW([m.embed.weight], lr=3e-3)
    rng = np.random.RandomState(0)
    for step in range(S1):
        X, Mk = batch(rng, BATCH); opt.zero_grad()
        lg, _ = m(X); masked_ce(lg, X, Mk).backward(); opt.step()
        if step % 100 == 0: print(f"  p1 {step}: sign {sign_acc():.3f}", flush=True)
    torch.save(m.state_dict(), P1)
    print(f"PHASE 1 done: sign {sign_acc():.3f} orig {orig_acc():.3f}", flush=True)
    with torch.no_grad():
        r = m.embed.weight[n_old]; sc = r.abs().mean().clamp(min=1e-8)
        t = torch.clamp(torch.round(r / sc), -1, 1)
        u, c = torch.unique(t, return_counts=True)
        print("  '-' row trits:", {int(a): int(b) for a, b in zip(u, c)}, flush=True)
    if os.environ.get("PHASE1_ONLY"): raise SystemExit
else:
    m.load_state_dict(torch.load(P1, map_location="cpu"))
    print(f"loaded phase-1: sign {sign_acc():.3f} orig {orig_acc():.3f}", flush=True)

# ---------- PHASE 2: trits only ----------
for p in m.parameters(): p.requires_grad = False
for L in m.quant_layers():
    L.weight.requires_grad = True; L.freeze_scale()
print("PHASE 2: everything continuous frozen; only the 42 weight matrices train (trits)", flush=True)
STATES = torch.tensor([-1., 0., 1.])
orig_trits = {L.tag: torch.clamp(torch.round(L.weight.data / L.s0), -1, 1).clone() for L in m.quant_layers()}
orig_idx = {t: (v + 1).long().unsqueeze(-1) for t, v in orig_trits.items()}
N_W = sum(L.weight.numel() for L in m.quant_layers())

def edit_loss():
    tot = 0.
    for L in m.quant_layers():
        z = -((L.weight / L.s0).unsqueeze(-1) - STATES).pow(2) / TAU
        tot = tot + (-F.log_softmax(z, -1).gather(-1, orig_idx[L.tag])).sum()
    return tot / N_W

@torch.no_grad()
def flip_pct():
    tf = 0
    for L in m.quant_layers():
        tf += int((torch.clamp(torch.round(L.weight / L.s0), -1, 1) != orig_trits[L.tag]).sum())
    return 100*tf/N_W

opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
rng = np.random.RandomState(1)
mf = open(os.path.join(OUT, "metrics.jsonl"), "w")
for step in range(S2+1):
    X, Mk = batch(rng, BATCH)
    opt.zero_grad(); lg, vq = m(X)
    task = masked_ce(lg, X, Mk); ed = edit_loss()
    (task + vq + LAM*ed).backward(); opt.step()
    if step % 50 == 0:
        sa, oa, fp = sign_acc(), orig_acc(), flip_pct()
        rec = dict(step=step, sign=round(sa,3), orig=round(oa,3), flip=round(fp,3),
                   edit=round(float(ed),4), task=round(float(task),3))
        mf.write(json.dumps(rec)+"\n"); mf.flush()
        print(f"p2 {step:4d}  sign {sa:.3f}  orig {oa:.3f}  flips {fp:.3f}%  task {float(task):.3f}", flush=True)
mf.close()
final = dict(lam=LAM, sign_acc=sa, orig_acc=oa, flip_pct=round(fp, 3), steps=S2)
json.dump(final, open(os.path.join(OUT, "final.json"), "w"))
torch.save({"model": m.state_dict(), "vocab": vocab, "cfg": cfg, "K": K}, os.path.join(OUT, "ckpt.pt"))
print("FINAL", json.dumps(final), flush=True)
