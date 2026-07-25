"""Leak-proof minimum-symbolic-edit (user's redesign): block gradients to embeddings AND every
other continuous channel, so learning can ONLY go into the trits.
PHASE 1 (vocabulary): train ONLY the new-token embedding rows; everything else frozen. (~0.43 plateau)
PHASE 2 (circuit):    freeze ALL continuous params (embeddings, biases, RMSNorm weights, and the
                      ternary row-scales via freeze_scale) — only quant-layer weights train, and the
                      forward depends on them only through their trits. Edit penalty as before.
The Phase-2 flip count is the true symbolic cost of the skill beyond vocabulary.
Env: LAM, STEPS2 (700), PHASE1_ONLY=1 to just build the shared phase-1 checkpoint.
"""
import json, os, numpy as np, torch, torch.nn.functional as F
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen, QuantLinear, RMSNorm

torch.set_num_threads(int(os.environ.get("THREADS", 2)))
LAM = float(os.environ.get("LAM", 3.0)); TAU = float(os.environ.get("TAU", 0.2))
S1, S2 = int(os.environ.get("STEPS1", 500)), int(os.environ.get("STEPS2", 700))
BATCH = 48; P1 = "runs_mse2/phase1.pt"
OUT = os.environ.get("OUT", f"runs_mse2/lam_{LAM:g}"); os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

ck = torch.load("runs_ternary/ckpt.pt", map_location="cpu"); cfg, K = ck["cfg"], ck["K"]
vneg = N.build_vocab_neg(ck["vocab"]); tok = D.Tokenizer(vneg); n_old = len(ck["vocab"])
m = MiniQwen(cfg, len(vneg), K); m.enable_ternary()
sd = ck["target"]; nsd = m.state_dict()
with torch.no_grad():
    for k, v in sd.items():
        if k in nsd and nsd[k].shape == v.shape: nsd[k].copy_(v)
    nsd["embed.weight"][:n_old].copy_(sd["embed.weight"])
m.load_state_dict(nsd)

def batch(rng, n):
    exs = [N.make_neg_example(rng) if rng.rand() < 0.5 else D.make_example(rng, rng.randint(1, 6)) for _ in range(n)]
    exs = [e for e in exs if e]; rows = []
    for e in exs:
        toks, pl, sk = D._example_tokens(e); ids = tok.enc(toks)[:cfg["block"]]
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
def acc_on(gen, seed0, nn=70):
    m.eval(); ok = cnt = 0
    for i in range(nn):
        e = gen(np.random.RandomState(seed0+i))
        if not e: continue
        toks, pl, sk = D._example_tokens(e); a = toks.index("A"); ids = torch.tensor([tok.enc(toks)])
        ok += int(m(ids)[0][0, a].argmax()) == ids[0, a+1].item(); cnt += 1
    m.train(); return ok/max(cnt, 1)
sign_acc = lambda: acc_on(lambda r: N.make_neg_example(r), 5000)
orig_acc = lambda: acc_on(lambda r: D.make_example(r, r.randint(1, 6)), 8600)

# ---------- PHASE 1: vocabulary only ----------
if not os.path.exists(P1):
    print("PHASE 1: training ONLY new-token embedding rows ...", flush=True)
    for p in m.parameters(): p.requires_grad = False
    m.embed.weight.requires_grad = True
    mask = torch.zeros_like(m.embed.weight); mask[n_old:] = 1.0
    m.embed.weight.register_hook(lambda g: g * mask)              # zero grads on old rows
    opt = torch.optim.AdamW([m.embed.weight], lr=3e-3)
    rng = np.random.RandomState(0)
    for step in range(S1):
        X, Mk = batch(rng, BATCH); opt.zero_grad()
        lg, _ = m(X); masked_ce(lg, X, Mk).backward(); opt.step()
        if step % 100 == 0: print(f"  p1 step {step}: sign {sign_acc():.3f}", flush=True)
    torch.save(m.state_dict(), P1)
    print(f"PHASE 1 done: sign {sign_acc():.3f} orig {orig_acc():.3f}  saved {P1}", flush=True)
    if os.environ.get("PHASE1_ONLY"): raise SystemExit
else:
    m.load_state_dict(torch.load(P1, map_location="cpu"))
    print(f"loaded shared phase-1 ckpt: sign {sign_acc():.3f} orig {orig_acc():.3f}", flush=True)

# ---------- PHASE 2: symbols only ----------
for p in m.parameters(): p.requires_grad = False
for L in m.quant_layers():
    L.weight.requires_grad = True                                  # ONLY quant weights train
    if L.bias is not None: L.bias.requires_grad = False
    L.freeze_scale()                                               # forward depends on trits only
print(f"PHASE 2: frozen everything continuous; trainable tensors = "
      f"{sum(p.requires_grad for p in m.parameters())} (the 42 weight matrices)", flush=True)

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
        print(f"p2 step {step:4d}  sign {sa:.3f}  orig {oa:.3f}  flips {fp:.3f}%  task {float(task):.3f}", flush=True)
mf.close()
final = dict(lam=LAM, sign_acc=sa, orig_acc=oa, flip_pct=round(fp, 3), steps=S2)
json.dump(final, open(os.path.join(OUT, "final.json"), "w"))
torch.save({"model": m.state_dict(), "vocab": vneg, "cfg": cfg, "K": K}, os.path.join(OUT, "ckpt.pt"))
print("FINAL", json.dumps(final), flush=True)
