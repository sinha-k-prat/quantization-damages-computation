"""Minimum Symbolic Edit training (user spec): learn the 'sign' skill from the ternary ckpt with
  L = L_task + L_quant + lam_e * L_edit
L_task  = masked CE on traces (50% sign-task, 50% original tasks)
L_quant = commitment |w - sg(c)|^2 (built into the ternary QuantLinear, beta=0.25)
L_edit  = CE(q_orig, p) with p = softmax(-(w/s - t)^2 / tau) over states t in {-1,0,+1}
          -> prices every SYMBOL change equally (unlike L1 on weights, where -1->+1 costs 2x).
Env: LAM (edit weight), TAU (0.2), STEPS (700), THREADS (2), OUT dir.
Writes OUT/metrics.jsonl every 50 steps (live dashboard) and OUT/final.json at the end.
"""
import copy, json, os, numpy as np, torch, torch.nn.functional as F
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen

torch.set_num_threads(int(os.environ.get("THREADS", 2)))
LAM = float(os.environ.get("LAM", 3.0)); TAU = float(os.environ.get("TAU", 0.2))
STEPS = int(os.environ.get("STEPS", 700)); BATCH = int(os.environ.get("BATCH", 48))
OUT = os.environ.get("OUT", f"runs_mse/lam_{LAM:g}")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

ck = torch.load("runs_ternary/ckpt.pt", map_location="cpu"); cfg, K = ck["cfg"], ck["K"]
vneg = N.build_vocab_neg(ck["vocab"]); tok = D.Tokenizer(vneg)
m = MiniQwen(cfg, len(vneg), K); m.enable_ternary()
sd = ck["target"]; nsd = m.state_dict()
with torch.no_grad():
    for k, v in sd.items():
        if k in nsd and nsd[k].shape == v.shape: nsd[k].copy_(v)
    nsd["embed.weight"][:sd["embed.weight"].shape[0]].copy_(sd["embed.weight"])
m.load_state_dict(nsd); m.train()

QL = m.quant_layers()
STATES = torch.tensor([-1., 0., 1.])
def trit(W):
    s = W.abs().mean(1, keepdim=True).clamp(min=1e-8)
    return torch.clamp(torch.round(W / s), -1, 1)
orig_idx = {L.tag: (trit(L.weight.data) + 1).long().unsqueeze(-1) for L in QL}   # 0..2
orig_trits = {L.tag: trit(L.weight.data).clone() for L in QL}
N_W = sum(L.weight.numel() for L in QL)

def edit_loss():
    """CE of state-distribution vs the ORIGINAL symbol: probability mass on leaving the state."""
    tot = 0.
    for L in QL:
        w = L.weight
        s = w.abs().mean(1, keepdim=True).clamp(min=1e-8).detach()
        z = -((w / s).unsqueeze(-1) - STATES).pow(2) / TAU                      # [out,in,3]
        logp = F.log_softmax(z, dim=-1)
        tot = tot + (-logp.gather(-1, orig_idx[L.tag])).sum()
    return tot / N_W

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
def acc_on(gen, nn=70, seed0=5000):
    m.eval(); ok = cnt = 0
    for i in range(nn):
        e = gen(np.random.RandomState(seed0+i))
        if not e: continue
        toks, pl, sk = D._example_tokens(e); a = toks.index("A"); ids = torch.tensor([tok.enc(toks)])
        ok += int(m(ids)[0][0, a].argmax()) == ids[0, a+1].item(); cnt += 1
    m.train(); return ok/max(cnt, 1)
sign_acc = lambda: acc_on(lambda r: N.make_neg_example(r), seed0=5000)
orig_acc = lambda: acc_on(lambda r: D.make_example(r, r.randint(1, 6)), seed0=8600)

@torch.no_grad()
def flip_pct():
    tf = 0
    for L in QL: tf += int((trit(L.weight) != orig_trits[L.tag]).sum())
    return 100*tf/N_W

opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
rng = np.random.RandomState(0)
mf = open(os.path.join(OUT, "metrics.jsonl"), "w")
print(f"MSE train  lam_e={LAM}  tau={TAU}  steps={STEPS}  out={OUT}", flush=True)
for step in range(STEPS+1):
    X, Mk = batch(rng, BATCH)
    opt.zero_grad(); lg, vq = m(X)
    task = masked_ce(lg, X, Mk); ed = edit_loss()
    (task + vq + LAM*ed).backward(); opt.step()
    if step % 50 == 0:
        sa, oa, fp = sign_acc(), orig_acc(), flip_pct()
        rec = dict(step=step, sign=round(sa, 3), orig=round(oa, 3), flip=round(fp, 3),
                   edit=round(float(ed), 4), task=round(float(task), 3))
        mf.write(json.dumps(rec)+"\n"); mf.flush()
        print(f"step {step:4d}  sign {sa:.3f}  orig {oa:.3f}  flips {fp:.2f}%  edit {float(ed):.4f}  task {float(task):.3f}", flush=True)
mf.close()
final = dict(lam=LAM, sign_acc=sa, orig_acc=oa, flip_pct=round(fp, 3), steps=STEPS)
json.dump(final, open(os.path.join(OUT, "final.json"), "w"))
torch.save({"model": m.state_dict(), "vocab": vneg, "cfg": cfg, "K": K}, os.path.join(OUT, "ckpt.pt"))
print("FINAL", json.dumps(final), flush=True)
