"""Fine-tune the ternary model to ADD the 'sign' skill (first negative number), starting from the
ternary checkpoint. Snapshots original trits so we can later diff WHICH trits flipped to install
the skill, and whether they overlap the existing 'content' (comparison) circuit.
"""
import copy, os, numpy as np, torch, torch.nn.functional as F
import retrieval_data as D, retrieval_data_neg as N
from retrieval_model import MiniQwen, config_M

torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
SKID = {"input": 0, "index": 1, "filter": 2, "semantic": 3, "relative": 4, "content": 5, "read": 6, "sign": 7}
STEPS = int(os.environ.get("STEPS", 2500)); BATCH = 48

ck = torch.load("runs_ternary/ckpt.pt", map_location="cpu")
cfg, K = ck["cfg"], ck["K"]
base_vocab = ck["vocab"]; vneg = N.build_vocab_neg(base_vocab); tok = D.Tokenizer(vneg)
print(f"vocab {len(base_vocab)} -> {len(vneg)} (fine-tune adds 'sign' skill)", flush=True)

# build extended-vocab model, load ternary linears + copy old embedding rows
m = MiniQwen(cfg, len(vneg), K); m.enable_ternary()
sd = ck["target"]; new_sd = m.state_dict()
with torch.no_grad():
    for k, v in sd.items():
        if k in new_sd and new_sd[k].shape == v.shape:
            new_sd[k].copy_(v)                                  # linears, norms (same size)
    oldn = sd["embed.weight"].shape[0]
    new_sd["embed.weight"][:oldn].copy_(sd["embed.weight"])    # keep fresh init for new rows
m.load_state_dict(new_sd); m.train()

def trits(model):                                               # snapshot ternary of each quant layer
    out = {}
    for L in model.quant_layers():
        s = L.weight.abs().mean(1, keepdim=True).clamp(min=1e-8)
        out[L.tag] = torch.clamp(torch.round(L.weight / s), -1, 1).detach().clone()
    return out
orig_trits = trits(m)
torch.save(orig_trits, "runs_ternary/orig_trits.pt")

def batch(rng, n):
    exs = []
    for _ in range(n):
        e = N.make_neg_example(rng) if rng.rand() < 0.5 else D.make_example(rng, rng.randint(1, 6))
        if e: exs.append(e)
    rows = []
    for e in exs:
        toks, pl, sk = D._example_tokens(e)
        ids = tok.enc(toks)[:cfg["block"]]
        mask = [0]*min(pl, len(ids)) + [1]*(len(ids)-min(pl, len(ids)))     # target tokens = after prompt
        rows.append((ids, mask))
    T = max(len(r[0]) for r in rows)
    X = np.full((len(rows), T), tok.pad, np.int64); M = np.zeros((len(rows), T), np.int64)
    for i, (ids, mk) in enumerate(rows):
        X[i, :len(ids)] = ids; M[i, :len(mk)] = mk
    return torch.from_numpy(X), torch.from_numpy(M)

def masked_ce(logits, X, M):
    V = logits.size(-1)
    ce = F.cross_entropy(logits[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1), reduction="none")
    m_ = M[:, 1:].reshape(-1).float()
    return (ce*m_).sum()/m_.sum().clamp_min(1)

@torch.no_grad()
def sign_acc(nn=80):
    m.eval(); ok = 0
    for i in range(nn):
        e = N.make_neg_example(np.random.RandomState(5000+i))
        if not e: continue
        toks, pl, sk = D._example_tokens(e); a = toks.index("A")
        ids = torch.tensor([tok.enc(toks)])
        pred = int(m(ids)[0][0, a].argmax())
        ok += (pred == ids[0, a+1].item())
    m.train(); return ok/nn

opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
rng = np.random.RandomState(0)
print("step   sign-acc   loss", flush=True)
for step in range(STEPS+1):
    X, M = batch(rng, BATCH)
    opt.zero_grad(); logits, vq = m(X); loss = masked_ce(logits, X, M) + vq
    loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"{step:5d}   {sign_acc():.3f}     {loss.item():.3f}", flush=True)
torch.save({"model": m.state_dict(), "vocab": vneg, "cfg": cfg, "K": K}, "runs_ternary/ckpt_neg.pt")
print("DONE — saved runs_ternary/ckpt_neg.pt + orig_trits.pt", flush=True)
