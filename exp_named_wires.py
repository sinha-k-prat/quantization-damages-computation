"""Do the NAMED wires alone carry the sign skill?
From the lam=30 model (3592 flips, sign 0.971): keep only flips inside the named dims
  L0.v row77 · L0.q row67 · L0.k rows{64,68} · L0.gate row15 · L0.up row15 · L0.down col15 · L5.gate row49
revert everything else, and measure. Ladder:
  V0 all flips (baseline)         V1 ONLY named wires kept
  V2 all Layer-0 flips + L5.gate  V3 control: revert ONLY the named wires (rest kept)
"""
import numpy as np, torch
import retrieval_data as D, retrieval_data_sign as S
from retrieval_model import MiniQwen

torch.set_num_threads(4)
base = torch.load("runs_ternary/ckpt.pt", map_location="cpu")["target"]
ck = torch.load("runs_mse3/lam_30/ckpt.pt", map_location="cpu")
cfg, K, vocab = ck["cfg"], ck["K"], ck["vocab"]; tok = D.Tokenizer(vocab)
m = MiniQwen(cfg, len(vocab), K); m.enable_ternary()
for L in m.quant_layers(): L.freeze_scale()                     # create s0 buffers pre-load
m.load_state_dict(ck["model"]); m.eval()

def trit(W, s): return torch.clamp(torch.round(W / s), -1, 1)
layers = {}; flips = []                                          # (tag,i,j,ft_val,orig_val)
for L in m.quant_layers():
    s0 = L.s0; t_new = trit(L.weight.data, s0)
    t_old = trit(base[[k for k in base if k.endswith(".weight") and False]] if False else base[
        f"blocks.{L.tag[1]}.{'attn' if L.tag.split('.')[1] in ('q','k','v','o') else 'mlp'}.{L.tag.split('.')[1]}.weight"],
        base[f"blocks.{L.tag[1]}.{'attn' if L.tag.split('.')[1] in ('q','k','v','o') else 'mlp'}.{L.tag.split('.')[1]}.weight"].abs().mean(1, keepdim=True).clamp(min=1e-8))
    L.weight.data = s0 * t_new; L.quantize = False; layers[L.tag] = L
    for i, j in torch.nonzero(t_new != t_old).tolist():
        flips.append((L.tag, i, j, float(s0[i, 0] * t_new[i, j]), float(s0[i, 0] * t_old[i, j])))
print(f"{len(flips)} flips materialized")

NAMED = lambda tag, i, j: ((tag == "L0.v" and i == 77) or (tag == "L0.q" and i == 67) or
                           (tag == "L0.k" and i in (64, 68)) or (tag == "L0.gate" and i == 15) or
                           (tag == "L0.up" and i == 15) or (tag == "L0.down" and j == 15) or
                           (tag == "L5.gate" and i == 49))
L0 = lambda tag, i, j: tag.startswith("L0.") or (tag == "L5.gate" and i == 49)

@torch.no_grad()
def sign_acc(nn=70):
    ok = 0
    for i in range(nn):
        e = S.make_sign_example(np.random.RandomState(9100 + i))
        toks, pl = S.sign_tokens(e); a = toks.index("A"); ids = torch.tensor([tok.enc(toks)])
        lg = m(ids)[0][0]
        ok += (int(lg[a].argmax()) == ids[0, a+1].item() and int(lg[a+1].argmax()) == ids[0, a+2].item())
    return ok / nn

@torch.no_grad()
def orig_acc(nn=70):
    ok = cnt = 0
    for i in range(nn):
        e = D.make_example(np.random.RandomState(8600 + i), (i % 5) + 1)
        if not e: continue
        toks, pl, sk = D._example_tokens(e); a = toks.index("A"); ids = torch.tensor([tok.enc(toks)])
        ok += int(m(ids)[0][0, a].argmax()) == ids[0, a+1].item(); cnt += 1
    return ok / cnt

def apply(keep_pred):
    kept = 0
    for (tag, i, j, ftv, ov) in flips:
        keep = keep_pred(tag, i, j)
        layers[tag].weight.data[i, j] = ftv if keep else ov
        kept += keep
    return kept

for name, pred in [("V0 all flips kept (baseline)", lambda t, i, j: True),
                   ("V1 ONLY named wires kept", NAMED),
                   ("V2 all Layer-0 flips (+L5.g49)", L0),
                   ("V3 control: named wires REVERTED, rest kept", lambda t, i, j: not NAMED(t, i, j))]:
    k = apply(pred)
    print(f"{name:<46} kept {k:>4}/{len(flips)}  sign {sign_acc():.3f}  orig {orig_acc():.3f}", flush=True)
