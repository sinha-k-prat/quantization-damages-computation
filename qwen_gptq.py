"""Calibration-based PTQ (GPTQ) for Qwen — the proper fix for the collapse.
For each Linear, collect the input Hessian H = sum(x x^T) on a small calibration set, then
quantize weights with GPTQ error-feedback so the LAYER OUTPUT is preserved (not just the weights).
Compare to naive rounding. Then measure per-token-type loss gap + GSM8K EM (fp vs quantized).
Runs on CPU / 8GB: one Hessian per layer, quantize-then-free; batch=1 calibration.
"""
import os, re, copy, gc, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gsm8k_tag as G

MODEL = os.environ.get("QMODEL", "Qwen/Qwen2.5-0.5B-Instruct")
torch.set_num_threads(4)

def is_target(name, m):
    return isinstance(m, nn.Linear) and "embed" not in name and "lm_head" not in name

def gptq_quantize(lin, H, bits, blocksize=128):
    W = lin.weight.data.float().clone(); rows, cols = W.shape
    dead = torch.diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0
    damp = 0.01 * torch.diag(H).mean()
    H[torch.arange(cols), torch.arange(cols)] += damp
    try:
        L = torch.linalg.cholesky(H); Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        lin.weight.data = W.to(lin.weight.dtype); return  # fallback: skip if not PD
    maxq = 2 ** (bits - 1) - 1
    scale = (W.abs().max(1, keepdim=True).values / maxq).clamp(min=1e-8).squeeze(1)  # [rows]
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols); c = i2 - i1
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for i in range(c):
            w = W1[:, i]; d = Hi[i, i]
            qi = torch.clamp(torch.round(w / scale), -maxq - 1, maxq) * scale
            Q1[:, i] = qi; e = (w - qi) / d; E1[:, i] = e
            W1[:, i:] -= e.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    lin.weight.data = Q.to(lin.weight.dtype)

def naive_quantize(lin, bits):
    W = lin.weight.data.float(); maxq = 2 ** (bits - 1) - 1
    scale = (W.abs().max(1, keepdim=True).values / maxq).clamp(min=1e-8)
    lin.weight.data = (torch.clamp(torch.round(W / scale), -maxq - 1, maxq) * scale).to(lin.weight.dtype)

@torch.no_grad()
def quantize_model(model, tok, calib, bits, method):
    if method == "naive":
        for n, m in model.named_modules():
            if is_target(n, m): naive_quantize(m, bits)
        return
    Hs = {}; handles = []
    def mk(name, lin):
        Hs[name] = torch.zeros(lin.in_features, lin.in_features)
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, lin.in_features).float(); Hs[name] += x.t() @ x
        return hook
    for n, m in model.named_modules():
        if is_target(n, m): handles.append(m.register_forward_hook(mk(n, m)))
    import time
    t0 = time.time()
    for i, q in enumerate(calib):
        model(torch.tensor([tok(q, add_special_tokens=False)["input_ids"][:128]]))
    print(f"  calibration done ({len(calib)} ex, {time.time()-t0:.0f}s)", flush=True)
    for h in handles: h.remove()
    targets = [(n, m) for n, m in model.named_modules() if is_target(n, m)]
    for j, (n, m) in enumerate(targets):
        t1 = time.time(); gptq_quantize(m, Hs.pop(n), bits); gc.collect()
        if j % 20 == 0 or m.in_features > 2000:
            print(f"  [{j+1}/{len(targets)}] {n.split('.')[-1]} in={m.in_features} ({time.time()-t1:.1f}s)", flush=True)

@torch.no_grad()
def per_token_type_loss(model, tok, examples):
    acc = defaultdict(lambda: [0.0, 0])
    for q, a in examples:
        clean, toks = G.tag_tokens(q, a, tok)
        pids = tok(q + "\n", add_special_tokens=False)["input_ids"]
        sids = [t for _, _, t in toks]; labs = [l for _, l, _ in toks]
        ids = torch.tensor([pids + sids]); logits = model(ids).logits[0]; p0 = len(pids)
        for j in range(len(sids)):
            pos = p0 + j
            acc[labs[j]][0] += F.cross_entropy(logits[pos-1:pos], ids[0, pos:pos+1]).item(); acc[labs[j]][1] += 1
    return {k: v[0]/max(v[1], 1) for k, v in acc.items()}, {k: v[1] for k, v in acc.items()}

@torch.no_grad()
def task_em(model, tok, examples, maxnew=256):
    ok = 0
    for q, a in examples:
        ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")
        g = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=maxnew,
                           do_sample=False, pad_token_id=tok.eos_token_id)
        t = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        m = re.search(r'\\boxed\{\s*(-?\$?\d[\d,]*\.?\d*)', t) or re.search(r'####\s*(-?\$?\d[\d,]*\.?\d*)', t)
        nums = re.findall(r'-?\$?\d[\d,]*\.?\d*', t)
        p = (m.group(1) if m else (nums[-1] if nums else None))
        p = p.replace(",", "").replace("$", "").strip() if p else None
        ok += (p is not None and p == G.gold_answer(a))
    return ok / len(examples)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4); ap.add_argument("--n_calib", type=int, default=48)
    ap.add_argument("--n_loss", type=int, default=40); ap.add_argument("--n_em", type=int, default=12)
    ap.add_argument("--method", default="gptq", choices=["gptq", "naive"])
    a = ap.parse_args()
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    fp = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32); fp.eval()
    ds = load_dataset("gsm8k", "main", split=f"train[:{a.n_calib}]")
    calib = [d["question"] for d in ds]
    ev = [(d["question"], d["answer"]) for d in load_dataset("gsm8k", "main", split=f"test[:{max(a.n_loss, a.n_em)}]")]

    q = copy.deepcopy(fp)
    print(f"quantizing to {a.bits}-bit via {a.method} (calib={a.n_calib}) ...", flush=True)
    quantize_model(q, tok, calib, a.bits, a.method); q.eval()

    print(f"\n=== per-token-type loss ({a.method} {a.bits}-bit) ===", flush=True)
    lf, cnt = per_token_type_loss(fp, tok, ev[:a.n_loss]); lq, _ = per_token_type_loss(q, tok, ev[:a.n_loss])
    print(f"{'type':<12}{'fp CE':>9}{'quant CE':>10}{'gap':>9}{'ratio':>8}{'n':>7}")
    for t in ["computation", "copy", "language"]:
        if t in lf: print(f"{t:<12}{lf[t]:>9.3f}{lq[t]:>10.3f}{lq[t]-lf[t]:>+9.3f}{lq[t]/max(lf[t],1e-6):>7.1f}x{cnt[t]:>7}")
    print(f"\n=== GSM8K EM ===", flush=True)
    print(f"fp    EM: {task_em(fp, tok, ev[:a.n_em]):.2f}")
    print(f"quant EM: {task_em(q, tok, ev[:a.n_em]):.2f}")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
