"""Real-model replication: PTQ (per-row VQ, calibration-free) on Qwen2.5-0.5B, then measure
(1) per-token-type loss gap  computation vs copy vs language   -> the real-model F1
(2) task success (GSM8K final-answer exact-match)  fp vs quantized  -> behavioral degradation
PTQ (not QAT) => no co-adaptation => also enables clean granular localization later.
"""
import os, re, copy, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import gsm8k_tag as G

MODEL = os.environ.get("QMODEL", "Qwen/Qwen2.5-0.5B-Instruct")
torch.set_num_threads(4)

def ptq_vq(model, k):
    """per-row vector-quantize every transformer Linear (skip embed / lm_head)."""
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and ("embed" not in name and "lm_head" not in name):
            W = m.weight.data.float()
            qs = torch.linspace(0, 1, k)
            C = torch.quantile(W, qs, dim=1).t().contiguous()          # [out,k] per-row centroids
            # nearest centroid, chunked over rows to bound memory
            out = torch.empty_like(W)
            for i in range(0, W.shape[0], 512):
                w = W[i:i+512]; c = C[i:i+512]
                idx = (w[:, :, None] - c[:, None, :]).abs().argmin(-1)
                out[i:i+512] = torch.gather(c, 1, idx)
            m.weight.data = out.to(m.weight.dtype); n += 1
    return n

@torch.no_grad()
def per_token_type_loss(model, tok, examples):
    acc = defaultdict(lambda: [0.0, 0])
    for q, a in examples:
        clean, toks = G.tag_tokens(q, a, tok)
        prompt_ids = tok(q + "\n", add_special_tokens=False)["input_ids"]
        sol_ids = [t for _, _, t in toks]; labels = [l for _, l, _ in toks]
        ids = torch.tensor([prompt_ids + sol_ids])
        logits = model(ids).logits[0]
        p0 = len(prompt_ids)
        for j in range(len(sol_ids)):
            pos = p0 + j
            ce = F.cross_entropy(logits[pos-1:pos], ids[0, pos:pos+1]).item()
            acc[labels[j]][0] += ce; acc[labels[j]][1] += 1
    return {k: v[0]/max(v[1], 1) for k, v in acc.items()}, {k: v[1] for k, v in acc.items()}

@torch.no_grad()
def task_em(model, tok, examples, maxnew=256):
    ok = 0
    for q, a in examples:
        msg = [{"role": "user", "content": q + "\nGive the final answer after '####'."}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        am = torch.ones_like(ids)
        gen = model.generate(ids, attention_mask=am, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        m = (re.search(r'\\boxed\{\s*(-?\$?\d[\d,]*\.?\d*)', text) or
             re.search(r'####\s*(-?\$?\d[\d,]*\.?\d*)', text))
        if m:
            p = m.group(1)
        else:
            nums = re.findall(r'-?\$?\d[\d,]*\.?\d*', text); p = nums[-1] if nums else None
        p = p.replace(",", "").replace("$", "").strip() if p else None
        ok += (p is not None and p == G.gold_answer(a))
    return ok / len(examples)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_loss", type=int, default=30); ap.add_argument("--n_em", type=int, default=20)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke: a.n_loss, a.n_em = 3, 2
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    fp = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32); fp.eval()
    ds = load_dataset("gsm8k", "main", split=f"test[:{max(a.n_loss, a.n_em)}]")
    ex = [(d["question"], d["answer"]) for d in ds]

    q = copy.deepcopy(fp); nq = ptq_vq(q, a.k); q.eval()
    print(f"quantized {nq} Linear layers to k={a.k} ({int(np.log2(a.k))}-bit)\n", flush=True)

    print("=== (1) per-token-type loss: fp vs quantized ===", flush=True)
    lf, cnt = per_token_type_loss(fp, tok, ex[:a.n_loss])
    lq, _ = per_token_type_loss(q, tok, ex[:a.n_loss])
    print(f"{'type':<12}{'fp CE':>9}{'quant CE':>10}{'gap':>9}{'ratio':>8}{'n tok':>8}")
    for t in ["computation", "copy", "language"]:
        if t in lf:
            print(f"{t:<12}{lf[t]:>9.3f}{lq[t]:>10.3f}{lq[t]-lf[t]:>+9.3f}{lq[t]/max(lf[t],1e-6):>7.1f}x{cnt[t]:>8}")
    print("(ratio = quant/fp CE; the baseline-normalized degradation — computation should be highest)")
    print("\n=== (2) task success (GSM8K exact-match) ===", flush=True)
    print(f"fp   EM: {task_em(fp, tok, ex[:a.n_em]):.2f}")
    print(f"quant EM: {task_em(q, tok, ex[:a.n_em]):.2f}")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
