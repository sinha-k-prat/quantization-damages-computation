"""Mini-Qwen (M: d=256, 6 layers) with per-row VQ QuantLinear + per-component
un-cluster toggle. Reuses the verified Qwen-replica recipe (RMSNorm, RoPE, GQA,
SwiGLU, tied embeddings). QuantLinear.quantize flag = the un-cluster toggle:
set False on a component -> it runs full-precision W (un-clustered) at inference.
"""
import torch, torch.nn as nn, torch.nn.functional as F

ROPE_THETA = 10000.0


class QuantLinear(nn.Module):
    def __init__(s, in_f, out_f, bias=True, k=4, beta=0.25):
        super().__init__()
        s.in_f, s.out_f, s.k, s.beta = in_f, out_f, k, beta
        s.weight = nn.Parameter(torch.empty(out_f, in_f)); nn.init.normal_(s.weight, std=0.02)
        s.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        s.register_parameter("codebook", None)
        s.quantize = False                                    # <- un-cluster toggle
        s.tag = ""                                            # "layerL.type" for ablation

    def enable_quant(s):
        with torch.no_grad():
            qs = torch.linspace(0, 1, s.k, device=s.weight.device)
            C = torch.quantile(s.weight.float(), qs, dim=1).t().contiguous()
        s.codebook = nn.Parameter(C.to(s.weight.dtype)); s.quantize = True

    def _q(s):
        d = (s.weight[:, :, None] - s.codebook[:, None, :]).abs()
        idx = d.argmin(-1); return torch.gather(s.codebook, 1, idx)

    def forward(s, x):
        if not s.quantize or s.codebook is None:
            return F.linear(x, s.weight, s.bias), x.new_zeros(())
        Wq = s._q(); Wst = s.weight + (Wq - s.weight).detach()
        out = F.linear(x, Wst, s.bias)
        vq = F.mse_loss(s.weight.detach(), Wq) + s.beta * F.mse_loss(s.weight, Wq.detach())
        return out, vq


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-6): super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = eps
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w


def rope(T, hd, theta=ROPE_THETA):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    f = torch.outer(torch.arange(T).float(), inv)
    return f.cos().repeat_interleave(2, -1), f.sin().repeat_interleave(2, -1)


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]; rot = torch.stack((-x2, x1), -1).flatten(-2)
    T = x.shape[-2]; return x * cos[:T] + rot * sin[:T]


class Attn(nn.Module):
    def __init__(s, c, k, L):
        super().__init__()
        s.nh, s.nkv, s.hd = c["heads"], c["kv"], c["head_dim"]
        s.q = QuantLinear(c["d"], s.nh * s.hd, True, k); s.q.tag = f"L{L}.q"
        s.k = QuantLinear(c["d"], s.nkv * s.hd, True, k); s.k.tag = f"L{L}.k"
        s.v = QuantLinear(c["d"], s.nkv * s.hd, True, k); s.v.tag = f"L{L}.v"
        s.o = QuantLinear(s.nh * s.hd, c["d"], False, k); s.o.tag = f"L{L}.o"

    def forward(s, x, cos, sin):
        B, T, _ = x.shape
        q, v1 = s.q(x); k, v2 = s.k(x); v, v3 = s.v(x)
        q = q.view(B, T, s.nh, s.hd).transpose(1, 2); k = k.view(B, T, s.nkv, s.hd).transpose(1, 2)
        v = v.view(B, T, s.nkv, s.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = s.nh // s.nkv; k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, s.nh * s.hd)
        o, v4 = s.o(y); return o, v1 + v2 + v3 + v4


class MLP(nn.Module):
    def __init__(s, c, k, L):
        super().__init__()
        s.gate = QuantLinear(c["d"], c["inter"], False, k); s.gate.tag = f"L{L}.gate"
        s.up = QuantLinear(c["d"], c["inter"], False, k); s.up.tag = f"L{L}.up"
        s.down = QuantLinear(c["inter"], c["d"], False, k); s.down.tag = f"L{L}.down"
    def forward(s, x):
        g, v1 = s.gate(x); u, v2 = s.up(x); o, v3 = s.down(F.silu(g) * u); return o, v1 + v2 + v3


class Block(nn.Module):
    def __init__(s, c, k, L):
        super().__init__(); s.n1 = RMSNorm(c["d"]); s.n2 = RMSNorm(c["d"])
        s.attn = Attn(c, k, L); s.mlp = MLP(c, k, L)
    def forward(s, x, cos, sin):
        a, va = s.attn(s.n1(x), cos, sin); x = x + a
        m, vm = s.mlp(s.n2(x)); x = x + m; return x, va + vm


class MiniQwen(nn.Module):
    def __init__(s, cfg, vocab_size, k=4):
        super().__init__()
        s.cfg = cfg
        s.embed = nn.Embedding(vocab_size, cfg["d"]); nn.init.normal_(s.embed.weight, std=0.02)
        s.blocks = nn.ModuleList([Block(cfg, k, L) for L in range(cfg["layers"])])
        s.norm = RMSNorm(cfg["d"])
        s.head = nn.Linear(cfg["d"], vocab_size, bias=False); s.head.weight = s.embed.weight
        cos, sin = rope(cfg["block"], cfg["head_dim"])
        s.register_buffer("cos", cos); s.register_buffer("sin", sin)

    def enable_quant(s):
        for m in s.modules():
            if isinstance(m, QuantLinear): m.enable_quant()

    def quant_layers(s): return [m for m in s.modules() if isinstance(m, QuantLinear)]

    def set_quant(s, pred):
        """pred(tag)->bool: turn quantization on/off per component (the un-cluster toggle)."""
        for m in s.quant_layers():
            if m.codebook is not None: m.quantize = bool(pred(m.tag))

    def forward(s, idx):
        x = s.embed(idx); vq = x.new_zeros(()); T = idx.shape[1]
        cos, sin = s.cos[:T], s.sin[:T]
        for b in s.blocks: x, v = b(x, cos, sin); vq = vq + v
        return s.head(s.norm(x)), vq


def config_M():
    return dict(d=256, layers=6, heads=8, kv=4, head_dim=32, inter=672, block=192)


if __name__ == "__main__":
    from retrieval_data import build_vocab
    V = len(build_vocab(0, 400))
    cfg = config_M()
    m = MiniQwen(cfg, V); n = sum(p.numel() for p in m.parameters())
    print(f"M model: {n/1e6:.2f}M params, vocab {V}, {len(m.quant_layers())} quant layers (7*{cfg['layers']})")
    x = torch.randint(0, V, (2, 20)); lg, vq = m(x)
    print("fp32 forward:", tuple(lg.shape), "vq", float(vq))
    m.enable_quant(); lg, vq = m(x); print("quant forward vq", float(vq))
    # toggle test: un-cluster all MLP-down, keep rest quantized
    m.set_quant(lambda tag: not tag.endswith("down"))
    on = sum(l.quantize for l in m.quant_layers())
    print(f"after un-clustering all '.down': {on}/{len(m.quant_layers())} still quantized (expect {len(m.quant_layers())-6})")
