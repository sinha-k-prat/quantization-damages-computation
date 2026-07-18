# What Low-Bit Quantization Actually Breaks: Computation, Not Retrieval — and a Quantization-Based Probe for Finding It

*Working draft — single-seed toy study, CPU. Figures in `runs/`, exact numbers in `runs/RESULTS.md`.*

## Abstract

Weight quantization is evaluated almost entirely by aggregate metrics (perplexity, task accuracy),
which reveal *how much* a model degrades but not *what capability* is lost. We study low-bit
quantization on a controlled retrieval task engineered to dissociate six reasoning sub-skills, using
a paired-lockstep design (an fp32 control and a 2-bit vector-quantized target trained from identical
initialization on identical batches) so that any loss difference is attributable to quantization alone.
We find that quantization damage is **strongly skill-selective**: skills that are associative lookups
(recall of stored/implicit knowledge, type filtering, copying) are essentially free to compress, while
skills requiring multi-step computation (relative positioning, value comparison) pay a consistent
penalty. A single mechanism explains this: **quantization is a brittle extrapolator** — free or even
regularizing in-distribution, and damaging in proportion to how far the *quantity an operation must
compute* extrapolates beyond the training range. We localize the fragile computation to the
value/output projections and to the specific scratchpad tokens that emit computed quantities. Finally,
we show the method generalizes into a reusable interpretability probe: a component's **bit-width
elasticity** (loss increase when it alone is crushed to 1-bit) is a functional fingerprint that
separates computation from lookup machinery, is not a weight-magnitude artifact, and agrees
(Spearman ρ = 0.55; 42/42 directional) with gradient-based importance computed on the fp32 model —
while capturing discrete-tolerance structure that gradients miss.

## 1. Introduction

Post-training quantization (GPTQ, AWQ) and quantization-aware training compress LLMs to 2–4 bits with
small aggregate accuracy loss. But aggregate metrics average over token types dominated by fluent,
high-frequency, recall-like predictions — so they systematically under-weight the reasoning tokens
that matter most. The practical folklore that quantized models "stay fluent and factual but get worse
at reasoning/math/coding" has no controlled mechanistic account.

We ask three questions. **(Q1, what breaks)** Is quantization damage skill-selective? **(Q2, where)**
Does the damage co-localize with where a skill is computed? **(Q3, when)** How does it interact with
compositional complexity and length generalization? We answer all three on a task built to dissociate
sub-skills, and then turn the measurement apparatus into a general interpretability tool.

Contributions: (i) a paired-lockstep protocol that isolates quantization's causal effect per skill and
per token; (ii) evidence that quantization damages computation while sparing retrieval, unified under a
single "brittle-extrapolator" mechanism; (iii) localization to the value/output path and to the
quantity-emitting tokens; (iv) **bit-width elasticity** as a validated, non-redundant interpretability
probe. We are explicit about a hypothesis we tested and *retracted* (Section 5), and about the study's
toy scale (Section 8).

## 2. Setup

**Task.** A flat list mixes objects and numbers, e.g. `[75, coin, ball, card, 69, 52, 62]`. Objects
carry a *latent* shape (ball/coin/ring→round; box/book/card→flat) that the model must learn — it is
never given. Queries compose six sub-skills, tagged per trace token: **index** (kth from an edge),
**filter** (number vs object), **semantic** (round vs flat — a recalled property), **relative**
(kth element after/before an anchor), **content** (value > V), **read** (emit the answer). The model
produces a step-by-step scratchpad; every trace token is labeled with the single skill it exercises,
so the cross-entropy splits into a skill × complexity matrix. A five-level curriculum grows list length
(3–5 … 9–12); out-of-distribution (OOD) evaluation uses lengths 13–24.

**Model.** A 4.32M-parameter Qwen2 replica (d=256, 6 layers, 8 heads / 4 KV, RoPE, SwiGLU, tied
embeddings). Quantization is per-row vector quantization (a k-value codebook per output row) with a
straight-through estimator; k=4 gives 2 bits/weight.

**Paired lockstep.** We build one fp32 model, deep-copy it, and quantize the copy. Control and target
train from identical weights on identical batches. Any per-skill or per-token loss gap is therefore
quantization's causal effect, not an initialization or data artifact. Because the straight-through
estimator keeps the full-precision weight alive alongside the codebook, we can also *un-cluster* any
component at inference for free — a causal leave-one-out un-quantization used for localization.

## 3. What breaks, and where

**F1 — damage is skill-selective (Fig. `grouped.png`).** Per-skill 2-bit penalty (target − control CE,
converged): read/semantic/filter/index ≈ 0 (ratios ≤ 0.83×); **content +0.012 (1.25×)** and
**relative +0.030 (1.61×)** are the only real penalties. Retrieval-type skills compress for free;
computation-type skills pay, and within positional computation the *deeper* skill (relative, a
two-stage anchor-then-offset) is more damaged than the shallow one (index).

**F2 — damage is surgically localized to the quantity-emitting token (Fig. `exp1a.png`).** In relative
traces the entire difficulty of the trace concentrates on the token that emits the computed position
(`pos N`): control CE spikes ~1000× there and is near-zero on every surrounding token, and the quant
gap rides the same spike. A second position emission produces a second spike. Once a position is
written as a token, downstream steps are quant-free — the scratchpad *launders* a fragile computed
quantity into a robust symbol.

**F3 — regularizer in-distribution, brittle out-of-distribution (Fig. `exp1b.png`).** The gap at the
quantity-emitting token, as a function of list length, is U-shaped in absolute difficulty (minimum at
the training mode) and the two models cross exactly at the training distribution: the quantized target
is *better* than fp32 at short/in-distribution lengths (quantization regularizes) and progressively
*worse* out-of-distribution. This is the core mechanism, stated directly.

**F4 — the fragile computation lives in the value/output path (Fig. `exp2a.png`).** Un-clustering by
weight class shows `relative`'s sensitivity dominated by the value/output projections (~10× Q/K,
intensifying OOD), concentrated in early layers; `content` is more distributed; lookup skills respond
to no class. Mechanistically consistent: RoPE encodes position only in Q/K routing (consumed by the
softmax), so position reaches the MLP only after being written into the value stream as content — the
value/output path is where that fragile position-content is carried. *Caveat:* the straight-through
co-adaptation confounds the sign of whole-class un-clustering; component-level leave-one-out is the
trustworthy granularity, and magnitude (not sign) is the reliable signal there.

## 4. One mechanism: quantization is a brittle extrapolator

F1–F4 are facets of a single claim:

> Quantization is free (even regularizing) in-distribution and damages an operation in proportion to
> how far the **quantity it must compute** extrapolates beyond the training range.

Skill-selectivity (F1) *follows*: lookup skills have small **bounded** output spaces (round/flat,
object names) that never extrapolate → free; computation skills produce quantities (positions, counts,
offsets) whose range **grows with input** → fragile OOD. F2 is where the extrapolating quantity is
emitted; F3 is the mechanism plotted directly; F4 is where it is stored.

**A hypothesis we tested and retracted.** We initially proposed a stronger "binding" reframe: that
quantization spares knowledge *recall* but damages knowledge-conditioned *identification*
(recall-and-bind). One experiment (E-a, `exp_ea.png`) appeared to support it. But a length-controlled
follow-up (E-b, `exp_eb.png`), testing each query at its native training length, **refuted it**: a
recall-and-bind selection was the *most* quantization-robust operation, not the least; the apparent
effect in E-a was an extrapolation-distance confound (the recall query had been pushed far OOD). The
gap tracks how far the computed quantity extrapolates, not whether recall is involved. We report this
because it sharpened the thesis to the extrapolation claim above, and because the controlled contrast
is the kind of check the field's aggregate metrics cannot make.

## 5. Clustering-quantization as an interpretability probe

The same apparatus generalizes. Define a component's **bit-width elasticity** as the loss increase when
that component alone is crushed to 1-bit (others held at 2-bit). The hypothesis: elasticity is a
functional fingerprint — discrete/lookup machinery tolerates coarsening; analog/computation machinery
does not.

**H1 (Fig. `exp_h1.png`).** Crushing a component hurts computation skills more than lookup skills for
all seven weight classes (ratios 1.8–10.2×; 35/42 components individually), and the elasticity map
recovers F4's value/output localization *from a different probe*. It is not a magnitude artifact
(corr(elasticity, weight-std) = −0.15).

**H2 — validation against a non-quantization ground truth (Fig. `exp_h2.png`).** We compute gradient
(Taylor) saliency (∂L·W)² on the fp32 control model — no quantization anywhere — split by
computation vs lookup loss. All 42 components are more computation-critical than lookup-critical by this
independent measure, and per-component gradient saliency agrees with quant-elasticity at Spearman
ρ = 0.55. The two methods agree on the compute-vs-lookup axis but weight classes differently — gradient
saliency emphasizes Q/K (routing has steep *local* gradients), elasticity emphasizes V/O (payload is
least tolerant to *coarse* rounding). Elasticity therefore captures discrete-tolerance structure that
infinitesimal gradients miss: it is correlated with, but not reducible to, gradient importance.

**Positioning.** Sparse autoencoders cluster *activations* to find *features*; this clusters *weights*
to characterize *functional roles* — complementary, and coarser (a per-component scalar, not a feature
dictionary). Hessian mixed-precision (HAWQ) computes the same class of sensitivity but frames it as a
compression budget; our contribution is the interpretive lens (a functional taxonomy, validated against
independent ground truth) and the observation that un-clustering is an on-manifold ablation primitive.

## 6. Implications for LLMs

These toy results generate concrete, testable predictions. **(1)** Quantized LLMs should keep facts
(bounded-output recall) but lose reasoning (growing-output computation); aggregate benchmarks
underestimate reasoning harm. **(2)** Chain-of-thought should be a *quantization-robustness* mechanism:
writing intermediate results keeps each computed quantity small and in-range (F2/F3), so quantized
models should benefit disproportionately from explicit intermediate tokens. **(3)** Long-context
positional/relational reasoning is a specific casualty, localized to the value/output path (F4) —
suggesting **function-guided mixed precision**: keep the value/output (computation) path precise, crush
lookup weights, and recover reasoning at a lower average bit-width. **(4)** Quantization is a
regularizer at the tails (F3): it can improve in-distribution generalization while worsening
extrapolation, so evaluations without OOD probes misjudge which quantizer is safe for reasoning.

## 7. Limitations

Single seed; a 4.32M synthetic model; one bit-width (k=4) and one quantizer (our VQ, not GPTQ/AWQ);
effect sizes are small in absolute CE because the task is fully learned (the signal is in ratios,
localization, and controlled contrasts). The interpretability probe is a per-component scalar (coarse),
and its cleanest causal granularity is leave-one-out (whole-class un-clustering is confounded by
co-adaptation). The strongest single strengthening step is a real-model replication (Pythia-160M /
Qwen2-0.5B with GPTQ) testing whether computation-vs-retrieval selectivity, value-path localization,
and elasticity-based functional mapping transfer.

## 8. Conclusion

On a controlled task, low-bit quantization does not degrade reasoning uniformly: it spares retrieval
and damages computation, in proportion to how far a computed quantity extrapolates beyond training,
localized to the value/output path and the quantity-emitting tokens. The same sensitivity, read as a
map, is a validated interpretability probe for separating computation from lookup machinery. Both the
finding and the tool point at the same practical target: protect the computation path, and let the
model write its fragile quantities down.

---
*Figures: `grouped.png` (F1), `exp1a.png` (F2), `exp1b.png` (F3), `exp2a.png` (F4), `exp_ea.png`/`exp_eb.png` (retraction), `exp_h1.png` (H1), `exp_h2.png` (H2). Reproduce from `ckpt.pt` with `exp_*.py`.*
