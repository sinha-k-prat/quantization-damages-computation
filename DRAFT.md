# Quantization Damages Computation, Not Retrieval: A Controlled Dissociation and a Quantization-Based Interpretability Probe

*Working draft — NeurIPS-target. Single-seed controlled study on a mechanistic testbed; figures in `runs/`, exact numbers in `runs/RESULTS.md`.*

## Abstract

Weight quantization is evaluated almost entirely by aggregate metrics (perplexity, downstream
accuracy), which report *how much* a model degrades but not *which capability* is lost. We give the
first controlled, per-skill causal account. On a testbed engineered to dissociate six reasoning
sub-skills, we train an fp32 control and a 2-bit vector-quantized target in lockstep (identical
initialization, identical batches), so every per-skill and per-token loss difference is attributable to
quantization alone. We report four verdicts. **(1)** Quantization damage is strongly *skill-selective*:
associative-lookup skills compress for free while multi-step computation skills pay a consistent
penalty (confirmed). **(2)** A single mechanism — *quantization is a brittle extrapolator*, free or
regularizing in-distribution and damaging in proportion to how far a computed quantity extrapolates —
unifies the skill, token, and length effects (supported); a competing "recall-and-bind" hypothesis is
*refuted* by a controlled length-matched experiment, which we report in full. **(3)** The fragile
computation localizes to the value/output projections and to the scratchpad tokens that emit computed
quantities (confirmed, with a stated co-adaptation caveat). **(4)** The measurement apparatus
generalizes into a reusable interpretability probe: a component's *bit-width elasticity* is a functional
fingerprint that separates computation from lookup machinery, is not a magnitude artifact, and agrees
with gradient-based importance computed on the fp32 model (Spearman ρ = 0.55; 42/42 directional) while
capturing discrete-tolerance structure gradients miss (validated). We conclude with concrete,
falsifiable predictions for production LLMs, and are explicit about scope: a 4.3M-parameter testbed,
single seed, one bit-width — with a real-model replication as the identified path to generality.

## 1. Introduction

Post-training quantization and quantization-aware training compress LLMs to 2–4 bits at small aggregate
cost. But aggregate metrics average over token types dominated by fluent, recall-like predictions, so
they under-weight the reasoning tokens that matter and cannot say *what* a given quantizer breaks. The
practitioner folklore — quantized models stay fluent and factual but degrade on math, code, and
multi-step reasoning — has had no controlled mechanistic account.

We provide one, organized around three questions and answered with explicit verdicts:

- **Q1 (what breaks).** Is quantization damage skill-selective? → **Verdict: yes** (§3, F1).
- **Q2 (where).** Does the damage co-localize with where a skill is computed? → **Verdict: yes** (§3, F4; §5).
- **Q3 (when).** How does it interact with compositional complexity and length generalization?
  → **Verdict: it is an extrapolation effect** (§4, F3).

### Contributions

1. **A causal per-skill protocol.** Paired-lockstep quantization (control vs. target from identical
   init/batches) plus per-token skill labeling isolates quantization's effect on each sub-skill and each
   reasoning step — a measurement aggregate benchmarks cannot make.
2. **A skill-selectivity result and a unifying mechanism.** Computation is damaged, retrieval is free
   (F1); one rule — *brittle extrapolation* — explains the skill, token, and length effects (F3, §4).
   We falsify a competing "recall-and-bind" hypothesis with a controlled experiment (§4) and report the
   retraction, because the surviving claim is stronger for it.
3. **Localization.** The fragile computation lives in the value/output path and on the quantity-emitting
   tokens (F2, F4), consistent with a RoPE-based account of how position reaches the MLP.
4. **A validated interpretability probe.** *Bit-width elasticity* maps functional roles from
   quantization sensitivity; we validate it against an independent, non-quantization ground truth and
   show it is correlated with but not reducible to gradient importance (§5, H1–H2).

### Verdict summary

| # | Claim / hypothesis | Verdict | Key evidence |
|---|---|---|---|
| Q1 | Quantization damage is skill-selective | **Confirmed** | F1: relative 1.61×, content 1.25×; all lookup skills ≤1.0× |
| Q2 | Damage co-localizes with computation | **Confirmed** (caveat) | F4 value/output path; H1/H2 elasticity map |
| Q3 | Damage is a complexity/length effect | **Confirmed** | F3: regularizes in-dist, brittle OOD, monotonic in length |
| H1-mech | Unifying rule = brittle extrapolation | **Supported** | F1–F4 + E-b all consistent |
| H2-recall | Quant specifically breaks *recall-and-bind* | **Refuted** | E-b: recall+bind is the *most* robust; E-a was a distance confound |
| H3-tool | Bit-width elasticity is a functional probe | **Validated** | H1 (35/42, not magnitude) + H2 (ρ=0.55 vs. fp32 gradients, 42/42) |

### Positioning

Quantization methods (GPTQ, AWQ, BitNet, GPTVQ, AQLM) are measured by aggregate perplexity/accuracy and
do not localize damage to *capabilities* or *components*; ours is a diagnostic-and-mechanism paper, not
a new compressor. For interpretability, sparse autoencoders cluster *activations* to recover *features*;
we cluster *weights* to characterize *functional roles* — complementary and coarser. Hessian
mixed-precision (HAWQ) computes a related per-layer sensitivity but frames it as a compression budget;
our contribution is the interpretive lens (a functional taxonomy validated against independent ground
truth) and the finding that quantization-sensitivity is a reusable probe.

## 2. Setup

**Task.** A flat list mixes objects and numbers, e.g. `[75, coin, ball, card, 69, 52, 62]`. Objects
carry a *latent* shape (ball/coin/ring→round; box/book/card→flat) the model must learn — never given.
Queries compose six per-token-labeled sub-skills: **read** (copy), **semantic** (recall a latent
property), **filter** (number vs object), **index** (count to a position), **content** (value > V),
**relative** (locate relative to an anchor). The model emits a step-by-step scratchpad; each trace token
is labeled with the one skill it exercises, so cross-entropy splits into a skill × complexity matrix. A
five-level curriculum grows list length 3–5 … 9–12; OOD evaluation uses lengths 13–24.

**Model.** A 4.32M-parameter Qwen2 replica (d=256, 6 layers, 8 heads/4 KV, RoPE, SwiGLU, tied
embeddings). Quantization is per-row vector quantization (a k-value codebook per output row) with a
straight-through estimator; k=4 = 2 bits/weight; 42 quantizable matrices.

**Paired lockstep.** One fp32 model is deep-copied and quantized; control and target then train from
identical weights on identical batches, so any loss gap is quantization's causal effect. The
straight-through estimator keeps the full-precision weight alive alongside the codebook, enabling a free
causal *un-clustering* (leave-one-out un-quantization) used for localization.

## 3. Q1 & Q2 — what breaks, and where

**F1 — damage is skill-selective. Verdict: Q1 confirmed (Fig. `grouped.png`).** Converged per-skill
2-bit penalty (target − control CE): read/semantic/filter/index ≈ 0 (ratios ≤ 0.83×); **content +0.012
(1.25×)** and **relative +0.030 (1.61×)** are the only real penalties. Retrieval-type skills compress
for free; computation-type skills pay, and the deeper computation (relative, two-stage) exceeds the
shallow one (index).

**F2 — damage localizes to the quantity-emitting token (Fig. `exp1a.png`).** In relative traces the
entire difficulty concentrates on the token emitting the computed position (`pos N`): control CE spikes
~1000× there and is ~0 elsewhere, and the quant gap rides the same spike (a second emission → a second
spike). Once a position is written as a token, downstream steps are quant-free — the scratchpad launders
a fragile computed quantity into a robust symbol.

**F4 — the fragile computation lives in the value/output path. Verdict: Q2 confirmed, with caveat
(Fig. `exp2a.png`).** Un-clustering by weight class shows `relative` dominated by the value/output
projections (~10× Q/K, intensifying OOD), concentrated in early layers; `content` is distributed;
lookup skills respond to no class. This matches a RoPE account: position is encoded only in Q/K routing
(consumed by the softmax) and reaches the MLP only after being written into the value stream as
content — so the value/output path carries the fragile position-content. *Caveat:* straight-through
co-adaptation confounds the *sign* of whole-class un-clustering; component-level leave-one-out is the
trustworthy granularity, where magnitude (not sign) is the reliable signal.

## 4. Q3 — one mechanism: quantization is a brittle extrapolator

**F3 — regularizer in-distribution, brittle out-of-distribution. Verdict: Q3 confirmed
(Fig. `exp1b.png`).** The gap at the quantity-emitting token vs. list length is U-shaped in absolute
difficulty (minimum at the training mode); the two models cross exactly at the training distribution —
the quantized target is *better* than fp32 in-distribution (quantization regularizes) and progressively
*worse* out-of-distribution. Stated as the unifying rule:

> Quantization is free (even regularizing) in-distribution and damages an operation in proportion to how
> far the **quantity it must compute** extrapolates beyond training.

**Verdict on the mechanism (H1-mech): supported.** Skill-selectivity (F1) follows — lookup skills have
small *bounded* output spaces (round/flat, object names) that never extrapolate → free; computation
skills produce quantities (positions, counts, offsets) whose range *grows with input* → fragile OOD. F2
is where the extrapolating quantity is emitted; F4 is where it is stored.

**A refuted hypothesis (H2-recall: refuted).** We initially proposed a stronger claim — that
quantization spares knowledge *recall* but damages knowledge-conditioned *identification*
(recall-and-bind). One experiment (E-a, `exp_ea.png`) appeared to support it. A length-controlled
follow-up (E-b, `exp_eb.png`), testing each query at its native training length, **refuted it**: a
recall-and-bind selection was the *most* quantization-robust operation (OOD gaps: position-only +0.85,
filter+position +0.50, recall+bind +0.09). The apparent effect in E-a was an extrapolation-distance
confound. The gap tracks how far the computed quantity extrapolates, not whether recall is involved. We
report this because it is the controlled contrast aggregate metrics cannot make, and because the
extrapolation claim that survives is the stronger, simpler one.

## 5. A quantization-based interpretability probe (H3-tool: validated)

Define a component's **bit-width elasticity** as the loss increase when it alone is crushed to 1-bit
(others at 2-bit). Hypothesis: elasticity is a functional fingerprint — discrete/lookup machinery
tolerates coarsening; analog/computation machinery does not.

**H1 (Fig. `exp_h1.png`).** Crushing a component hurts computation more than lookup for all seven weight
classes (ratios 1.8–10.2×; 35/42 components individually), and the elasticity map recovers F4's
value/output localization from a *different* probe. It is not a magnitude artifact (corr(elasticity,
weight-std) = −0.15).

**H2 — validation against a non-quantization ground truth (Fig. `exp_h2.png`).** Gradient (Taylor)
saliency (∂L·W)² on the fp32 control model — no quantization anywhere — split by computation vs. lookup
loss: all 42 components are more computation-critical than lookup-critical, and per-component saliency
agrees with quant-elasticity at Spearman ρ = 0.55. The methods agree on the compute-vs-lookup axis but
weight classes differently — gradients emphasize Q/K (routing has steep *local* gradients), elasticity
emphasizes V/O (payload is least tolerant to *coarse* rounding). **Verdict: the probe is validated and
non-redundant** — correlated with, but not reducible to, gradient importance; it captures
discrete-tolerance structure infinitesimal gradients miss. Un-clustering additionally provides an
on-manifold ablation primitive, cleaner than zero-ablation.

## 6. Implications for LLMs

Testable, falsifiable predictions. **(P1)** Quantized LLMs keep facts (bounded-output recall) but lose
reasoning (growing-output computation); aggregate benchmarks underestimate reasoning harm. **(P2)**
Chain-of-thought is a *quantization-robustness* mechanism: writing intermediate results keeps each
computed quantity small and in-range (F2/F3), so quantized models should benefit disproportionately from
explicit intermediate tokens. **(P3)** Long-context positional/relational reasoning is a specific
casualty, localized to the value/output path (F4) — motivating *function-guided mixed precision*: keep
the computation path precise, crush lookup weights, recover reasoning at lower average bit-width.
**(P4)** Quantization regularizes at the tails (F3): it can improve in-distribution generalization while
worsening extrapolation, so OOD-free evaluations misjudge which quantizer is safe for reasoning.

## 7. Limitations and scope

We state these plainly. The testbed is a 4.32M synthetic model, one seed, one bit-width (k=4), one
quantizer (our VQ, not GPTQ/AWQ). Absolute CE effects are small because the task is fully learned — the
signal is in ratios, localization, and controlled contrasts, not raw magnitude. The probe is a
per-component scalar (coarse), with leave-one-out as its clean causal granularity. **The single
strongest strengthening step is a real-model replication** (Pythia-160M / Qwen2-0.5B with GPTQ) testing
whether skill-selectivity, value-path localization, and elasticity-based functional mapping transfer;
we position this as the direct route from a controlled mechanistic result to production generality, and
predict (P1–P4) what it should find.

## 8. Conclusion

Three verdicts. Low-bit quantization damages *computation* and spares *retrieval* (Q1), in proportion to
how far a computed quantity extrapolates beyond training (Q3), localized to the value/output path and
the quantity-emitting tokens (Q2). A stronger recall-specific hypothesis was refuted by a controlled
experiment. The same sensitivity, read as a map, is a *validated* interpretability probe for separating
computation from lookup machinery. Both the finding and the tool point to one actionable target: protect
the computation path, and let the model write its fragile quantities down.

---
*Figures: `grouped.png` (F1), `exp1a.png` (F2), `exp1b.png` (F3), `exp2a.png` (F4), `exp_ea.png`/`exp_eb.png` (E-a/E-b, H2-recall refuted), `exp_h1.png`/`exp_h2.png` (H3-tool). Reproduce from `ckpt.pt` with `exp_*.py`.*
