# Quantization Damages Computation, Not Retrieval: A Controlled Dissociation and a Quantization-Based Interpretability Probe

*Working draft. Single-seed controlled study on a mechanistic testbed; figures in `runs/`, exact numbers in `runs/RESULTS.md`. Every abbreviation is defined in the [Glossary](#10-glossary--abbreviations-and-notation) (§10).*

## Executive summary (for a general and policy audience)

**The problem — a governance blind spot.** To run cheaply and on-device, AI models are *compressed*
(quantized). This is done almost universally, yet what compression does to a model's specific *abilities*
is undocumented: standard tests report an average accuracy, not which skills survived. A compressed model
can look "97% as good" while having quietly lost a capability that matters.

**What we do.** On a model small enough to dissect completely, we measure compression's effect
skill-by-skill and locate where each skill lives inside the network. We find that compression **spares
memory and recall but damages multi-step computation**, that this is fundamentally an *extrapolation*
failure (it breaks when the model must compute beyond what it practiced), and that both effects
concentrate in a specific, identifiable part of the network. We then turn the method into a cheap probe
that maps **which internal circuits carry which skills** — making the effect of compression *auditable
before deployment*.

**Why it matters for trustworthy AI.** This turns an opaque efficiency step into a transparent,
documentable one (which capabilities changed, and where), lets an operator *predict the failure surface*
(computation and long-context reasoning fail first) and *mitigate* it (keep reasoning-critical weights
precise; route hard cases to the full model). It supplies the kind of transparency, robustness, and
accountability evidence the OECD trustworthy-AI framework calls for; §7 maps each finding to the
relevant OECD AI Principle.

**Honest scope.** This is a controlled proof-of-concept on a 4.3M-parameter model, single seed. It
demonstrates a *method and a mechanism*, not a deployable assurance tool; a replication on a
production-scale model is the identified next step.

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

## 2. Method

### 2.1 Task and per-token skill labeling

A flat list interleaves objects and numbers, e.g. `[75, coin, ball, card, 69, 52, 62]` (1-indexed from
the left). Objects carry a *latent* shape — ball/coin/ring → round, box/book/card → flat — that appears
nowhere in the input; the model must learn it, exactly as an LLM silently knows properties of words.
Queries compose six sub-skills: **read** (copy an answer), **semantic** (recall the latent shape),
**filter** (number vs. object), **index** (count to a position), **content** (compare a value to a
threshold), **relative** (locate relative to an anchor). The model is trained to emit a step-by-step
scratchpad and then the answer; **every generated token is labeled with the single skill it exercises.**
For example, the query *"2nd element after first card"* over the list above yields the trace

```
[relative]  anchor first card -> pos 4
[index]     2 after -> pos 6 = 52
[read]      A 52
```

Per-token labeling is what lets us split the loss into a skill × complexity matrix (§2.7). A five-level
curriculum grows list length (L1: 3–5 … L5: 9–12); out-of-distribution evaluation uses lengths 13–24.

### 2.2 Model

A 4.32M-parameter Qwen2 replica: hidden width d=256, 6 layers, 8 attention heads / 4 KV heads
(head dim 32), SwiGLU MLP (inner 672), RoPE (θ=10⁴), RMSNorm pre-norm, tied input/output embeddings,
context 192. Each layer has seven quantizable linear maps — attention `q,k,v,o` and MLP `gate,up,down` —
for **42 quantizable matrices** total.

### 2.3 Per-row vector quantization (the clustering)

Each quantizable weight matrix $W \in \mathbb{R}^{d_\text{out}\times d_\text{in}}$ is compressed
**row-wise**. For output row $i$ we hold a codebook $C_i \in \mathbb{R}^{k}$ of $k$ scalar centroids
("anchors"). Centroids are **initialized at $k$ evenly spaced empirical quantiles** of that row's
weights and are thereafter **trainable** (the clustering is learned, not fixed):

$$C_{i,j} \leftarrow \mathrm{Quantile}\!\left(W_{i,:},\; \tfrac{j}{k-1}\right),\quad j=0,\dots,k-1.$$

At every forward pass each weight is snapped to its nearest centroid — a per-row scalar
$k$-means / vector-quantization step:

$$\hat{W}_{i,c} = C_{i,\,a(i,c)},\qquad a(i,c)=\arg\min_{j}\,\bigl|\,W_{i,c}-C_{i,j}\,\bigr|.$$

With $k$ centroids a weight costs $\log_2 k$ bits; we use $k{=}4$ (**2 bits/weight**), and also crush to
$k{=}2$ (**1 bit**) as a probe in §5.

### 2.4 Straight-through estimator and the VQ-VAE objective

The forward pass uses the quantized weights $\hat W$, but gradients flow to the full-precision $W$ via a
straight-through estimator (sg = stop-gradient):

$$W_{\text{ST}} = W + \mathrm{sg}\!\left[\hat W - W\right],\qquad
\text{forward}=\hat W,\quad \frac{\partial W_{\text{ST}}}{\partial W}=I.$$

Thus the full-precision $W$ is **kept alive alongside the codebook throughout training** — the property
that later enables free un-clustering (§2.6). The codebook is trained with a per-matrix VQ-VAE objective:

$$\mathcal{L}_{\text{vq}} = \underbrace{\bigl\|\mathrm{sg}[W]-\hat W\bigr\|_2^2}_{\text{codebook: centroids}\to\text{weights}}
\;+\; \beta\underbrace{\bigl\|W-\mathrm{sg}[\hat W]\bigr\|_2^2}_{\text{commitment: weights}\to\text{centroids}},\qquad \beta=0.25.$$

### 2.5 A worked example (real weights from the trained model)

One real output row of matrix `L1.v`, its learned 2-bit codebook, and the result of snapping each weight
to its nearest anchor (first ten entries):

| weight $W_{i,c}$ | 0.007 | 0.022 | 0.009 | −0.036 | −0.027 | −0.049 | −0.011 | −0.004 | 0.017 | −0.007 |
|---|---|---|---|---|---|---|---|---|---|---|
| **2-bit** $\hat W$ (k=4) | 0.006 | 0.037 | 0.006 | −0.020 | −0.020 | −0.055 | −0.020 | 0.006 | 0.006 | 0.006 |
| **1-bit** $\hat W$ (k=2) | 0.075 | 0.075 | 0.075 | −0.081 | −0.081 | −0.081 | −0.081 | −0.081 | 0.075 | −0.081 |

Learned codebook for this row: k=4 → `{−0.055, −0.020, 0.006, 0.037}`; k=2 → `{−0.081, 0.075}`. Every
weight is replaced by the nearest allowed value — the entire matrix is representable by 4 (or 2) numbers
per row plus an index per weight.

### 2.6 The un-cluster toggle (free leave-one-out un-quantization)

Because the STE keeps $W$ alive, setting a component's quantize flag to false makes its forward pass use
$W$ directly — an **exact, retraining-free un-quantization**. Toggling it off for one component while all
others stay quantized is a clean causal *leave-one-out* probe, used for localization in §3 (F4) and as
the basis of the interpretability tool in §5.

### 2.7 Loss decomposition

Let $\text{ce}_t$ be the per-token cross-entropy and $s_t$ the skill label of target token $t$. The
per-skill loss and the causal quantization penalty for skill $s$ are

$$\mathrm{CE}_s=\frac{\sum_t \mathbb{1}[s_t=s]\,\text{ce}_t}{\sum_t \mathbb{1}[s_t=s]},\qquad
\Delta_s=\mathrm{CE}_s^{\text{target}}-\mathrm{CE}_s^{\text{control}}.$$

Because control and target share initialization and batches (§2.8), $\Delta_s$ is quantization's causal
effect on skill $s$. We report $\Delta_s$ and the ratio $\mathrm{CE}_s^{\text{target}}/\mathrm{CE}_s^{\text{control}}$.

### 2.8 Paired-lockstep training

We build one fp32 model, deep-copy it, and quantize the copy. Control and target then train from
**identical initialization on identical batches**, so any loss gap is attributable to quantization
alone. Control minimizes masked cross-entropy; the target minimizes masked cross-entropy plus
$\sum_{\text{matrices}}\mathcal{L}_{\text{vq}}$. Both use AdamW (lr $3\times10^{-4}$, weight decay 0.01),
batch 48, 10k steps; the curriculum raises the maximum level from L1 to L5 over the first 60% of
training. A single seed is used throughout (a limitation, §8).

## 3. Q1 & Q2 — what breaks, and where

**F1 — damage is skill-selective. Verdict: Q1 confirmed (Fig. `grouped.png`).** Converged per-skill
2-bit penalty (target − control CE): read/semantic/filter/index ≈ 0 (ratios ≤ 0.83×); **content +0.012
(1.25×)** and **relative +0.030 (1.61×)** are the only real penalties. Retrieval-type skills compress
for free; computation-type skills pay, and the deeper computation (relative, two-stage) exceeds the
shallow one (index).

![F1 — per-skill 2-bit penalty: computation pays, retrieval is free](runs/grouped.png)

**F2 — damage localizes to the quantity-emitting token (Fig. `exp1a.png`).** In relative traces the
entire difficulty concentrates on the token emitting the computed position (`pos N`): control CE spikes
~1000× there and is ~0 elsewhere, and the quant gap rides the same spike (a second emission → a second
spike). Once a position is written as a token, downstream steps are quant-free — the scratchpad launders
a fragile computed quantity into a robust symbol.

![F2 — difficulty and quant damage spike surgically on the `pos N` token](runs/exp1a.png)

**F4 — the fragile computation lives in the value/output path. Verdict: Q2 confirmed, with caveat
(Fig. `exp2a.png`).** Un-clustering by weight class shows `relative` dominated by the value/output
projections (~10× Q/K, intensifying OOD), concentrated in early layers; `content` is distributed;
lookup skills respond to no class. This matches a RoPE account: position is encoded only in Q/K routing
(consumed by the softmax) and reaches the MLP only after being written into the value stream as
content — so the value/output path carries the fragile position-content. *Caveat:* straight-through
co-adaptation confounds the *sign* of whole-class un-clustering; component-level leave-one-out is the
trustworthy granularity, where magnitude (not sign) is the reliable signal.

![F4 — the fragile computation localizes to the value/output (V/O) weight class](runs/exp2a.png)

## 4. Q3 — one mechanism: quantization is a brittle extrapolator

**F3 — regularizer in-distribution, brittle out-of-distribution. Verdict: Q3 confirmed
(Fig. `exp1b.png`).** The gap at the quantity-emitting token vs. list length is U-shaped in absolute
difficulty (minimum at the training mode); the two models cross exactly at the training distribution —
the quantized target is *better* than fp32 in-distribution (quantization regularizes) and progressively
*worse* out-of-distribution. Stated as the unifying rule:

> Quantization is free (even regularizing) in-distribution and damages an operation in proportion to how
> far the **quantity it must compute** extrapolates beyond training.

![F3 — regularizer in-distribution, brittle out-of-distribution; the models cross at the training range](runs/exp1b.png)

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

![E-a — the confounded result that *looked* like recall-fragility](runs/exp_ea.png)

![E-b — length-controlled refutation: recall-and-bind is the *most* robust, not the least](runs/exp_eb.png)

## 5. A quantization-based interpretability probe (H3-tool: validated)

Define a component's **bit-width elasticity** as the loss increase when it alone is crushed to 1-bit
(others at 2-bit). Hypothesis: elasticity is a functional fingerprint — discrete/lookup machinery
tolerates coarsening; analog/computation machinery does not.

**H1 (Fig. `exp_h1.png`).** Crushing a component hurts computation more than lookup for all seven weight
classes (ratios 1.8–10.2×; 35/42 components individually), and the elasticity map recovers F4's
value/output localization from a *different* probe. It is not a magnitude artifact (corr(elasticity,
weight-std) = −0.15).

![H1 — bit-width elasticity map: crushing a component hurts computation more than lookup](runs/exp_h1.png)

**H2 — validation against a non-quantization ground truth (Fig. `exp_h2.png`).** Gradient (Taylor)
saliency (∂L·W)² on the fp32 control model — no quantization anywhere — split by computation vs. lookup
loss: all 42 components are more computation-critical than lookup-critical, and per-component saliency
agrees with quant-elasticity at Spearman ρ = 0.55. The methods agree on the compute-vs-lookup axis but
weight classes differently — gradients emphasize Q/K (routing has steep *local* gradients), elasticity
emphasizes V/O (payload is least tolerant to *coarse* rounding). **Verdict: the probe is validated and
non-redundant** — correlated with, but not reducible to, gradient importance; it captures
discrete-tolerance structure infinitesimal gradients miss. Un-clustering additionally provides an
on-manifold ablation primitive, cleaner than zero-ablation.

![H2 — the elasticity map agrees with independent gradient importance on the fp32 model (ρ=0.55)](runs/exp_h2.png)

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

## 7. Trustworthy-AI impact (alignment with the OECD AI Principles)

Compression is a *governance blind spot*: applied ubiquitously for efficiency, its effect on specific
capabilities goes undocumented. Our method and probe supply evidence toward three of the five OECD AI
Principles. (The Principles are values, not a certification; the concrete home for a tool like this is
the OECD.AI *Catalogue of Tools & Metrics for Trustworthy AI*, under transparency and robustness.)

| OECD Principle | What our work contributes | Evidence |
|---|---|---|
| **Transparency & Explainability** | Circuit-level localization: *which weights load-bear which skills*, and *which capabilities a compression step changed and where* — a question no aggregate benchmark answers | F1, F2, F4; H1–H2 (the probe) |
| **Robustness, Security & Safety** | Predicts the *failure surface* before deployment: computation and out-of-distribution/long-context reasoning fail first, while recall is safe — enabling pre-deployment risk assessment aggregate accuracy hides | F1, F3 |
| **Accountability** | Localized, documented capability-deltas provide audit-trail evidence for model cards and impact assessments (what changed, attributable to which circuits) | F4, H1 |

**A concrete governance example.** A public agency plans to deploy a *quantized* language model to help
triage benefit applications. Before deployment it runs the probe and finds that multi-step reasoning
over long case histories is a fragile, computation-heavy circuit that degrades under 2-bit compression,
while factual recall is untouched. The trustworthiness assessment can now document: *compression is safe
for lookup-style tasks but degrades long-context reasoning; keep the reasoning-critical weight path in
higher precision (function-guided mixed precision); route complex cases to the full-precision model.*
That is a transparency + robustness + accountability decision the agency can defend — enabled by the
tool, not by aggregate accuracy.

**Honest bound.** This supports the *transparency and robustness* slice of trustworthiness; it is not a
fairness or societal-impact measure, and it is validated on a testbed (see §8). As a *tool aligned with*
the OECD Principles it is defensible today; as a *deployed governance instrument* it awaits the
real-model replication.

## 8. Limitations and scope

We state these plainly. The testbed is a 4.32M synthetic model, one seed, one bit-width (k=4), one
quantizer (our VQ, not GPTQ/AWQ). Absolute CE effects are small because the task is fully learned — the
signal is in ratios, localization, and controlled contrasts, not raw magnitude. The probe is a
per-component scalar (coarse), with leave-one-out as its clean causal granularity. **The single
strongest strengthening step is a real-model replication** (Pythia-160M / Qwen2-0.5B with GPTQ) testing
whether skill-selectivity, value-path localization, and elasticity-based functional mapping transfer;
we position this as the direct route from a controlled mechanistic result to production generality, and
predict (P1–P4) what it should find.

## 9. Conclusion

Three verdicts. Low-bit quantization damages *computation* and spares *retrieval* (Q1), in proportion to
how far a computed quantity extrapolates beyond training (Q3), localized to the value/output path and
the quantity-emitting tokens (Q2). A stronger recall-specific hypothesis was refuted by a controlled
experiment. The same sensitivity, read as a map, is a *validated* interpretability probe for separating
computation from lookup machinery. Both the finding and the tool point to one actionable target: protect
the computation path, and let the model write its fragile quantities down.

## 10. Glossary — abbreviations and notation

*Everything abbreviated in this paper, in plain language.*

**Quantization and training**

| Term | Meaning |
|---|---|
| **quantization** | Compressing a model by rounding each weight to one of a small set of allowed values (fewer bits per weight). |
| **CE** | Cross-entropy — the standard next-token training/evaluation loss. Lower = the model is less "surprised" = better. |
| **bit / k / bit-width** | A codebook of `k` allowed values costs `log₂k` bits per weight. `k=4` → 2 bits; `k=2` → 1 bit. |
| **fp32** | 32-bit floating point — full precision. Our uncompressed "control" model. |
| **VQ** | Vector quantization — replacing each weight with the nearest entry in a small learned *codebook*. |
| **codebook / centroid / anchor** | The small set of allowed values (here, per matrix-row) that weights are snapped to. |
| **VQ-VAE** | Vector-Quantized Variational Auto-Encoder — the origin of the *codebook + commitment* loss we use (§2.4). |
| **STE** | Straight-Through Estimator — a trick that lets gradients flow through the (non-differentiable) rounding step, keeping the full-precision weights trainable. |
| **sg[·]** | Stop-gradient — an operation that blocks gradients from flowing (used inside the VQ loss). |
| **QAT** | Quantization-Aware Training — training with quantization simulated in the forward pass (our setting). |
| **PTQ** | Post-Training Quantization — quantizing a finished model without further training (e.g. GPTQ, AWQ). |
| **β (beta)** | Weight on the commitment term of the VQ loss (we use 0.25). |
| **un-clustering** | Turning quantization *off* for one component so it uses its full-precision weight — free because the STE kept that weight alive (§2.6). |

**Named prior quantization methods** (cited for context in §1 Positioning)

| Term | Meaning |
|---|---|
| **GPTQ** | A widely used one-shot post-training weight quantizer. |
| **AWQ** | Activation-aware Weight Quantization — protects the weights that see large activations. |
| **BitNet** | An approach to training LLMs at ~1 bit per weight. |
| **GPTVQ** | A post-training *vector*-quantization method. |
| **AQLM** | Additive Quantization for Language Models — multi-codebook PTQ. |
| **HAWQ** | Hessian-AWare Quantization — assigns bit-widths by second-order sensitivity (related measurement to our probe, different framing). |

**Model architecture**

| Term | Meaning |
|---|---|
| **LLM** | Large Language Model. |
| **MLP** | Multi-Layer Perceptron — the feed-forward block; here its three maps are *gate, up, down*. |
| **Q, K, V, O** | The Query, Key, Value, and Output projection matrices inside attention. |
| **GQA / KV heads** | Grouped-Query Attention — fewer Key/Value heads than Query heads (8 Q / 4 KV here). |
| **RoPE** | Rotary Position Embedding — encodes token position by rotating Q and K (not V — central to §3's mechanism). |
| **RMSNorm** | Root-Mean-Square normalization. |
| **SwiGLU** | A gated feed-forward activation (Swish gate). |
| **d / head dim** | Hidden width (256) / per-head width (32). |

**Analysis and interpretability**

| Term | Meaning |
|---|---|
| **OOD** | Out-Of-Distribution — inputs beyond the training range (here, lists longer than seen in training). |
| **`pos N`** | A scratchpad token where the model writes a computed *position* as a number (e.g. `pos 4`). The hardest, most fragile step (§3). |
| **SAE** | Sparse Autoencoder — the mainstream interpretability method, which operates on *activations* (we operate on *weights*). |
| **bit-width elasticity** | Our probe: how much the loss rises when one component alone is crushed to 1 bit. High = it does fragile computation (§5). |
| **Taylor / gradient saliency** | `(∂L·W)²` — a standard importance measure, used as our independent (non-quantization) validation (§5, H2). |
| **ρ (rho)** | Spearman rank correlation — agreement between two rankings (1 = identical order). |
| **Δₛ (Delta)** | The quantization penalty for skill *s* = (target CE − control CE) on that skill's tokens (§2.7). |

**Labels used in this paper**

| Term | Meaning |
|---|---|
| **Q1 / Q2 / Q3** | The three research questions (what breaks / where / when). |
| **F1–F4** | The four main findings (skill-selectivity / token localization / regularizer-brittle / value-path). |
| **E-a, E-b** | The experiment pair that tested and *refuted* the "recall-and-bind" hypothesis. |
| **H1, H2** | The interpretability-tool experiments (elasticity map; non-quantization validation). |
| **P1–P4** | The four testable predictions for production LLMs (§6). |
| **L1–L5** | Curriculum difficulty levels (bands of list length). |
| **the six skills** | read, semantic, filter, index, content, relative (defined in §2.1). |

**Policy / governance**

| Term | Meaning |
|---|---|
| **OECD** | Organisation for Economic Co-operation and Development — the intergovernmental body whose *AI Principles* define trustworthy-AI values. |
| **OECD AI Principles** | Human-centred, transparent, robust/safe, accountable, inclusive AI (§7). |
| **OECD.AI Catalogue of Tools & Metrics** | The OECD's public repository of practical trustworthy-AI tools — the concrete home for a probe like ours. |

---
*Figures: `grouped.png` (F1), `exp1a.png` (F2), `exp1b.png` (F3), `exp2a.png` (F4), `exp_ea.png`/`exp_eb.png` (E-a/E-b, H2-recall refuted), `exp_h1.png`/`exp_h2.png` (H3-tool). Reproduce from `ckpt.pt` with `exp_*.py`.*
