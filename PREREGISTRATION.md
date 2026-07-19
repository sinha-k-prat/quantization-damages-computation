# Pre-registration — experiments, predicted outcomes, and decision rules

*For each experiment: what a positive result shows, what a negative shows, and the exact metric/threshold
that decides. "Unsuccessful" is defined by a threshold, not judgment. (Reviewer R1's "pre-register how it
could be wrong.")*

## A. NeurIPS-critical (need GPU)

**A1 · Seeds on the toy** — are the per-skill penalties real or seed-noise?
- ✅ Success: relative & content gaps stay positive across 3–5 seeds → selectivity is real.
- ❌ Fail: gaps straddle 0 or flip sign → the central number was seed-noise.
- 📏 Decide: mean±std of Δₛ across seeds. **Success = 95% CI excludes 0 (mean>2·std)** for relative & content.

**A2 · Real-model GPTQ replication** — does "computation degrades more" hold with a real quantizer? *(the make-or-break)*
- ✅ Success: computation > copy > language degradation, and model still works → transfers to real LMs (R2).
- ❌ Fail: ratios ~equal across token types → selectivity is a toy/VQ artifact.
- 📏 Decide: ratio (quant CE / fp CE) per token-type over ≥2000 tokens each. **Success = computation ratio ≥1.5× copy ratio, non-overlapping bootstrap CIs.**

**A3 · Bit-width sweep (real model, 8/4/3/2-bit GPTQ)** — does selectivity sharpen as bits drop?
- ✅ Success: (computation − copy) gap grows monotonically as bits fall → dose-response.
- ❌ Fail: gap flat/non-monotonic → doesn't scale with compression (weaker, not fatal).
- 📏 Decide: plot (computation − copy) ratio-gap vs bits. **Success = monotone, ≥3 of 4 steps correct direction.**

**A4 · EM-selectivity (real model, GPTQ)** — does the model fail MORE on computation-heavy problems while staying fluent?
- ✅ Success: EM drop concentrated on high-arithmetic problems; recall/low-arithmetic preserved.
- ❌ Fail: uniform drop or total collapse (crude VQ already collapsed to 0 → EM uninformative without GPTQ).
- 📏 Decide: bin GSM8K by #arithmetic-steps; regress (fp EM − quant EM) on step-count. **Success = positive slope, p<0.05.**

## B. Tool / localization (need GPU + PTQ specifically)

**B1 · Granular row/col/subspace localization (PTQ)** — is the fragile computation low-rank inside the matrix?
- ✅ Success: restoring a small fraction (rows/cols/rank) recovers most of the penalty → low-dimensional; enables surgical mixed-precision.
- ❌ Fail (diffuse): needs near-full restoration. ❌ Fail (blocked): no recovery at all — **seen under QAT (co-adaptation); this is why it needs PTQ.**
- 📏 Decide: fragile-CE recovery vs fraction restored (subspace mode). **Success = ≥90% recovery at ≤20% restored.** Fail = >60% needed.

**B2 · PTQ ablation on the toy (R3)** — is the split real quantization physics, or specific to our learned VQ?
- ✅ Success: same computation>lookup ordering under rounding-PTQ → quantizer-agnostic.
- ❌ Fail: vanishes/reverses under PTQ → VQ artifact; scope claims to VQ.
- 📏 Decide: per-skill penalty under GPTQ-style PTQ on the toy. **Success = same sign & ordering as QAT.**

## C. Mechanistic circuits (CPU/checkpoint; C3/C4 need retrain)

**C1 · Locate-then-type 2×2 (L1:SA fetch head)** — is L1:SA causally the fetch, and the robust-lookup type?
- ✅ Success: patching L1:SA flips the answer AND un-clustering it barely moves CE → "attention-fetch = robust" cell confirmed.
- ❌ Fail: patch doesn't flip → L1:SA isn't the fetch head; re-locate.
- 📏 Decide: (a) patch-flip rate; (b) |ΔCE| from un-clustering L1:SA. **Success = flip >70% AND |ΔCE| < value-path MLP's.**

**C2 · Retroactive-fetch control (list-position vs `=` probe)** — is fetch built retroactively at the working position, not during reading? *(cheap, CPU, run now)*
- ✅ Success: target-location decodable at `=` (~0.9) but chance at list positions → retroactive fetch confirmed.
- ❌ Fail: decodable at list positions too → baked in during reading (contradicts causal-order story).
- 📏 Decide: probe accuracy at a list position vs `=`. **Success = list-position ≤0.6 (≈chance 0.5), `=` ≥0.85.**

**C3 · Repeated-entity "pull all X"** — does fetch aggregate all occurrences, and does an MLP do the rank? *(needs retrain)*
- ✅ Success: L1:SA attends to all X; ablating a downstream MLP collapses rank.
- ❌ Fail: attention picks one, or no MLP-rank dependence → "pull all X, MLP ranks" is wrong.
- 📏 Decide: attention mass all-X vs top-1; MLP-ablation rank drop. **Success = >60% mass spread AND MLP-ablation drops rank EM >20 pts.**

**C4 · Task-first retrain** — does query-before-list create a streaming-filter circuit vs retroactive fetch? *(needs retrain)*
- ✅ Success: "which items matter" decodable at list positions during reading in task-first (chance in task-last).
- ❌ Fail: same retroactive circuit → task order cosmetic.
- 📏 Decide: relevance-probe accuracy at list positions, task-first vs task-last. **Success = task-first ≥0.8, task-last ≤0.6.**

## Priorities
- **Can make/break the paper:** A1 (is the toy real?) and A2 (does it transfer?). If A2 fails, it stays a toy study.
- **Cheapest / runnable now:** C2 (CPU).
- Everything in A/B needs GPU; C3/C4 need a retrain.
