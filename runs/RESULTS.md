# Retrieval quantization study — consolidated results (K=4, 10k-step single-seed run)

Model: MiniQwen M (d=256, 6L, 4.32M params), VQ k=4 (2-bit), paired control(fp32)+target(VQ).
Checkpoint: `runs/ckpt.pt`. Raw logs: `runs/logs/`. Curriculum: L1(len3-5)…L5(len9-12); OOD=len13-24.

## THESIS (after E-b refuted the recall reframe)
Quantization is a **brittle extrapolator**: free/regularizing in-distribution, damages an operation in
proportion to how far the QUANTITY it must compute extrapolates beyond training. Skill-selectivity follows
(bounded-output lookup skills = free; growing-output computation skills = fragile OOD).

## F1 — per-skill penalty (target−control CE, mean last 4 L5 evals, `grouped.png`)
read −0.0001 (0.65×) · semantic −0.0001 (0.53×) · filter −0.0001 (0.63×) · index −0.008 (0.83×)
→ all ~ZERO (retrieval/lookup, bounded output).
content +0.012 (1.25×) · relative +0.030 (1.61×) → the only penalties (computation, growing output).

## F2 — localization to `pos N` token (`exp1a.png`)
Control CE & quant gap both spike ~1000× on the `pos N` token, flat elsewhere; 2nd spike on 2nd `pos N`.
Difficulty & damage are surgically on the position-computation token.

## F3 — regularizer in-dist / brittle OOD, monotonic in length (`exp1b.png`)
Gap at `pos N` vs list length: U-shaped abs CE (min at train mode len~9); models CROSS at train dist.
Gap −3.8 (len5) → 0 (len~9) → +1.4 (len21). VQ better short/in-dist, worse & diverging OOD.

## F4 — weight-class localization (`exp2a.png` in-dist, `exp2a_ood.png` OOD)
relative dominated by V/O (value/output) path: in-dist |−0.042| vs Q/K |−0.004| (10×); OOD |−0.103| vs
Q/K |−0.003| (30×), intensifies OOD. content spread across classes. filter/semantic/read ~0 everywhere.
Component-level (3a): early-layer (L0) dominant. CAVEAT: STE/QAT co-adaptation confounds the SIGN at
class level (un-clustering a whole class breaks co-adaptation) → magnitude=load-bearing, sign unreliable at
class granularity; component-level LOO is the trustworthy causal read.

## E-a — implicit(recall) vs explicit(identity) id at the identification token (`exp_ea.png`)
EXPLICIT (pos of named anchor): in-dist +0.030, OOD −0.174.  IMPLICIT (shape-selected obj): in-dist +0.005, OOD +0.525.
LOOKED like recall-fragility BUT confounded by extrapolation distance (T4 pushed to len13-20). See E-b.

## E-b — CONTROLLED (each query at NATIVE length), refutes recall reframe (`exp_eb.png`)
OOD gap: T1 position +0.85 · T3 filter+pos +0.50 · T4 recall+bind +0.09  (in-dist all ~0).
Recall+bind is MOST robust. Gap tracks OOD control CE (6.66/1.95/0.47) = extrapolation distance of the
COMPUTED QUANTITY (T1 counts to unseen positions; T4 ordinal stays 1st-3rd), NOT recall. F5 RETRACTED.

## STATUS
Mechanism half complete on toy. NeurIPS main ~5% as-is (scale/single-seed/no-real-quantizer/diagnostic-not-method);
strong workshop. Bridge to main track = real-model replication (Pythia-160M/Qwen2-0.5B + GPTQ). TODO: multi-seed
error bars (4a), bit-width sweep (4b), periodic-ckpt+resume in trainer, real-model replication, CoT-as-robustness test.
