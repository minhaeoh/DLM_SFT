# Concrete Experimental Plan — Step ≠ Thinking (LLaDA-8B)

*Companion to `DLM_Reasoning_Objective_v2.md`. Phenomenon (P), Mechanism (E1–E5), and Method follow the layers defined there.*

---

## 0. Checkpoint inventory

LongCoT/ShortCoT × SFT data scale {4k, 8k, 16k, 32k, 64k, 128k} × epochs 1–8 = **2 × 6 × 8 = 96 checkpoints**.

---

## Phase 0 — Checkpoint selection (before any P/E work)

Collapse the 96-checkpoint grid first; do not run mechanistic probes across the full grid.

> **Note on cheap setting (steps/block = 8):** A prior calibration experiment on MATH-500 (N=16, LongCoT gen=4096) showed that steps=2048 (steps/block=16) causes catastrophic generation collapse — `</answer>` infinite loop, 6.2% vs 37.5% at full steps. steps/block=8 (→ 1024 total steps) is expected to be worse. Using it for ShortCoT only would introduce an unfair asymmetry between the two CoT types. Therefore, both ShortCoT and LongCoT use token-per-step (steps/block = 32) throughout Phase 0.

- **0a. Best-epoch selection.** For each (CoT type × data scale) pair, pick the best-epoch checkpoint on a stratified MATH500 subset (N=32; difficulty levels 1–5, ~6–7 per level). Controls the dip-and-recovery confound.
  - **Pre-filter with eval_loss:** From each run's `trainer_state.json`, rank all 16 checkpoints by eval_loss and evaluate only the top-2 (LongCoT) or top-3 (ShortCoT) candidates per scale. This collapses 96 → 12 (LongCoT) + 18 (ShortCoT) = 30 eval targets.
  - **Decoding setting:** token-per-step (steps/block = 32) for both. ShortCoT: gen=1024, steps=1024. LongCoT: gen=4096, steps=4096.
  - **Execution:** run 2 independent single-GPU jobs in parallel (one per H200) to halve wall-time.
  - **Time budget:** ShortCoT 18 ckpts × ~30 min ≈ 9h (wall ~4.5h with 2-GPU parallel); LongCoT 12 ckpts × ~7h ≈ 84h (wall ~42h, 3 overnight runs of 2 parallel jobs each).
  - Output: **12 surviving checkpoints** (2 × 6).

- ~~**0b. Rank-consistency calibration.**~~ *Omitted.* A prior calibration experiment already established that no cheap setting exists for LongCoT (steps=2048 collapses to 6.2%), and the fairness constraint requires the same decoding setting for both CoT types. Both run at token-per-step throughout, so rank-consistency between cheap and full settings is not a concern.

- **0c. Final probe set.** From the 12, choose a 2×3 grid: both CoT types × {small = 4k, mid = 32k, large = 128k}. **6 checkpoints** carried into P/E. The scale axis tests whether the phenomenon weakens with more SFT.

---

## Phase 1 — Phenomenon (P): show step ≠ thinking

### P1 — Scaling dissociation
On the 6 checkpoints, sweep the three DLM axes independently on the stratified subset:

- *diffusion:* steps/block ∈ {8, 16, 32, 64}, fixed length
- *sequential:* gen length ∈ {1024, 2048, 3072}
- *parallel:* block size ∈ {16, 32, 64}
Plot accuracy vs. each axis, stratified by difficulty level. Prediction: diffusion & sequential saturate as difficulty rises; parallel buys no reasoning depth. Log forward-pass count per run as the true cost axis.

### P2 — Counterfactual remasking (flagship)
- Run clean token-per-step trajectories; at each confirmed position, cache commit-step hidden states + logits.
- At a late step (e.g. 0.8·T), re-score every confirmed position under the *current full context*. A token is "wasted-correctable" if the re-scored argmax differs and is closer to ground truth.
- **Metric:** wasted-correction rate; correlate with (final correctness, difficulty level).
- **Mechanism control:** separate "could fix but didn't" (re-scored argmax correct → preservation bias, supports H3) from "cannot fix" (re-scored argmax still wrong → knowledge absent). Report as two separate rates. Only the first supports H3.

---

## Phase 2 — Mechanism (E1–E5)

Same 6 checkpoints, stratified subset, token-per-step.

- **E1 — Commitment gap.** Per confirmed position: commit-time belief vs. final-context belief (KL + top-1 prob drop), plus t*, normalized t*, EOS-arrival step. Prediction: gap grows with difficulty; early commits fail to track later evidence.
- **E2 — Info signature.** Step-wise entropy + MI (MINE), read for relative monotonicity. Target: MI gain stalls / non-monotone on dependency-heavy items.
- **E3 — Layer dynamics / trigger substrate.** Early-step→early-layer vs. late-step→late-layer shift. Key deliverable: quantify whether **cross-layer logit-lens disagreement** at a position predicts the committed token is wrong (AUROC for "committed token wrong"). This is the candidate trigger signal for the method.
- **E4 — Causal interventions.** (a) step truncation → accuracy curve; (b) hidden-state patching with the pre-registered **layer × position grid**; (c) frozen-token test: inject a correction, show it cannot propagate to an already-committed position → causal demonstration of H2.
- **E5 — Correction dynamics.** Real remasking is impossible in LLaDA, so measure via counterfactual re-scoring (not observed flips). Distinguish proactive (silent internal flip) vs. reactive (text-triggered); dissect *why* the model reproduces its own wrong committed token (preservation-bias mechanism).

---

## Phase 3 — Method pilot

On 1–2 checkpoints, gate remasking on the E1/E3 internal signal (commit-vs-context divergence, cross-layer disagreement, entropy resurgence) via `localize → remask → re-denoise`. **Baseline to beat:** confidence-only remasking (ReMDM-style). Success = recovers accuracy / converts steps into reasoning depth above that baseline on the pilot subset.

---

## Open design decisions

- **Difficulty stratification:** native MATH500 levels 1–5, or re-stratify by measured per-checkpoint pass rate.
- **P/E checkpoint count:** 6 (2 CoT × {4k, 32k, 128k}) keeps it tractable; drop to 4 (2 CoT × {4k, 128k}) if compute is tight.
