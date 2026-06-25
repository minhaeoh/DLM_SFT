# Step ≠ Thinking: Diagnosing and Repairing the Reasoning Deficit of Masked Diffusion Language Models
 
## 0. One-line thesis
 
Current masked DLMs (LLaDA) do **not** convert denoising steps into sequential reasoning time. We show this is a *structural* consequence of conditional-independence + irreversible commitment (the Parallel–Sequential Contradiction), mechanistically dissect *why* it happens at the representation level, and propose a minimal, interpretability-derived **internal-trigger remasking mechanism** so that "step = thinking" can hold.
 
---
 
## 1. Core Objective (revised)
 
The original framing assumed `step = thinking = AR CoT length` and sought to **confirm** it. Both Chen et al. (2025, arXiv:2510.09544, the "PSC" paper) and our pilot results falsify the strong version. We therefore pivot from a confirmation study to a three-part argument:
 
1. **Phenomenon** — demonstrate that the diffusion-step axis fails to act as sequential reasoning time, especially as difficulty rises.
2. **Mechanism** — explain *why*, at the representation level, using our interpretability probe suite.
3. **Method** — sketch a mechanistically-motivated fix that lets steps become thinking, and position it precisely against the (already active) remasking literature.
This restructures the contribution: the **mechanistic analysis is the moat**; the method falls out of it as a natural consequence rather than as yet another remasking heuristic.
 
---
 
## 2. Motivation / Why the Pivot
 
- **PSC (Parallel–Sequential Contradiction).** Parallel decoding conflicts with the causal order needed for rigorous reasoning. DLLMs show genuine parallelism only for *directly-decidable* outputs and revert to AR-like behavior as difficulty rises; AR-prompting nearly doubles decoding steps without quality gain; PSC restricts self-reflection, reasoning depth, and exploratory breadth. (Chen et al. 2025). This paper diagnoses (does not solve) the problem and gives us a named handle for the Phenomenon layer.
- **The LRM lesson.** The explosive gains of DeepSeek-R1 / Qwen-style LRMs come largely from an *internalized* error-correction mechanism — the model detects its own error without an external signal and revises mid-reasoning ("wait" → reconsider) — not from new knowledge. Within known mathematics, both AR and DLM solve comparably; the gap is self-correction.
- **The DLM correction problem.** A DLM has *neither* form of correction:
  - **Forward-additive** (AR-style: write "wait", then condition future tokens on it) — weak, due to fixed canvas + parallel independence.
  - **In-place** (diffusion-native: remask a wrong token, re-predict under fuller context) — *structurally disabled* because confirmed tokens are frozen in standard LLaDA inference.
---
 
## 3. Hypotheses
 
- **H1 — Independence (root of PSC).** Per-step parallel confirmations are sampled conditionally independently from their position marginals, so the sequential dependency required for multi-step reasoning is broken.
- **H2 — No native self-correction.** Because confirmed tokens are frozen, the DLM-native correction (remask + re-predict) is disabled; combined with H1, the model can express *neither* correction mode.
- **H3 — Latent capacity exists but is discarded (the bridge to method).** Under fuller context, the model often *already holds* the information needed to correct a wrong committed token, but the decoding scheme throws it away. If true, the fix is a control problem (when/where to remask), not a knowledge problem.
---
 
## 4. Phenomenon Layer (P): Show that step ≠ thinking
 
**P1 — Scaling dissociation.** Vary the three DLM scaling axes (parallel / diffusion / sequential) and show that diffusion and sequential scaling saturate under difficulty (consistent with PSC) while parallel scaling does not purchase reasoning depth. Extends PSC with a mechanistic readout rather than re-deriving it.
 
**P2 — ★ Counterfactual Remasking (flagship).** Take *confirmed* trajectories; at later steps, re-score the already-confirmed positions under the current full context.
- **Metric:** fraction of confirmed tokens that would flip to a *more-correct* value if revision were permitted ("wasted-correction rate"); correlate with final answer correctness and with difficulty.
- **Payoff:** a high wasted-correction rate simultaneously proves (a) latent self-correction capacity (H3) and (b) that the decoding scheme discards it by design.
- **Critical control:** separate *"could fix but doesn't"* (preservation bias — see D3IM/SCOPE) from *"cannot fix"* (knowledge genuinely absent). This control is what makes P2 a mechanism claim, not just a behavioral one.
---
 
## 5. Mechanistic Layer (E1–E5, reframed): Explain *why*
 
Each probe is reframed from *confirming* step=thinking to *explaining* its failure.
 
**E1 — Step-wise Logit Lens / Commitment Gap.** Track the answer-token logit trajectory and convergence step `t*`. *New target:* at each confirmed position, measure the gap between **commit-time belief** and **final-context belief**, showing that confirmed tokens lock in early and fail to track later evidence — the representational face of irreversibility. Log absolute + normalized (`t*` / actual steps) counts and the EOS-arrival step.
 
**E2 — Information-Theoretic Signature of PSC.** Step-wise entropy reduction and MI (MINE estimator), interpreted for **relative monotonicity**, not absolute values. *New target:* test whether MI gain stalls or becomes non-monotone precisely on dependency-heavy items — the information-theoretic fingerprint of the independence failure (H1).
 
**E3 — Layer Dynamics & the Trigger Substrate.** Early-step→early-layer (planning) vs. late-step→late-layer (finalization) shift. *New target:* does this shift break under PSC, and does **cross-layer logit-lens disagreement** at a position predict that the committed token is wrong? (→ direct candidate trigger signal for the method.)
 
**E4 — Causal Interventions.**
- *Step truncation:* early stop → accuracy drop.
- *Hidden-state patching:* inject late-stage hidden states into early trajectories; pre-registered **layer × position grid** sweep given known sensitivity.
- *Frozen-token test:* show that without remask, an injected correction cannot propagate to an already-committed position — the causal demonstration of H2.
**E5 — Correction Dynamics (centerpiece).** Redefine from *"does correction happen"* to *"is correction structurally suppressed."* Track token flips and hidden-state realignment; distinguish **proactive** (silent internal flip) vs. **reactive** (text-triggered) correction. *New target:* mechanistically dissect **preservation bias** — *why* does the model reproduce its own wrong committed token rather than correct it? (Note: re-masking of confirmed tokens is impossible by design in standard LLaDA decoding, which shapes how E5 must be measured — via counterfactual re-scoring rather than observed flips.)
 
---
 
## 6. Method Sketch: an internal-trigger remasking loop
 
**Goal:** make `step = thinking` by making steps *revisable*, gated by an internal inconsistency signal — the DLM analog of "wait."
 
**Trigger candidates (derived from E1/E3):**
1. Commit-time vs. current-context confidence divergence (KL / probability drop on a committed position).
2. Cross-layer logit-lens disagreement at a committed position.
3. Entropy resurgence — a position whose uncertainty rises again after having dropped.
**Correction loop:** `localize → remask flagged positions → re-denoise under full context`.
 
**The hard part is localization / credit assignment** ("which earlier token is the wrong one"). Our probes supply that signal; *this* is the novel lever, not remasking per se.
 
**Positioning (the space is crowded — differentiate, don't reinvent):**
- **ReMDM** (Wang et al. 2025): training-free remasking sampler, marginal-preserving (usable on pretrained models), but no error-identification — remasks stochastically.
- **RemeDi** ("Don't Settle Too Early", arXiv:2509.23653): confidence-based remask + remask-aware SFT/RL. Closest to a full learned method.
- **D3IM / SCOPE** ("Revise, Don't Freeze", arXiv:2606.01026): parameter-free visible-to-visible revision; surfaces **preservation bias**.
- **RDD** (Reversible Diffusion Decoding, arXiv:2602.00150): introduces reversibility against block-level error propagation from irreversible commitment.
- **Our angle:** a *mechanistically-derived minimal trigger* justified by interpretability (E1/E3) rather than a heuristic confidence score — plus the first mechanistic explanation of the preservation bias these papers observe but do not open up.
---
 
## 7. Success Criteria (reframed)
 
The thesis is supported if:
- **Phenomenon:** counterfactual remasking (P2) shows a substantial wasted-correction rate that scales with difficulty, and diffusion/sequential scaling dissociates from reasoning gain (P1).
- **Mechanism:** committed-token belief fails to track later context (E1); MI gain stalls on dependency-heavy items (E2); and a specific internal signal (E1/E3) predicts beneficial flips above a confidence-only baseline.
- **Method:** gating remasking on that signal recovers accuracy / converts steps into reasoning depth on a pilot, beating confidence-only remasking — turning the discarded latent capacity (H3) into realized correction.
---
 
## 8. Experimental Setup & Confounds (carried over / updated)
 
- **Model:** LLaDA-8B (ShortCoT vs. LongCoT SFT). Fixed 4096 canvas.
- **Confound controls:** truncation fairness (LongCoT not penalized by the 4096 limit); dip-and-recovery (evaluate best-epoch checkpoints).
- **Decoding for probes:** token-per-step (steps/block = 32, block size 32) preserves within-step conditioning and gives clean attribution. *Note:* cheaper steps/block reintroduces the H1 independence error — keep this in mind when interpreting counterfactual remasking, and prefer the token-per-step setting for the phenomenon/mechanism experiments even if the broad benchmark sweep uses a reduced setting.
- **Compute:** calibration-first remains (reduced steps/block + stratified MATH500 for the sweep), but the P/E attribution experiments should run at token-per-step on a small, difficulty-stratified subset rather than the full sweep.
---

