# LLM Phase Transition

Mapping where language model reliability collapses as context gets noisier — borrowing phase-transition framing from physics to describe the breakdown.

![phase diagram](assets/phase_diagram.png)

The heatmap shows accuracy as a function of distractor density × gold passage position. The curves on the right are sigmoidal — there's a sharp transition point, not a gradual linear degradation. Middle placement degrades fastest, consistent with the Lost-in-the-Middle effect.

## Notebooks

**01 — Entropy Probe**
Does first-token Shannon entropy increase when the model receives contradicting context? Main finding: poisoned context doesn't make the model uncertain — it makes it confidently wrong. Entropy drops, not spikes.

**02 — Attention & Position**
Mechanistic verification of the position effect using GPT-2 attention weights. Tracks how much attention the model pays to the gold passage depending on where it sits in the context window.

**03 — Agent Failure Taxonomy**
ReAct agent tested across three conditions: clean wiki, wrong-fact injection, and adversarial prompt injection. Key finding: wrong facts map to the distractor density axis (high-salience noise), while injections map to the adversarial intent axis — the attacker shifts not just whether an error occurs, but which error.

**04 — Evaluation Pipeline**
Async multi-model eval harness with response caching, baseline run on clean factual QA. Key finding: metric choice massively affects reported accuracy — F1 + substring is the right combination for the phase diagram experiments.

**05 — Phase Diagram**
Sweeps distractor density × gold-passage position on fictional entities to force passage-reading. Key finding: accuracy collapse is sharp and logistic — n* drops as the gold passage moves toward the end, with start position the most robust due to primacy bias.

**06 — Steering Vectors**
Tests whether a memory-vs-context steering vector can shift n* upward, expanding the safe operating region. Key finding: the circuit is not linearly steerable at the residual stream — accuracy is flat across all α values at every layer. Attention ablation experiments show the distractor signal enters via specific heads, not the residual stream directly.

## Stack

- `transformers`, `transformer_lens`
- `torch`, `pandas`, `matplotlib`, `scipy`

## Status

Active. SAE feature analysis coming next.
