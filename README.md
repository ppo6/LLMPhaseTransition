# LLM Phase Transition

Exploring how language model reliability degrades as context becomes noisier — borrowing phase-transition framing from physics to describe the collapse.

## What's here

**01_entropy_probe** — does first-token Shannon entropy increase when the model receives context that contradicts its parametric knowledge? Testing entropy as a cheap real-time signal for distractor pressure.

Main finding: poisoned context doesn't make the model uncertain — it makes it confidently wrong. Entropy drops, not spikes.

## Stack

- `transformers` (Qwen2-0.5B-Instruct)
- `torch`, `pandas`, `matplotlib`

## Status

Work in progress. Adding attention analysis and full phase diagram sweep next.
