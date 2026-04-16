# Support-Constrained Generation for Syllogistic Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen2--1.5B-orange)](https://huggingface.co/Qwen/Qwen2-1.5B)

This repository provides a minimal demonstration of **support-constrained generation** inspired by the paper *"Support-Constrained Variational Meta Chain-of-Thought: A Nullity-Semantics Framework"* (Yakunin, 2026). It shows how a standard causal language model (Qwen2-1.5B) assigns non‑zero probability to logically impossible conclusions, and how a simple logical mask eliminates this "null leakage" while preserving the model's relative preferences among valid continuations.

## What Does This Experiment Test?

The experiment evaluates a baseline language model on simple Aristotelian syllogisms and contradictory premises. For each example, we compute the exact log-probability of several candidate conclusions under the model and then apply a hard **logical mask** $\chi(c) \in \{0,1\}$ that zeroes out any continuation that is logically incompatible with the premises.

Two key metrics are reported:
- **Null mass (baseline)** – the total probability mass that the unmasked model assigns to **invalid** conclusions.
- **Deadlock rate (masked)** – the fraction of examples where **no** valid continuation exists (i.e., all candidates are masked out).

The masked distribution is guaranteed to have **exactly zero probability** on forbidden tokens – a direct implementation of the *nullity* principle from Brusentsov's ternary logic.

## How It Works

1. **Syllogism generation** – Synthetic prompts of the form  
   `All A are B. All B are C. Therefore:` are created, with one logically correct completion and several incorrect ones.  
   A fraction of examples use contradictory premises (e.g., `A is true. Not A is true. Therefore:`) where **no** valid conclusion exists.

2. **Exact log-probability computation** – Using Hugging Face Transformers and `Qwen/Qwen2-1.5B` on CPU, we compute the conditional log-likelihood of each candidate continuation by summing the log-probabilities of all its tokens (teacher-forcing style).

3. **Masked renormalization** – The baseline probabilities are element‑wise multiplied by the mask $\chi(c)$. If the sum of masked probabilities is zero, the example is flagged as a **deadlock**. Otherwise, the distribution is renormalized over the admissible set.

4. **Verification** – The code asserts that after masking, **every** forbidden token has exactly zero probability (null leakage = 0).

## Results (100 random examples)

| Metric | Value |
|--------|-------|
| Mean null mass (unmasked baseline) | 0.3011 |
| Proportion of examples with P(null) > 0 | 1.0000 |
| Deadlock rate after masking | 0.3000 |
| Masked null leakage (assertion check) | 0 (exactly) |

**Interpretation:**
- The baseline model wastes about **30% of its probability mass** on logically impossible answers.
- **Every single example** exhibits some degree of null leakage.
- After applying the logical mask, the forbidden completions receive **zero** probability, while the relative ordering of valid completions remains unchanged.
- The deadlock rate matches the fraction of contradictory premises (30%), demonstrating that the masking correctly identifies situations where no valid continuation exists.

These observations align with the theoretical claims of the paper: standard autoregressive models conflate logical necessity with surface plausibility, and support constraints provide a clean, structural remedy.
