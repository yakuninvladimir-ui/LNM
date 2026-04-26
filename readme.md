# Support-Constrained Generation for Syllogistic and Arithmetic Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen2--1.5B-orange)](https://huggingface.co/Qwen/Qwen2-1.5B)

This repository contains a minimal yet rigorous demonstration of **support‑constrained generation** inspired by the paper  
*"Support-Constrained Variational Meta Chain-of-Thought: A Nullity-Semantics Framework"* (Yakunin, 2026).

We show that a standard causal language model (**Qwen2‑1.5B**) systematically assigns non‑zero probability to logically impossible completions, and that a simple logical mask eliminates this *null leakage* while preserving relative preferences among valid continuations.  
The experiment also validates the **deadlock mechanism** – when no admissible continuation exists, the system correctly signals a logical contradiction.

---

## What Does This Experiment Test?

We evaluate the base model on three types of synthetic tasks:

| Task type        | Example prompt                                     | Correct completion          | Incorrect completions               |
|------------------|----------------------------------------------------|-----------------------------|-------------------------------------|
| **syllogism**    | `All A are B.\nAll B are C.\nTherefore:`           | ` All A are C.`             | ` Some A are not C.`<br>` No A are C.` |
| **arithmetic**   | `12 + 7 =`                                         | ` 19`                       | ` 20`, ` 18`                        |
| **contradiction**| `X is true.\nNot X is true.\nTherefore:`           | *(none)*                    | ` X is true.`, ` Not X is true.`    |

For each example we compute the exact log‑probability of every candidate continuation (teacher‑forcing style) and then apply a hard logical mask $\chi(c) \in \{0,1\}$ provided by an **ideal oracle verifier** (which knows the ground‑truth answer).

Two key metrics are reported:

- **`null_mass`** – the total probability mass that the unmasked model assigns to **logically invalid** completions.
- **`deadlock`** – whether the set of admissible continuations is empty after masking (i.e. no valid token exists).

---

## Key Results (N = 1000)

| Task type       | Count | Mean `null_mass` | `null_positive` rate | `deadlock` rate |
|-----------------|-------|------------------|----------------------|-----------------|
| **contradiction** | 298   | ≈ 1.0            | 1.0                  | 1.0             |
| **arithmetic**    | 300   | 0.08             | 1.0                  | 0.0             |
| **syllogism**     | 402   | 0.004            | 1.0                  | 0.0             |

**Observations:**

1. **Null leakage is universal.**  
   In every non‑contradiction example the base model assigns strictly positive probability to at least one invalid completion (`null_positive = 1.0`).

2. **The magnitude of leakage varies.**  
   The model is far more confident on syllogistic patterns (mean null mass ~0.4%) than on exact arithmetic (mean null mass ~8%), reflecting the inherent difficulty of precise numerical reasoning for a pure language model.

3. **Deadlock occurs exactly when it should.**  
   All contradictory prompts result in an empty admissible set after masking (`deadlock = 1`). This is the precise condition for emitting a special `[LOGICAL_ERROR]` token and triggering backtracking in a full Meta‑CoT system.

4. **Masking guarantees zero null probability.**  
   By construction, after renormalizing over the masked support, every invalid token receives **exactly zero** probability. This empirically verifies **Proposition 1 (Strict exclusion of Null)** from the theoretical framework.

---

## How to Reproduce

### Requirements

- Python 3.8+
- PyTorch ≥ 2.0
- Transformers ≥ 4.30
- NumPy, tqdm

Install dependencies:

```bash
pip install torch transformers numpy tqdm
