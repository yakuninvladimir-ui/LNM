# Support-Constrained Generation for Syllogistic and Arithmetic Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Qwen2--1.5B-orange)](https://huggingface.co/Qwen/Qwen2-1.5B)

# Low-Cost Logical Control for LLMs

This repository demonstrates a minimal framework for analyzing and improving reasoning reliability in language models.

It contains two complementary experiments:

1. **Post-hoc logical filtering (GSM8K)**
2. **Support-constrained inference (distribution-level analysis)**

The goal is not to build a production system, but to show a structural property of LLMs:

> Language models assign probability mass to logically invalid reasoning.

---

# Contents

## 1. GSM8K Experiment

File: `gsm8k.py`

Implements:

- Multi-sample reasoning (self-consistency)
- Cheap logical consistency scoring
- Filtering of inconsistent trajectories
- Data collection for distillation

### Output

- `lowcost_logic_metrics.json`
- `logical_distillation_data.jsonl`

### Example results

```
Baseline: 36%
Soft (consensus + filtering): 40%
TopScore: 37%
Filtered_Out_Avg: ~39%
```

Interpretation:

- ~40% of generated reasoning is logically inconsistent
- Removing inconsistent trajectories improves accuracy
- Logic alone is weaker than statistical consensus

---

## 2. Support-Constrained Inference

File: `masked_syllogism.py`

Implements:

- Exact log-probability computation
- Masked distributions via χ(x)
- Measurement of invalid probability mass

### Metrics

- **Null mass** — probability assigned to invalid outputs
- **Deadlock** — no valid continuation exists

### Output

- `results.csv`

---

# Core Idea

We model generation as:

```
P(x) = P_valid(x) + P_invalid(x)
```

Where:

- `P_invalid` corresponds to logically inconsistent reasoning

Key observation:

> LLMs systematically assign non-zero probability to invalid reasoning.

---

# Two Regimes

## 1. Post-hoc filtering (cheap)

```
generate → filter → select
```

- No model changes
- Works with any API
- +3–5% accuracy improvement

## 2. Support-constrained inference (correct)

```
modify P(x) → mask invalid support → renormalize
```

- Requires access to logits
- Provides theoretical grounding

---

# Connection

Post-hoc filtering can be interpreted as:

> A Monte Carlo approximation of support-constrained inference

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Running Experiments

## GSM8K

```bash
python gsm8k.py
```

Make sure to set:

```python
MODEL_PATH = "path/to/your/gguf"
```

---

## Masked Syllogism

```bash
python masked_syllogism.py
```

---

# Notes

- GSM8K experiment uses `llama.cpp` backend
- Masked experiment uses HuggingFace transformers
- CPU-only execution is supported (slow but works)

---

# Limitations

- Logical checks are local (arithmetic only)
- No full logical entailment
- Sensitive to output formatting
- Not a replacement for reasoning models

---

# Interpretation

This repository shows:

1. A large fraction of LLM outputs are internally inconsistent
2. Simple filtering improves results
3. The root problem lies in probability allocation

---

# Takeaway

> The issue is not generation quality, but probability support.

Correct reasoning requires:

```
Constraining the support of P(x)
```

---

# Status

Experimental / research prototype.

Not intended as a production library.
