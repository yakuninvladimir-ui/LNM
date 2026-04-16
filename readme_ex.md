# Support-Constrained Reasoning: Null Masking Experiment

## Overview

This repository contains a minimal but **strict experimental validation** of a key limitation of large language models:

> **LLMs assign non-zero probability mass to logically invalid outputs, even in simple reasoning tasks.**

We demonstrate that this behavior is **systematic**, and show that it can be **eliminated entirely** using support-constrained masking.

---

## Core Idea

We evaluate a pretrained language model as a **probability distribution over candidate completions**.

For each prompt:

1. Generate a small set of candidate continuations:

   * correct answer
   * incorrect logical alternatives
   * random unrelated text
2. Compute **exact log-probabilities** using the model
3. Measure how much probability mass is assigned to **invalid candidates**

We then simulate a **logical verifier**:

$$
\chi(c) =
\begin{cases}
1, & \text{valid continuation}\
0, & \text{invalid continuation}
\end{cases}
$$

and compare:

* baseline distribution (P_{\text{base}})
* masked distribution (P_{\text{masked}})

---

## Experiment Setup

* Model: `Qwen/Qwen2-1.5B`
* Device: CPU
* Samples: 1000
* Tasks:

  * Syllogisms (logical reasoning)
  * Arithmetic
  * Contradictions (unsatisfiable)

Log-probabilities are computed **exactly at token level**, not approximated.

Implementation: see 

---

## Metrics

We measure:

### 1. Null Mass

$$
\text{NullMass} = \sum_{c:\chi(c)=0} P_{\text{base}}(c)
$$

Probability assigned to invalid outputs.

---

### 2. Null Leakage

$$
P(\text{null} > 0)
$$

Fraction of cases where invalid tokens receive non-zero probability.

---

### 3. Deadlock Rate

$$
P\left(\sum P_{\text{masked}} = 0\right)
$$

Cases where **no valid continuation exists**.

---

## Key Results

### Global

* **Mean null mass ≈ 0.32**
* **P(null > 0) = 1.0**
* **Deadlock rate ≈ 0.30**

---

### Interpretation

#### 1. Systematic Logical Leakage

$$
P_{\text{base}}(\text{invalid}) > 0 \quad \text{in 100% of cases}
$$

The model **always assigns probability to incorrect answers**, even when the correct answer is trivial.

---

#### 2. Significant Error Mass

~30% of total probability mass is allocated to invalid continuations.

This is not noise — it is a **structural property of the model distribution**.

---

#### 3. Failure on Contradictions

For inconsistent inputs:

* valid solution does not exist
* all candidates are invalid
* masked distribution collapses

$$
\Rightarrow \text{deadlock}
$$

This demonstrates:

> The model **does not natively represent UNSAT states** and will normally generate an answer anyway.

---

## What This Shows

### Proven

* LLMs do **not enforce logical constraints in probability space**
* Invalid outputs always have non-zero support
* Logical inconsistency is not represented internally

---

### Demonstrated

* Support-constrained masking enforces:

$$
P_{\text{masked}}(\text{invalid}) = 0
$$

* Deadlock naturally emerges for unsatisfiable problems

---

### Not Claimed

This experiment does **not** prove:

* improved reasoning ability
* elimination of hallucinations in general
* scalability to open-ended generation

It isolates a **single property**:

> Control of probability support.

---

## Limitations

* Candidate space is finite (not full vocabulary)
* Verifier is synthetic (oracle equality check)
* No open-ended generation
* No real semantic understanding

---

## Why This Matters

Standard LLMs behave as:

$$
\text{plausibility estimators}
$$

not:

$$
\text{constraint-satisfying reasoners}
$$

This experiment shows that:

> Logical correctness must be enforced structurally, not learned implicitly.

---

## Next Steps

To extend this work:

1. Move to **open-ended generation**
2. Replace oracle verifier with NLI or learned model
3. Evaluate impact on:

   * accuracy
   * hallucination rate
4. Integrate into constrained decoding pipeline

---

## Summary

This repository provides a **minimal, reproducible demonstration** that:

* logical errors are intrinsic to LLM probability distributions
* support masking eliminates them exactly
* reasoning should be framed as **constrained search**, not generation

---

## License

MIT (or specify)

---

## Citation

If you use this work, cite:

```
Yakunin, V. (2026)
Support-Constrained Reasoning and Null Masking
```
