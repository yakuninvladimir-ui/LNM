# =====================================================
# GSM8K: Low-Cost Filtering + Cheap Logic Graph + Logging
# =====================================================

import re
import json
from collections import defaultdict
from decimal import Decimal
from fractions import Fraction
from typing import Optional, List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from llama_cpp import Llama

# ===================== CONFIG =====================

class Config:
    TEMPERATURE = 0.75
    TRIES = 5
    LIMIT = 300
    MODEL_PATH = "qwen/qwen2.5-7b-instruct-q4_k_m.gguf"
    SEED = 42
    MAX_TOKENS = 256
    TOP_P = 0.92
    REPEAT_PENALTY = 1.12
    N_CTX = 1024
    N_BATCH = 512
    N_GPU_LAYERS = -1

CFG = Config()

DEADLOCK_TOKEN = "[DEADLOCK]"

# ===================== REGEX =====================

FINAL_ANSWER_RE = re.compile(r"####\s*([^\n\r]*)")
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

EQ_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)

# ===================== NORMALIZATION =====================

def normalize_text(text: str) -> str:
    if not text:
        return text

    t = text.lower()
    t = re.sub(r"[\$€£¥₽]", "", t)
    t = re.sub(r"\b(dollars?|usd|eur|руб(лей|ля)?)\b", "", t)
    t = re.sub(r"(\d)\s+(\d)", r"\1\2", t)
    t = t.replace(",", "")
    t = t.replace("×", "*").replace("÷", "/")

    return t

# ===================== NUMERIC =====================

def to_fraction(x: str) -> Optional[Fraction]:
    try:
        return Fraction(Decimal(x))
    except:
        return None

def eval_expr(a, op, b):
    if a is None or b is None:
        return None
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if op == "/":
        if b == 0:
            return None
        return a / b
    return None

# ===================== ANSWER =====================

def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None

    m = FINAL_ANSWER_RE.search(text)
    if m:
        nums = NUMBER_RE.findall(m.group(1))
        if nums:
            return nums[-1]

    nums = NUMBER_RE.findall(text)
    if nums:
        return nums[-1]

    return None

def parse_numeric(token: Optional[str]) -> Optional[Fraction]:
    if token is None:
        return None
    try:
        return Fraction(Decimal(token.replace(",", "")))
    except:
        return None

def numeric_equal(a: Optional[str], b: Optional[str]) -> bool:
    pa = parse_numeric(a)
    pb = parse_numeric(b)
    if pa is None or pb is None:
        return False
    return pa == pb

# ===================== GRAPH =====================

def extract_equations(text: str):
    text = normalize_text(text)
    eqs = []

    for a, op, b, c in EQ_RE.findall(text):
        fa = to_fraction(a)
        fb = to_fraction(b)
        fc = to_fraction(c)
        tr = eval_expr(fa, op, fb)

        eqs.append((fa, op, fb, fc, tr))

    return eqs

def has_contradiction(eqs):
    seen = {}

    for a, op, b, claimed, true_r in eqs:
        if true_r is None or claimed is None:
            return True

        if claimed != true_r:
            return True

        key = (a, op, b)

        if key in seen and seen[key] != claimed:
            return True

        seen[key] = claimed

    return False

def stability_score(eqs):
    counts = defaultdict(int)

    for _, _, _, claimed, _ in eqs:
        if claimed is not None:
            counts[claimed] += 1

    return max(counts.values()) if counts else 0

def cheap_logic_score(text: str) -> float:
    if not text:
        return -1e9

    eqs = extract_equations(text)

    if eqs:
        if has_contradiction(eqs):
            return -100.0

    score = 0.0

    if eqs:
        score += 2.0
        score += float(stability_score(eqs))

    if extract_answer(text) is not None:
        score += 1.0

    return score

# ===================== SELECTION =====================

def select_baseline(pool: List[str]) -> str:
    return pool[0] if pool else DEADLOCK_TOKEN

def select_soft(pool: List[str]) -> str:
    valid = [c for c in pool if cheap_logic_score(c) >= 0]

    if not valid:
        return DEADLOCK_TOKEN

    by_answer: Dict[str, List[str]] = defaultdict(list)

    for c in valid:
        ans = extract_answer(c)
        if ans is not None:
            by_answer[ans].append(c)

    if by_answer:
        best = max(by_answer.items(), key=lambda x: len(x[1]))
        return best[1][0]

    return max(valid, key=cheap_logic_score)

def select_top_score(pool: List[str]) -> str:
    if not pool:
        return DEADLOCK_TOKEN
    return max(pool, key=cheap_logic_score)

# ===================== MODEL =====================

class LLM:
    def __init__(self, model_path: str):
        self.model = Llama(
            model_path=model_path,
            n_ctx=CFG.N_CTX,
            n_gpu_layers=CFG.N_GPU_LAYERS,
            n_batch=CFG.N_BATCH,
            verbose=False,
        )

    def generate(self, prompt: str, seed: int) -> str:
        out = self.model(
            prompt,
            max_tokens=CFG.MAX_TOKENS,
            temperature=CFG.TEMPERATURE,
            top_p=CFG.TOP_P,
            repeat_penalty=CFG.REPEAT_PENALTY,
            seed=seed,
        )
        return out["choices"][0]["text"].strip()

# ===================== DATA =====================

def load_gsm8k(limit=CFG.LIMIT):
    ds = load_dataset("gsm8k", "main", split="test")

    data = []
    for item in ds:
        q = item["question"]
        a = item["answer"]

        m = FINAL_ANSWER_RE.search(a)
        gt = None
        if m:
            nums = NUMBER_RE.findall(m.group(1))
            if nums:
                gt = nums[-1]

        data.append((q, gt))

    return data[:limit]

# ===================== EXPERIMENT =====================

def run_experiment(llm, data):

    stats = {
        "n": 0,
        "baseline_correct": 0,
        "soft_correct": 0,
        "top_correct": 0,
        "filtered_out": 0,
    }

    distillation_data = []

    for i, (q, gt) in enumerate(tqdm(data)):

        prompt = (
            q
            + "\nSolve step by step carefully.\n"
            + "Give the final answer after ####.\n"
        )

        pool = []

        for attempt in range(CFG.TRIES):
            seed = CFG.SEED + i * 1000 + attempt
            out = llm.generate(prompt, seed)
            pool.append(out)

        # ===== СБОР ДАННЫХ =====
        for attempt_text in pool:
            score = cheap_logic_score(attempt_text)
            ans = extract_answer(attempt_text)

            distillation_data.append({
                "question": q,
                "trajectory": attempt_text,
                "final_answer": ans,
                "is_correct": numeric_equal(ans, gt),
                "logic_score": score,
                "has_contradiction": score < -50
            })

        valid_count = sum(1 for c in pool if cheap_logic_score(c) >= 0)
        stats["filtered_out"] += (len(pool) - valid_count)

        out_base = select_baseline(pool)
        out_soft = select_soft(pool)
        out_top = select_top_score(pool)

        stats["baseline_correct"] += int(numeric_equal(extract_answer(out_base), gt))
        stats["soft_correct"] += int(numeric_equal(extract_answer(out_soft), gt))
        stats["top_correct"] += int(numeric_equal(extract_answer(out_top), gt))

        stats["n"] += 1

    n = stats["n"]

    return {
        "metrics": {
            "Acc_Baseline": stats["baseline_correct"] / n,
            "Acc_Soft": stats["soft_correct"] / n,
            "Acc_TopScore": stats["top_correct"] / n,
            "Filtered_Out_Avg": stats["filtered_out"] / n,
        },
        "distillation_data": distillation_data
    }

# ===================== MAIN =====================

def main():
    np.random.seed(CFG.SEED)

    data = load_gsm8k()
    llm = LLM(CFG.MODEL_PATH)

    results = run_experiment(llm, data)

    print("\nRESULTS")
    for k, v in results["metrics"].items():
        print(f"{k}: {v:.3%}")

    with open("lowcost_logic_metrics.json", "w") as f:
        json.dump(results["metrics"], f, indent=2)

    with open("logical_distillation_data.jsonl", "w", encoding="utf-8") as f:
        for ex in results["distillation_data"]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\nSaved metrics and distillation data")

if __name__ == "__main__":
    main()