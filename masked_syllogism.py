import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np

# --- МОДЕЛЬ (CPU-совместимая) ---
MODEL_NAME = "Qwen/Qwen2-1.5B"  # можно заменить на любую causal LM

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cpu")
model.to(device)

# --- ФУНКЦИЯ: точный log P(continuation | prompt) ---
def logprob_conditional(prompt: str, continuation: str) -> float:
    # токенизируем по отдельности, затем склеиваем
    enc_p = tokenizer(prompt, return_tensors="pt")
    enc_c = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)

    input_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [1, T, V]

    # стандартный сдвиг
    shift_logits = logits[:, :-1, :]            # предсказываем токен t+1
    shift_labels = input_ids[:, 1:]             # истинные токены

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, T-1]

    # суммируем ТОЛЬКО по токенам continuation
    cont_len = enc_c["input_ids"].shape[1]
    if cont_len == 0:
        return 0.0

    # последние cont_len позиций соответствуют continuation
    cont_token_log_probs = token_log_probs[:, -cont_len:]
    return cont_token_log_probs.sum().item()

# --- ГЕНЕРАЦИЯ ЗАДАЧ ---
def generate_syllogism():
    A = random.choice(["A","X","P","M"])
    B = random.choice(["B","Y","Q","N"])
    C = random.choice(["C","Z","R","K"])

    prompt = f"All {A} are {B}.\nAll {B} are {C}.\nTherefore:"
    correct = f" All {A} are {C}."
    wrong = [
        f" Some {A} are not {C}.",
        f" No {A} are {C}.",
    ]
    return prompt, correct, wrong

def generate_contradiction():
    X = random.choice(["A","X","P"])
    prompt = f"{X} is true.\nNot {X} is true.\nTherefore:"
    # в этой игрушечной постановке считаем, что корректного вывода нет
    correct = None
    wrong = [
        f" {X} is true.",
        f" Not {X} is true.",
    ]
    return prompt, correct, wrong

# --- ВЕРИФИКАТОР χ ---
def chi(token: str, correct: str) -> int:
    if correct is None:
        return 0  # всё запрещено → deadlock
    return 1 if token == correct else 0

# --- ОДИН ПРОГОН ---
def evaluate_one(prompt, correct, candidates):
    logits = []
    for c in candidates:
        lp = logprob_conditional(prompt, c)
        logits.append(lp)

    logits = np.array(logits, dtype=np.float64)
    # стабильный softmax
    p_base = np.exp(logits - np.max(logits))
    Z = p_base.sum()
    if Z == 0:
        p_base = np.ones_like(p_base) / len(p_base)
    else:
        p_base = p_base / Z

    mask = np.array([chi(c, correct) for c in candidates], dtype=np.float64)

    # метрики
    null_mass = p_base[mask == 0].sum()                 # масса на запрещённых
    has_null_positive = float((p_base[mask == 0] > 0).any())

    # masked
    p_masked_unnorm = p_base * mask
    if p_masked_unnorm.sum() == 0:
        deadlock = 1
        p_masked = None
    else:
        deadlock = 0
        p_masked = p_masked_unnorm / p_masked_unnorm.sum()

    return {
        "p_base": p_base,
        "p_masked": p_masked,
        "null_mass": float(null_mass),
        "null_positive": has_null_positive,
        "deadlock": deadlock,
        "candidates": candidates,
        "correct": correct,
        "prompt": prompt
    }

# --- БАТЧ ---
def run_batch(N=100, mix=0.7, seed=0):
    random.seed(seed)
    results = []

    for _ in range(N):
        if random.random() < mix:
            prompt, correct, wrong = generate_syllogism()
            candidates = [correct] + wrong
        else:
            prompt, correct, wrong = generate_contradiction()
            candidates = wrong  # все запрещены

        res = evaluate_one(prompt, correct, candidates)
        results.append(res)

    # агрегаты
    null_mass_mean = np.mean([r["null_mass"] for r in results])
    null_positive_rate = np.mean([r["null_positive"] for r in results])
    deadlock_rate = np.mean([r["deadlock"] for r in results])

    print("=== AGGREGATES ===")
    print(f"N: {N}")
    print(f"Mean null mass (baseline): {null_mass_mean:.4f}")
    print(f"Rate P(null)>0 (baseline): {null_positive_rate:.4f}")
    print(f"Deadlock rate (masked): {deadlock_rate:.4f}")
    print("Expected (theory): masked null mass = 0.0 exactly")

    # sanity check для masked
    for r in results:
        if r["p_masked"] is not None:
            mask = np.array([chi(c, r["correct"]) for c in r["candidates"]])
            # все запрещённые должны иметь 0
            assert np.all(r["p_masked"][mask == 0] == 0.0)

    print("Sanity check passed: masked null leakage = 0")

    return results

if __name__ == "__main__":
    run_batch(N=100)