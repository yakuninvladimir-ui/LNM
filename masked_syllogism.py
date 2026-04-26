import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import csv
from tqdm import tqdm

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2-1.5B"
N = 1000
SEED = 42

# =========================
# INIT
# =========================
random.seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device("cpu")
model.to(device)

# =========================
# LOGPROB (STRICT)
# =========================
def logprob(prompt, continuation):
    enc_p = tokenizer(prompt, return_tensors="pt")
    enc_c = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)

    input_ids = torch.cat([enc_p["input_ids"], enc_c["input_ids"]], dim=1).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    cont_len = enc_c["input_ids"].shape[1]
    return token_log_probs[:, -cont_len:].sum().item()

# =========================
# TASK GENERATORS
# =========================

def syllogism():
    A, B, C = random.sample(["A","B","C","X","Y","Z"], 3)
    prompt = f"All {A} are {B}.\nAll {B} are {C}.\nTherefore:"
    correct = f" All {A} are {C}."
    wrong = [
        f" Some {A} are not {C}.",
        f" No {A} are {C}."
    ]
    return prompt, correct, wrong, "syllogism"

def contradiction():
    X = random.choice(["A","X","P"])
    prompt = f"{X} is true.\nNot {X} is true.\nTherefore:"
    correct = None
    wrong = [
        f" {X} is true.",
        f" Not {X} is true."
    ]
    return prompt, correct, wrong, "contradiction"

def arithmetic():
    a = random.randint(2, 20)
    b = random.randint(2, 20)
    correct_val = a + b
    prompt = f"{a} + {b} ="
    correct = f" {correct_val}"
    wrong = [
        f" {correct_val + random.randint(1,3)}",
        f" {correct_val - random.randint(1,3)}"
    ]
    return prompt, correct, wrong, "arithmetic"

def generate_task():
    r = random.random()
    if r < 0.4:
        return syllogism()
    elif r < 0.7:
        return arithmetic()
    else:
        return contradiction()

# =========================
# VERIFIER
# =========================
def chi(token, correct):
    if correct is None:
        return 0
    return 1 if token == correct else 0

# =========================
# EXPERIMENT
# =========================
results = []
RANDOM_POOL = [
    " The sky is green.",
    " Therefore, cats can fly.",
    " This statement is unrelated.",
    " No conclusion can be drawn.",
    " The system is undefined.",
    " Possibly true under assumptions.",
    " This contradicts previous statements."
]
for i in tqdm(range(N)):
    random_text_candidates = random.sample(RANDOM_POOL, k=3)
    prompt, correct, wrong, task_type = generate_task()
    candidates = ([correct] + wrong + random_text_candidates) if correct else (wrong + random_text_candidates)

    logits = []
    for c in candidates:
        logits.append(logprob(prompt, c))

    logits = np.array(logits)

    # softmax
    p_base = np.exp(logits - np.max(logits))
    p_base /= p_base.sum()

    mask = np.array([chi(c, correct) for c in candidates])

    null_mass = p_base[mask == 0].sum()
    null_positive = int((p_base[mask == 0] > 0).any())

    # masked
    p_masked_unnorm = p_base * mask
    deadlock = int(p_masked_unnorm.sum() == 0)

    results.append({
        "task": task_type,
        "null_mass": float(null_mass),
        "null_positive": null_positive,
        "deadlock": deadlock
    })

# =========================
# AGGREGATION
# =========================

def summarize(results):
    null_mass = np.mean([r["null_mass"] for r in results])
    null_positive = np.mean([r["null_positive"] for r in results])
    deadlock = np.mean([r["deadlock"] for r in results])

    print("\n=== GLOBAL ===")
    print(f"N: {len(results)}")
    print(f"Mean null mass: {null_mass:.4f}")
    print(f"P(null > 0): {null_positive:.4f}")
    print(f"Deadlock rate: {deadlock:.4f}")

    # по типам задач
    print("\n=== BY TASK TYPE ===")
    for t in set(r["task"] for r in results):
        subset = [r for r in results if r["task"] == t]
        print(f"\n[{t}]")
        print(f"  count: {len(subset)}")
        print(f"  null_mass: {np.mean([r['null_mass'] for r in subset]):.4f}")
        print(f"  null_positive: {np.mean([r['null_positive'] for r in subset]):.4f}")
        print(f"  deadlock: {np.mean([r['deadlock'] for r in subset]):.4f}")

summarize(results)

# =========================
# SAVE CSV
# =========================
with open("results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task", "null_mass", "null_positive", "deadlock"])
    writer.writeheader()
    writer.writerows(results)

print("\nSaved to results.csv")
