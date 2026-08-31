# PAD-Curriculum

> Official implementation of the EMNLP 2026 paper:  
> **"Expanding the Solvable Frontier: Policy-Adaptive Curriculum Learning for Reasoning RL"**  




## 💡 Overview

Existing curriculum methods for reasoning RL treat problem difficulty as a **static** attribute. **PAD-Curriculum** argues that difficulty is **policy-dependent**:
1. Problems beyond current student capability provide sparse outcome rewards.
2. Problems currently solved by greedy decoding can still provide rich stochastic learning signals during GRPO rollouts.

PAD-Curriculum dynamically tracks the student's evolving solvable frontier:
- **Correctness-Aware Filtering**: Only correctly solved greedy trajectories enter the fine-grained difficulty ranking; unsolved problems are deferred.
- **Discrepancy-Guided Ordering**: Solved problems are sorted by normalized **Scorer $\rightarrow$ Student KL discrepancy** ($D_i = \frac{KL_i}{H_{score,i} + \epsilon}$) in **ascending order** (easy/consolidated $\to$ harder frontier).
- **Dynamic Re-Ranking**: Every $K$ policy updates, the remaining unconsumed training queue is re-evaluated and re-ranked.
- **Purely Scheduling**: The frozen curriculum scorer is **never** used as a distillation target or GRPO reference model.

---

## 🛠️ Architecture & veRL Integration

PAD-Curriculum is built natively on top of the [veRL](https://github.com/volcengine/verl) RL training framework:

| Module | veRL Component | Responsibility in PAD-Curriculum |
| :--- | :--- | :--- |
| **`pad_trainer.py`** | `RayPPOTrainer` (Trainer Loop) | Replaces static DataLoader with dynamic `CurriculumQueue` and handles $K$-step re-ranking |
| **`scoring.py`** | `actor_rollout_wg` & `scorer_wg` | Computes token-level $D_{KL}(\pi_{score} \parallel \pi_\theta)$ and scorer entropy $H_{score}$ |
| **`verifier.py`** | `custom_reward_function` | Rule-based exact/numeric math verifier for outcome rewards and correctness filtering |
| **`dataset.py`** | `DataProto` / Data Preprocessing | Standard splits & token-level deduplication for GSM8K, MATH, and NuminaMath-CoT |

---

## 🚀 Quick Start

### 1. Requirements

```bash
pip install -r requirements.txt
```

### 2. Training with veRL

Launch GRPO training with PAD-Curriculum:
```bash
bash scripts/train.sh configs/pad_config.yaml Qwen/Qwen3-8B Qwen/Qwen3-32B gsm8k
```

### 3. Evaluation
```bash
python3 pad_curriculum/main.py eval --model-path /path/to/checkpoint --dataset math



```
