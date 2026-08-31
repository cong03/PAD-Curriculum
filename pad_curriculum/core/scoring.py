# -*- coding: utf-8 -*-
"""
Discrepancy-Guided Difficulty Estimation.

Calculates Scorer -> Student KL divergence and scorer entropy along student-generated trajectories:
    KL_i = (1 / L_i) * sum_k D_KL(pi_score(·|x_{i,k}) || pi_theta(·|x_{i,k}))
    H_score,i = (1 / L_i) * sum_k H(pi_score(·|x_{i,k}))
    D_i = KL_i / (H_score,i + eps)

Notes:
  - x_{i,k} is the prefix (prompt + response tokens up to k-1) from greedy decoding.
  - KL direction is Scorer -> Student.
  - Computed only for problems verified as correct; unsolved problems are deferred.
  - The scorer is frozen and used only for curriculum ordering.
"""

from __future__ import annotations

import math
from typing import List, Protocol, Sequence, Tuple

DEFAULT_EPS = 1e-8


def compute_discrepancy_from_logprobs(
    scorer_log_probs: Sequence[Sequence[float]],
    student_log_probs: Sequence[Sequence[float]],
    eps: float = DEFAULT_EPS,
) -> Tuple[float, float, float]:
    """
    Computes normalized discrepancy score from full-vocabulary log-probabilities.

    Args:
      scorer_log_probs: List of log-probabilities for each token position from scorer model P.
      student_log_probs: List of log-probabilities for each token position from student model Q.
      eps: Numerical stability constant.

    Returns:
      (difficulty_score D_i, avg_kl, avg_entropy)
    """
    seq_len = len(scorer_log_probs)
    if seq_len == 0:
        return 0.0, 0.0, 0.0
    assert len(student_log_probs) == seq_len

    total_kl = 0.0
    total_entropy = 0.0
    for s_logprobs, t_logprobs in zip(scorer_log_probs, student_log_probs):
        assert len(s_logprobs) == len(t_logprobs)
        kl_k = 0.0
        h_k = 0.0
        for log_p, log_q in zip(s_logprobs, t_logprobs):
            p = math.exp(log_p)
            if p > 0.0:
                kl_k += p * (log_p - log_q)
                h_k -= p * log_p
        total_kl += max(0.0, kl_k)
        total_entropy += max(0.0, h_k)

    avg_kl = total_kl / seq_len
    avg_entropy = total_entropy / seq_len
    difficulty = avg_kl / (avg_entropy + eps)
    return difficulty, avg_kl, avg_entropy


def kl_entropy_from_logits_torch(
    scorer_logits,
    student_logits,
    eps: float = DEFAULT_EPS,
    chunk_size: int = 128,
) -> Tuple[float, float, float]:
    """
    Computes normalized discrepancy score directly from logits using chunking to save GPU memory.

    Args:
      scorer_logits / student_logits: Tensors of shape (L, V).
      eps: Numerical stability constant.
      chunk_size: Number of token positions processed per chunk.

    Returns:
      (difficulty_score D_i, avg_kl, avg_entropy)
    """
    import torch

    assert scorer_logits.shape == student_logits.shape
    seq_len = scorer_logits.shape[0]
    if seq_len == 0:
        return 0.0, 0.0, 0.0

    total_kl = torch.zeros((), dtype=torch.float64, device=scorer_logits.device)
    total_h = torch.zeros((), dtype=torch.float64, device=scorer_logits.device)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        s_lp = torch.log_softmax(scorer_logits[start:end].float(), dim=-1)
        t_lp = torch.log_softmax(student_logits[start:end].float(), dim=-1)

        p = s_lp.exp()
        kl = (p * (s_lp - t_lp)).sum(dim=-1)
        h = -(p * s_lp).sum(dim=-1)

        total_kl += kl.sum()
        total_h += h.sum()

    avg_kl = total_kl / seq_len
    avg_h = total_h / seq_len
    avg_kl = torch.clamp(avg_kl, min=0.0)
    avg_h = torch.clamp(avg_h, min=0.0)
    difficulty = avg_kl / (avg_h + eps)
    return float(difficulty.cpu()), float(avg_kl.cpu()), float(avg_h.cpu())


class GreedyDecoder(Protocol):
    """Protocol for greedy trajectory generation from the student model."""

    def greedy_decode(self, prompts: List[str]) -> List[str]:
        ...


class DiscrepancyScorer(Protocol):
    """Protocol for computing Scorer -> Student KL discrepancy."""

    def score(
        self,
        prompts: List[str],
        greedy_responses: List[str],
        eps: float = DEFAULT_EPS,
    ) -> List[Tuple[float, float, float]]:
        ...
