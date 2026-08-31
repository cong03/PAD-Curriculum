"""
Policy-Adaptive Curriculum Learning for Reasoning RL.
Implements discrepancy scoring and curriculum queue management.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ProblemItem:
    problem_id: str
    prompt: str
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    greedy_response: Optional[str] = None
    is_solved: Optional[bool] = None
    discrepancy_score: Optional[float] = None
    last_evaluated_step: int = -1


def compute_token_kl_and_entropy(
    scorer_log_probs: List[List[float]],
    student_log_probs: List[List[float]],
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """
    Computes Scorer -> Student KL divergence and scorer entropy along the response trajectory,
    and returns the normalized discrepancy score.
    """
    seq_len = len(scorer_log_probs)
    if seq_len == 0:
        return 0.0, 0.0, 0.0

    total_kl = 0.0
    total_entropy = 0.0

    for s_lp, t_lp in zip(scorer_log_probs, student_log_probs):
        kl_k = 0.0
        entropy_k = 0.0
        for log_p, log_q in zip(s_lp, t_lp):
            p = math.exp(log_p)
            if p > 0:
                kl_k += p * (log_p - log_q)
                entropy_k -= p * log_p
        
        total_kl += max(0.0, kl_k)
        total_entropy += max(0.0, entropy_k)

    avg_kl = total_kl / seq_len
    avg_entropy = total_entropy / seq_len
    difficulty_score = avg_kl / (avg_entropy + eps)

    return difficulty_score, avg_kl, avg_entropy


class CurriculumQueue:
    """
    Curriculum queue manager:
    1. Correctly solved problems are sorted by normalized discrepancy score in ascending order.
    2. Unsolved problems are deferred to the end of the queue without fine-grained ranking.
    3. Each problem is consumed at most once per epoch.
    4. Remaining unconsumed problems are re-ranked every K policy update steps.
    """

    def __init__(self, problems: List[ProblemItem], eps: float = 1e-8):
        self.all_problems: List[ProblemItem] = problems
        self.eps = eps
        self.active_queue: List[ProblemItem] = []
        self.consumed_in_epoch: set = set()
        self.reset_epoch()

    def reset_epoch(self):
        """Resets consumption tracking and initializes the active queue for a new epoch."""
        self.consumed_in_epoch.clear()
        self.active_queue = list(self.all_problems)

    def is_empty(self) -> bool:
        return len(self.active_queue) == 0

    def remaining_count(self) -> int:
        return len(self.active_queue)

    def get_remaining_problems(self) -> List[ProblemItem]:
        return list(self.active_queue)

    def update_and_reorder(self, evaluated_items: List[ProblemItem]):
        """
        Updates problem statuses and rebuilds the active queue:
        - Solved problems are placed first and sorted by discrepancy_score ascending.
        - Unsolved problems are placed at the end without fine-grained ordering.
        """
        solved_pool: List[ProblemItem] = []
        unsolved_pool: List[ProblemItem] = []

        for item in evaluated_items:
            if item.is_solved:
                if item.discrepancy_score is None:
                    item.discrepancy_score = 0.0
                solved_pool.append(item)
            else:
                item.discrepancy_score = None
                unsolved_pool.append(item)

        solved_pool.sort(key=lambda x: x.discrepancy_score)
        self.active_queue = solved_pool + unsolved_pool

    def pop_batch(self, batch_size: int) -> List[ProblemItem]:
        """
        Pops batch_size problems from the front of the queue.
        Popped problems are marked as consumed in the current epoch.
        """
        count = min(batch_size, len(self.active_queue))
        batch = self.active_queue[:count]
        self.active_queue = self.active_queue[count:]
        for item in batch:
            self.consumed_in_epoch.add(item.problem_id)
        return batch
