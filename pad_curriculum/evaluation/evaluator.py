# -*- coding: utf-8 -*-
"""
Evaluation Protocol for Mathematical Reasoning Benchmarks.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..core.curriculum import ProblemItem
from ..core.verifier import RuleBasedVerifier


@dataclass
class EvalResult:
    dataset: str
    num_examples: int
    num_runs: int
    seeds: List[int]
    accuracies: List[float]
    mean_accuracy: float
    std_accuracy: float


GeneratorFn = Callable[[List[str], int], List[str]]


def evaluate_dataset(
    items: Sequence[ProblemItem],
    generator: GeneratorFn,
    dataset_name: str = "eval",
    verifier: Optional[RuleBasedVerifier] = None,
    seeds_large: Sequence[int] = (42, 43, 44),
    seeds_small_count: int = 10,
    small_threshold: int = 500,
) -> EvalResult:
    """
    Evaluates exact match accuracy across benchmark datasets.
    Repeats evaluation 3 times for large sets (>=500) and 10 times for small sets (<500).
    """
    if verifier is None:
        verifier = RuleBasedVerifier()

    n = len(items)
    if n >= small_threshold:
        seeds = list(seeds_large)
    else:
        seeds = list(range(42, 42 + seeds_small_count))

    prompts = [it.prompt for it in items]
    ground_truths = [it.ground_truth for it in items]

    run_accs: List[float] = []
    for s in seeds:
        responses = generator(prompts, s)
        assert len(responses) == n
        correct = sum(verifier(r, gt) for r, gt in zip(responses, ground_truths))
        acc = (correct / n) * 100.0 if n > 0 else 0.0
        run_accs.append(acc)

    mean_acc = statistics.mean(run_accs) if run_accs else 0.0
    std_acc = statistics.stdev(run_accs) if len(run_accs) > 1 else 0.0

    return EvalResult(
        dataset=dataset_name,
        num_examples=n,
        num_runs=len(seeds),
        seeds=seeds,
        accuracies=run_accs,
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
    )
