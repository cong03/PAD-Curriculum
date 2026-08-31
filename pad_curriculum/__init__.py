# -*- coding: utf-8 -*-
from pad_curriculum.core.curriculum import CurriculumQueue, ProblemItem, compute_token_kl_and_entropy
from pad_curriculum.core.scoring import compute_discrepancy_from_logprobs, DEFAULT_EPS
from pad_curriculum.core.verifier import RuleBasedVerifier, verify_batch
from pad_curriculum.trainer.pad_trainer import PADTrainer, PADConfig

__all__ = [
    "CurriculumQueue",
    "ProblemItem",
    "compute_token_kl_and_entropy",
    "compute_discrepancy_from_logprobs",
    "DEFAULT_EPS",
    "RuleBasedVerifier",
    "verify_batch",
    "PADTrainer",
    "PADConfig",
]
