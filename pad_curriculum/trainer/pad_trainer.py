# -*- coding: utf-8 -*-
"""
PADTrainer: Curriculum-guided RL Trainer on top of veRL RayPPOTrainer.

Implements dynamic curriculum queue construction, greedy trajectory evaluation,
and periodic K-step re-ranking while delegating GRPO policy optimization to veRL.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.curriculum import CurriculumQueue, ProblemItem
from ..core.scoring import DEFAULT_EPS
from ..core.verifier import RuleBasedVerifier

logger = logging.getLogger(__name__)


@dataclass
class PADConfig:
    # GRPO Base Hyperparameters
    global_batch_size: int = 128
    learning_rate: float = 1e-6
    training_epochs: int = 20
    group_size: int = 10
    max_response_length: int = 2048

    # Curriculum Specific
    rerank_interval: int = 40
    eps: float = DEFAULT_EPS

    # Model paths
    student_model_path: str = ""
    scorer_model_path: str = ""

    # Data
    train_problems: List[ProblemItem] = field(default_factory=list)


class PADTrainer:
    """
    Curriculum-guided GRPO Trainer.
    Extends standard RL workflows with dynamic solvable frontier tracking.
    """

    def __init__(
        self,
        pad_config: PADConfig,
        verl_trainer,
        scorer_wg=None,
        verifier: Optional[RuleBasedVerifier] = None,
    ):
        self.cfg = pad_config
        self.verl_trainer = verl_trainer
        self.scorer_wg = scorer_wg
        self.verifier = verifier or RuleBasedVerifier()

        self.queue = CurriculumQueue(
            problems=list(pad_config.train_problems),
            eps=pad_config.eps,
        )

        self.global_step = 0
        self.current_epoch = 0

    def _greedy_decode_student(self, prompts: List[str]) -> List[str]:
        batch = self._build_generation_batch(prompts)
        gen_output = self.verl_trainer.actor_rollout_wg.generate_sequences(
            batch, do_sample=False, max_new_tokens=self.cfg.max_response_length
        )
        responses = self._decode_responses(gen_output)
        return responses

    def _compute_discrepancy_scores(
        self,
        prompts: List[str],
        greedy_responses: List[str],
    ) -> List[float]:
        if self.scorer_wg is None:
            logger.warning("scorer_wg not configured, discrepancy score defaulted to 0.0")
            return [0.0] * len(prompts)

        scores = []
        for prompt, response in zip(prompts, greedy_responses):
            student_logits = self._get_logits(
                self.verl_trainer.actor_rollout_wg, prompt, response
            )
            scorer_logits = self._get_logits(
                self.scorer_wg, prompt, response
            )

            from ..core.scoring import kl_entropy_from_logits_torch
            d_i, _, _ = kl_entropy_from_logits_torch(
                scorer_logits, student_logits, eps=self.cfg.eps
            )
            scores.append(d_i)

        return scores

    def _get_logits(self, worker_group, prompt: str, response: str):
        """
        Extracts logits for the response tokens under the given model/worker group.
        Handles both single PyTorch models and veRL Ray Worker Groups.
        """
        full_text = prompt + response
        inputs = self.verl_trainer.tokenizer(
            full_text, return_tensors="pt"
        )
        prompt_len = len(self.verl_trainer.tokenizer(prompt)["input_ids"])

        import torch
        with torch.no_grad():
            if hasattr(worker_group, "model"):
                inputs = {k: v.to(worker_group.device) for k, v in inputs.items()}
                outputs = worker_group.model(**inputs)
                logits = outputs.logits
            elif hasattr(worker_group, "compute_logits"):
                # Ray worker group interface
                from verl import DataProto
                dp = DataProto.from_dict({
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                })
                logits = worker_group.compute_logits(dp)
            else:
                inputs = {k: v.to(getattr(worker_group, "device", "cuda")) for k, v in inputs.items()}
                outputs = worker_group(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs

        response_logits = logits[0, prompt_len - 1 : -1, :]
        return response_logits

    def _evaluate_and_rank(self, problems: List[ProblemItem]) -> List[ProblemItem]:
        prompts = [p.prompt for p in problems]
        ground_truths = [p.ground_truth for p in problems]

        logger.info(f"[PAD] Greedy decoding for {len(prompts)} problems...")
        greedy_responses = self._greedy_decode_student(prompts)

        logger.info(f"[PAD] Verifying correctness...")
        correctness = [
            self.verifier(resp, gt) for resp, gt in zip(greedy_responses, ground_truths)
        ]

        solved_indices = [i for i, c in enumerate(correctness) if c == 1]
        if solved_indices:
            logger.info(f"[PAD] Computing discrepancy for {len(solved_indices)} solved problems...")
            solved_prompts = [prompts[i] for i in solved_indices]
            solved_responses = [greedy_responses[i] for i in solved_indices]
            discrepancy_scores = self._compute_discrepancy_scores(
                solved_prompts, solved_responses
            )
        else:
            discrepancy_scores = []

        score_idx = 0
        for i, item in enumerate(problems):
            item.greedy_response = greedy_responses[i]
            item.is_solved = bool(correctness[i])
            item.last_evaluated_step = self.global_step
            if item.is_solved:
                item.discrepancy_score = discrepancy_scores[score_idx]
                score_idx += 1
            else:
                item.discrepancy_score = None

        return problems

    def _build_curriculum(self):
        logger.info(f"[PAD] Epoch {self.current_epoch}: Building curriculum queue...")
        all_problems = self.queue.get_remaining_problems()
        evaluated = self._evaluate_and_rank(all_problems)
        self.queue.update_and_reorder(evaluated)
        logger.info(
            f"[PAD] Queue built: "
            f"solved={sum(1 for p in evaluated if p.is_solved)}, "
            f"unsolved={sum(1 for p in evaluated if not p.is_solved)}"
        )

    def _rerank_remaining(self):
        remaining = self.queue.get_remaining_problems()
        if not remaining:
            return

        logger.info(
            f"[PAD] Step {self.global_step}: Re-ranking {len(remaining)} remaining problems..."
        )
        evaluated = self._evaluate_and_rank(remaining)
        self.queue.update_and_reorder(evaluated)

    def _train_step(self, batch_problems: List[ProblemItem]) -> Dict[str, Any]:
        batch = self._build_training_batch(batch_problems)
        metrics = self.verl_trainer._train_step(batch)
        self.global_step += 1
        return metrics

    def _build_generation_batch(self, prompts: List[str]):
        from verl import DataProto
        import torch

        input_ids = []
        attention_mask = []
        for p in prompts:
            encoded = self.verl_trainer.tokenizer(p, return_tensors="pt")
            input_ids.append(encoded["input_ids"][0])
            attention_mask.append(encoded["attention_mask"][0])

        max_len = max(len(ids) for ids in input_ids)
        padded_ids = []
        padded_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            pad_len = max_len - len(ids)
            padded_ids.append(
                torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            )
            padded_mask.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )

        batch = DataProto.from_dict({
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_mask),
        })
        return batch

    def _build_training_batch(self, problems: List[ProblemItem]):
        prompts = [p.prompt for p in problems]
        batch = self._build_generation_batch(prompts)
        batch.non_tensor_batch["ground_truth"] = [p.ground_truth for p in problems]
        return batch

    def _decode_responses(self, gen_output) -> List[str]:
        responses = []
        for i in range(len(gen_output.batch["responses"])):
            response_ids = gen_output.batch["responses"][i]
            response_ids = response_ids[response_ids != self.verl_trainer.tokenizer.pad_token_id]
            text = self.verl_trainer.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(text)
        return responses

    def fit(self):
        logger.info("[PAD] Starting PAD-Curriculum training")
        logger.info(f"[PAD] Hyperparameters: {self.cfg}")

        for epoch in range(self.cfg.training_epochs):
            self.current_epoch = epoch
            logger.info(f"[PAD] ===== Epoch {epoch + 1}/{self.cfg.training_epochs} =====")

            self.queue.reset_epoch()
            self._build_curriculum()

            while not self.queue.is_empty():
                batch_problems = self.queue.pop_batch(self.cfg.global_batch_size)
                metrics = self._train_step(batch_problems)

                if self.global_step % 10 == 0:
                    logger.info(
                        f"[PAD] Step {self.global_step}: "
                        f"loss={metrics.get('actor/loss', 0):.4f}, "
                        f"remaining={self.queue.remaining_count()}"
                    )

                if self.global_step % self.cfg.rerank_interval == 0:
                    self._rerank_remaining()

            logger.info(f"[PAD] Epoch {epoch + 1} completed")

        logger.info("[PAD] Training finished")

    def evaluate(self, test_problems: List[ProblemItem]) -> Dict[str, float]:
        from ..evaluation.evaluator import evaluate_dataset

        def generator(prompts: List[str], seed: int) -> List[str]:
            return self._greedy_decode_student(prompts)

        result = evaluate_dataset(
            items=test_problems,
            generator=generator,
            dataset_name="test",
            verifier=self.verifier,
        )

        return {
            "mean_accuracy": result.mean_accuracy,
            "std_accuracy": result.std_accuracy,
            "num_runs": result.num_runs,
        }
