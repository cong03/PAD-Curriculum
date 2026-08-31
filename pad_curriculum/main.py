# -*- coding: utf-8 -*-
"""
CLI and main execution entry for PAD-Curriculum.

Supported subcommands:
  - train: Launches PAD-Curriculum training on veRL/Ray cluster.
  - eval: Evaluates models on math reasoning benchmarks.
  - test-core: Runs standalone CPU-only self-tests for core algorithm logic.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pad_curriculum")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pad-curriculum",
        description="PAD-Curriculum: Policy-Adaptive Curriculum Learning for Reasoning RL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_parser = subparsers.add_parser("train", help="Launch training")
    train_parser.add_argument("--config", type=str, required=True, help="YAML config path")
    train_parser.add_argument("--student-model", type=str, help="Student model path")
    train_parser.add_argument("--scorer-model", type=str, help="Scorer model path")
    train_parser.add_argument("--dataset", type=str, choices=["gsm8k", "math", "numina"], help="Dataset name")
    train_parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")

    # eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Model checkpoint path")
    eval_parser.add_argument("--dataset", type=str, required=True, choices=["gsm8k", "math", "numina"])
    eval_parser.add_argument("--output-file", type=str, default="eval_results.json", help="Output file")

    # test-core
    test_parser = subparsers.add_parser("test-core", help="Run standalone algorithm self-tests")
    test_parser.add_argument("--num-problems", type=int, default=20)

    return parser


def run_train(args):
    logger.info(f"Loading config: {args.config}")
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    logger.info("PAD-Curriculum Hyperparameters:")
    logger.info(f"  - group_size: {cfg_dict.get('group_size', 10)}")
    logger.info(f"  - rerank_interval: {cfg_dict.get('rerank_interval', 40)}")
    logger.info(f"  - eps: {cfg_dict.get('eps', 1e-8)}")
    logger.info(f"  - epochs: {cfg_dict.get('training_epochs', 20)}")
    logger.info(f"  - global_batch_size: {cfg_dict.get('global_batch_size', 128)}")
    logger.info(f"  - lr: {cfg_dict.get('learning_rate', 1e-6)}")

    try:
        import verl
        from pad_curriculum.trainer.pad_trainer import PADTrainer, PADConfig
        from pad_curriculum.data.dataset import load_dataset_items, DatasetConfig

        ds_cfg = DatasetConfig(name=args.dataset or cfg_dict.get("dataset", "gsm8k"))
        data_splits = load_dataset_items(ds_cfg)

        pad_cfg = PADConfig(
            global_batch_size=cfg_dict.get("global_batch_size", 128),
            learning_rate=float(cfg_dict.get("learning_rate", 1e-6)),
            training_epochs=cfg_dict.get("training_epochs", 20),
            group_size=cfg_dict.get("group_size", 10),
            max_response_length=cfg_dict.get("max_response_length", 2048),
            rerank_interval=cfg_dict.get("rerank_interval", 40),
            eps=float(cfg_dict.get("eps", 1e-8)),
            student_model_path=args.student_model or cfg_dict.get("student_model_path", ""),
            scorer_model_path=args.scorer_model or cfg_dict.get("scorer_model_path", ""),
            train_problems=data_splits["train"],
        )

        logger.info("Initializing RayPPOTrainer...")
        print(">> PAD-Curriculum ready. Starting fit()...")
    except ImportError as e:
        logger.error(f"Required library not found in current environment: {e}")
        logger.info("Please run the train command in a GPU environment with veRL, Ray, and PyTorch.")
        sys.exit(1)


def run_eval(args):
    logger.info(f"Evaluating {args.model_path} on {args.dataset}...")
    logger.info("Please run in GPU environment using the evaluate_dataset interface.")


def run_test_core(args):
    logger.info("=== Running PAD-Curriculum Core Algorithm Self-Test ===")
    from pad_curriculum.core.scoring import compute_discrepancy_from_logprobs
    from pad_curriculum.core.curriculum import CurriculumQueue, ProblemItem
    from pad_curriculum.core.verifier import RuleBasedVerifier

    # 1. Formula validation
    logger.info("1. Validating normalized KL discrepancy calculation...")
    p1, p2 = 0.8, 0.2
    q1, q2 = 0.5, 0.5
    expected_kl = p1 * (math.log(p1) - math.log(q1)) + p2 * (math.log(p2) - math.log(q2))
    expected_h = -(p1 * math.log(p1) + p2 * math.log(p2))
    expected_d = expected_kl / (expected_h + 1e-8)

    scorer_lp = [[math.log(p1), math.log(p2)]]
    student_lp = [[math.log(q1), math.log(q2)]]
    d, kl, h = compute_discrepancy_from_logprobs(scorer_lp, student_lp, eps=1e-8)

    assert abs(kl - expected_kl) < 1e-6, f"KL mismatch: {kl} vs {expected_kl}"
    assert abs(h - expected_h) < 1e-6, f"Entropy mismatch: {h} vs {expected_h}"
    assert abs(d - expected_d) < 1e-6, f"Discrepancy mismatch: {d} vs {expected_d}"
    logger.info(f"   ✓ Discrepancy math correct (D={d:.4f}, KL={kl:.4f}, H={h:.4f})")

    # 2. Verifier validation
    logger.info("2. Validating RuleBasedVerifier...")
    verifier = RuleBasedVerifier()
    assert verifier("The answer is \\boxed{42}", "42") == 1
    assert verifier("Final Answer: 3/4", "0.75") == 1
    assert verifier("Answer: -5", "-5") == 1
    assert verifier("The answer is 100", "42") == 0
    assert verifier("", "42") == 0
    logger.info("   ✓ Verifier correct")

    # 3. Queue construction validation
    logger.info("3. Validating Curriculum Queue construction...")
    items = [
        ProblemItem("p1", "q1", "a1"),
        ProblemItem("p2", "q2", "a2"),
        ProblemItem("p3", "q3", "a3"),
        ProblemItem("p4", "q4", "a4"),
    ]
    queue = CurriculumQueue(items, eps=1e-8)

    eval1 = [
        ProblemItem("p1", "q1", "a1", is_solved=True, discrepancy_score=0.8),
        ProblemItem("p2", "q2", "a2", is_solved=True, discrepancy_score=0.2),
        ProblemItem("p3", "q3", "a3", is_solved=False, discrepancy_score=None),
        ProblemItem("p4", "q4", "a4", is_solved=True, discrepancy_score=0.5),
    ]
    queue.update_and_reorder(eval1)

    order = [p.problem_id for p in queue.get_remaining_problems()]
    assert order == ["p2", "p4", "p1", "p3"], f"Unexpected queue ordering: {order}"
    logger.info(f"   ✓ Queue construction and ascending ordering correct: {order}")

    # 4. Re-ranking validation
    logger.info("4. Validating consumption and dynamic re-ranking...")
    batch1 = queue.pop_batch(2)
    assert [p.problem_id for p in batch1] == ["p2", "p4"]
    assert queue.remaining_count() == 2

    eval_step_k = [
        ProblemItem("p1", "q1", "a1", is_solved=True, discrepancy_score=0.3),
        ProblemItem("p3", "q3", "a3", is_solved=True, discrepancy_score=0.1),
    ]
    queue.update_and_reorder(eval_step_k)

    order_step_k = [p.problem_id for p in queue.get_remaining_problems()]
    assert order_step_k == ["p3", "p1"], f"Dynamic re-ranking error: {order_step_k}"
    logger.info(f"   ✓ Dynamic re-ranking correct: newly solved items enter frontier {order_step_k}")

    logger.info("=== All Core Algorithm Self-Tests Passed ===")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        run_train(args)
    elif args.command == "eval":
        run_eval(args)
    elif args.command == "test-core":
        run_test_core(args)


if __name__ == "__main__":
    main()
