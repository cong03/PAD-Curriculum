# -*- coding: utf-8 -*-
"""
Dataset Loading and Preprocessing.

Supports GSM8K, MATH, and NuminaMath-CoT benchmarks with token-level deduplication
and overlap filtering.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..core.curriculum import ProblemItem
from ..core.verifier import normalize_text


def normalize_problem_text(text: str) -> str:
    """Normalizes problem text for deduplication."""
    if not text:
        return ""
    s = text.strip()
    s = re.sub(r"\\,|\\;|\\quad|\\qquad", " ", s)
    s = re.sub(r"\\boxed\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def problem_fingerprint(text: str) -> str:
    return hashlib.md5(normalize_problem_text(text).encode("utf-8")).hexdigest()


def token_similarity(a: str, b: str) -> float:
    ta = set(normalize_problem_text(a).split())
    tb = set(normalize_problem_text(b).split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def deduplicate(
    items: List[ProblemItem],
    reference_texts: Optional[Sequence[str]] = None,
    similarity_threshold: float = 0.9,
) -> List[ProblemItem]:
    seen = set()
    kept: List[ProblemItem] = []
    for item in items:
        fp = problem_fingerprint(item.prompt)
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(item)

    if reference_texts:
        ref_norm = [normalize_problem_text(t) for t in reference_texts]
        filtered: List[ProblemItem] = []
        for item in kept:
            norm = normalize_problem_text(item.prompt)
            is_dup = any(
                token_similarity(norm, r) >= similarity_threshold for r in ref_norm
            )
            if not is_dup:
                filtered.append(item)
        return filtered

    return kept


def _make_id(dataset: str, idx: int, subset: Optional[str] = None) -> str:
    base = f"{dataset}-{idx}"
    return f"{subset}-{base}" if subset else base


def parse_gsm8k(record: Dict[str, Any], idx: int) -> Optional[ProblemItem]:
    q = record.get("question", "")
    a = record.get("answer", "")
    if not q or not a:
        return None
    m = re.search(r"####\s*(.+)", a)
    gt = m.group(1).strip() if m else a.strip()
    return ProblemItem(
        problem_id=_make_id("gsm8k", idx),
        prompt=q.strip(),
        ground_truth=gt,
        metadata={"dataset": "gsm8k", "raw_answer": a},
    )


def parse_math(record: Dict[str, Any], idx: int) -> Optional[ProblemItem]:
    q = record.get("problem", "")
    sol = record.get("solution", "")
    if not q or not sol:
        return None
    from ..core.verifier import extract_answer_with_boxed
    gt = extract_answer_with_boxed(sol)
    if gt is None:
        lines = [ln.strip() for ln in sol.strip().splitlines() if ln.strip()]
        gt = lines[-1] if lines else sol.strip()
    return ProblemItem(
        problem_id=_make_id("math", idx),
        prompt=q.strip(),
        ground_truth=gt,
        metadata={
            "dataset": "math",
            "level": record.get("level", ""),
            "type": record.get("type", ""),
            "raw_solution": sol,
        },
    )


def parse_numina(record: Dict[str, Any], idx: int, subset: str) -> Optional[ProblemItem]:
    q = record.get("problem", "")
    sol = record.get("solution", "")
    if not q or not sol:
        return None
    from ..core.verifier import extract_answer_with_boxed
    gt = extract_answer_with_boxed(sol)
    if gt is None:
        lines = [ln.strip() for ln in sol.strip().splitlines() if ln.strip()]
        gt = lines[-1] if lines else sol.strip()
    return ProblemItem(
        problem_id=_make_id("numina", idx, subset),
        prompt=q.strip(),
        ground_truth=gt,
        metadata={"dataset": "numina", "subset": subset, "raw_solution": sol},
    )


NUMINA_SUBSETS = [
    "cn_k12",
    "synthetic_math",
    "orca_math",
    "olympiads",
    "synthetic_amc",
    "aops_forum",
    "amc_aime",
]


@dataclass
class DatasetConfig:
    name: str
    train_size: Optional[int] = None
    test_size: Optional[int] = None
    seed: int = 42
    cache_dir: Optional[str] = None
    similarity_threshold: float = 0.9


def load_dataset_items(cfg: DatasetConfig) -> Dict[str, List[ProblemItem]]:
    name = cfg.name.lower()
    if name == "gsm8k":
        return _load_gsm8k(cfg)
    if name == "math":
        return _load_math(cfg)
    if name in ("numina", "numinamath", "numinamath-cot"):
        return _load_numina(cfg)
    raise ValueError(f"Unknown dataset: {cfg.name}")


def _load_gsm8k(cfg: DatasetConfig) -> Dict[str, List[ProblemItem]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", cache_dir=cfg.cache_dir)
    train = [parse_gsm8k(r, i) for i, r in enumerate(ds["train"])]
    test = [parse_gsm8k(r, i) for i, r in enumerate(ds["test"])]
    train = [x for x in train if x]
    test = [x for x in test if x]

    test_texts = [t.prompt for t in test]
    train = deduplicate(train, reference_texts=test_texts, similarity_threshold=cfg.similarity_threshold)
    test = deduplicate(test, similarity_threshold=cfg.similarity_threshold)

    if cfg.train_size:
        train = train[: cfg.train_size]
    if cfg.test_size:
        test = test[: cfg.test_size]
    return {"train": train, "test": test}


def _load_math(cfg: DatasetConfig) -> Dict[str, List[ProblemItem]]:
    from datasets import load_dataset

    ds = load_dataset("EleutherAI/hendrycks_math", "algebra", cache_dir=cfg.cache_dir)
    train = [parse_math(r, i) for i, r in enumerate(ds["train"])]
    test = [parse_math(r, i) for i, r in enumerate(ds["test"])]
    train = [x for x in train if x]
    test = [x for x in test if x]

    test_texts = [t.prompt for t in test]
    train = deduplicate(train, reference_texts=test_texts, similarity_threshold=cfg.similarity_threshold)
    test = deduplicate(test, similarity_threshold=cfg.similarity_threshold)

    if cfg.train_size:
        train = train[: cfg.train_size]
    if cfg.test_size:
        test = test[: cfg.test_size]
    return {"train": train, "test": test}


def _load_numina(cfg: DatasetConfig) -> Dict[str, List[ProblemItem]]:
    from datasets import load_dataset

    all_items: List[ProblemItem] = []
    for subset in NUMINA_SUBSETS:
        try:
            ds = load_dataset("AI-MO/NuminaMath-CoT", subset, cache_dir=cfg.cache_dir)
        except Exception:
            continue
        split = "train" if "train" in ds else list(ds.keys())[0]
        for i, r in enumerate(ds[split]):
            item = parse_numina(r, i, subset)
            if item:
                all_items.append(item)

    ref_texts: List[str] = []
    for name in ("gsm8k", "math"):
        try:
            ref = load_dataset_items(DatasetConfig(name=name, cache_dir=cfg.cache_dir))
            ref_texts.extend([t.prompt for t in ref["test"]])
        except Exception:
            pass

    all_items = deduplicate(all_items, reference_texts=ref_texts, similarity_threshold=cfg.similarity_threshold)

    rng = random.Random(cfg.seed)
    rng.shuffle(all_items)
    test_size = cfg.test_size or 2000
    train_size = cfg.train_size or 8000
    test = all_items[:test_size]
    train = all_items[test_size : test_size + train_size]
    return {"train": train, "test": test}
