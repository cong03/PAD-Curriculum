# -*- coding: utf-8 -*-
"""
Rule-Based Verifier for Mathematical Reasoning.

Extracts final answers from model-generated solutions and performs exact
or numerical equivalence matching. Used for both RL outcome rewards and
curriculum trajectory correctness filtering.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional


def _extract_boxed_content(text: str) -> Optional[str]:
    """Extracts content inside the outermost \\boxed{...}, handling nested braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    brace_start = text.find("{", idx)
    if brace_start == -1:
        return None
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : i].strip()
    return None


def strip_latex_boxes(text: str) -> str:
    """Recursively removes \\boxed{...} wrappers while preserving internal content."""
    while "\\boxed" in text:
        content = _extract_boxed_content(text)
        if content is None:
            break
        idx = text.rfind("\\boxed")
        brace_start = text.find("{", idx)
        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    text = text[:idx] + content + text[i + 1 :]
                    break
    return text


def extract_answer_with_boxed(text: str) -> Optional[str]:
    """Extracts boxed answer content if present."""
    return _extract_boxed_content(text)


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extracts final answer from solution text.
    Priority:
      1) Explicit markers (e.g. 'final answer is', 'answer:')
      2) LaTeX \\boxed{...}
      3) Last non-empty line
    """
    if not text:
        return None

    patterns = [
        r"(?:final\s+answer\s*(?:is)?\s*[:=]?\s*)(.*)",
        r"(?:answer\s*[:=]\s*)(.*)",
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            candidate = m.group(1).strip()
            if candidate:
                return _clean_answer(candidate)

    boxed = extract_answer_with_boxed(text)
    if boxed is not None:
        return _clean_answer(boxed)

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return _clean_answer(lines[-1])

    return None


def _clean_answer(ans: str) -> str:
    ans = strip_latex_boxes(ans)
    ans = ans.strip().strip(".").strip()
    return ans


def _parse_number(s: str):
    s = s.strip()
    if not s:
        return None
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return None
        return num / den
    s_clean = s.replace(",", "").replace(" ", "")
    try:
        if "." in s_clean or "e" in s_clean or "E" in s_clean:
            return float(s_clean)
        return int(s_clean)
    except ValueError:
        return None


def normalize_text(s: str) -> str:
    """Normalizes whitespace, LaTeX spaces, and casing."""
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"\\,|\\;|\\quad|\\qquad", " ", s)
    s = re.sub(r"\s+", "", s)
    s = s.lower()
    return s


def exact_match(response: str, ground_truth: str) -> bool:
    return normalize_text(response) == normalize_text(ground_truth)


def numeric_match(response: str, ground_truth: str, tol: float = 1e-6) -> bool:
    if normalized_exact_match_with_factors(response, ground_truth):
        return True
    n1 = _parse_number(response)
    n2 = _parse_number(ground_truth)
    if n1 is None or n2 is None:
        return False
    return abs(n1 - n2) <= tol * max(1.0, abs(n2))


def normalized_exact_match_with_factors(response: str, ground_truth: str) -> bool:
    def norm_num(s: str) -> str:
        s = normalize_text(s)
        return s.replace("−", "-").replace("×", "*").replace("÷", "/")

    return norm_num(response) == norm_num(ground_truth)


@dataclass
class RuleBasedVerifier:
    """Rule-based answer verification returning binary correctness (0 or 1)."""

    tol: float = 1e-6
    exact_only: bool = False
    answer_extractor: Callable[[str], Optional[str]] = extract_final_answer

    def verify(self, response: str, ground_truth: str) -> bool:
        if not response or not ground_truth:
            return False
        pred = self.answer_extractor(response)
        if pred is None:
            return False
        return self.match(pred, ground_truth)

    def match(self, prediction: str, ground_truth: str) -> bool:
        if exact_match(prediction, ground_truth):
            return True
        if self.exact_only:
            return False
        return numeric_match(prediction, ground_truth, self.tol)

    def __call__(self, response: str, ground_truth: str) -> int:
        return int(self.verify(response, ground_truth))


def verify_batch(
    verifier: RuleBasedVerifier,
    responses: List[str],
    ground_truths: List[str],
) -> List[int]:
    assert len(responses) == len(ground_truths)
    return [verifier(resp, gt) for resp, gt in zip(responses, ground_truths)]
