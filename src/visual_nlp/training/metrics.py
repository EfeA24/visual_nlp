"""Gradient-free evaluation metrics. Token accuracy is not sufficient."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import torch

from visual_nlp.training.losses import box_iou, cxcywh_to_xyxy


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(prev[j] + 1, current[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = current
    return prev[-1]


def cer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    pairs = list(zip(refs, hyps))
    if not pairs:
        return 0.0
    num = sum(_edit_distance(list(ref), list(hyp)) for ref, hyp in pairs)
    den = sum(max(len(ref), 1) for ref, _ in pairs)
    return num / den


def wer(refs: Iterable[str], hyps: Iterable[str]) -> float:
    pairs = list(zip(refs, hyps))
    if not pairs:
        return 0.0
    num = sum(_edit_distance(ref.split(), hyp.split()) for ref, hyp in pairs)
    den = sum(max(len(ref.split()), 1) for ref, _ in pairs)
    return num / den


def exact_match(refs: Iterable[str], hyps: Iterable[str]) -> float:
    pairs = list(zip(refs, hyps))
    if not pairs:
        return 0.0
    return sum(ref == hyp for ref, hyp in pairs) / len(pairs)


def digit_exact_match(refs: Iterable[str], hyps: Iterable[str]) -> float:
    pairs = [(ref, hyp) for ref, hyp in zip(refs, hyps) if any(ch.isdigit() for ch in ref)]
    if not pairs:
        return 0.0
    return sum(ref == hyp for ref, hyp in pairs) / len(pairs)


_TURKISH_CASE = set("Iİıi")
_PUNCT = set(" .,;:!?-_/()[]{}'\"%&@#+*=<>€$£₺°")


def _align_edits(ref: Sequence[str], hyp: Sequence[str]) -> list[tuple[str, str | None, str | None]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    i, j = n, m
    ops: list[tuple[str, str | None, str | None]] = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = "eq" if ref[i - 1] == hyp[j - 1] else "sub"
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1] if j > 0 else None))
            j -= 1
    ops.reverse()
    return ops


def reader_error_summary(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
    counts = {
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "digit_errors": 0,
        "turkish_case_errors": 0,
        "punctuation_errors": 0,
        "other_substitutions": 0,
    }
    for ref, hyp in zip(refs, hyps):
        for op, a, b in _align_edits(list(ref), list(hyp)):
            if op == "eq":
                continue
            if op == "ins":
                counts["insertions"] += 1
                continue
            if op == "del":
                counts["deletions"] += 1
                continue
            counts["substitutions"] += 1
            if (a is not None and a.isdigit()) or (b is not None and b.isdigit()):
                counts["digit_errors"] += 1
            elif a in _TURKISH_CASE and b in _TURKISH_CASE:
                counts["turkish_case_errors"] += 1
            elif (a in _PUNCT) or (b in _PUNCT):
                counts["punctuation_errors"] += 1
            else:
                counts["other_substitutions"] += 1
    return {key: float(value) for key, value in counts.items()}


def reader_metrics(refs: Sequence[str], hyps: Sequence[str]) -> dict[str, float]:
    return {
        "cer": cer(refs, hyps),
        "wer": wer(refs, hyps),
        "exact": exact_match(refs, hyps),
        "digit_exact": digit_exact_match(refs, hyps),
        **reader_error_summary(refs, hyps),
    }


def detection_prf(
    pred_boxes: Sequence[torch.Tensor],
    pred_scores: Sequence[torch.Tensor],
    target_boxes: Sequence[torch.Tensor],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
    box_format: str = "cxcywh",
) -> dict[str, float]:
    tp = fp = fn = 0
    for preds, scores, targets in zip(pred_boxes, pred_scores, target_boxes):
        keep = scores >= score_threshold
        preds = preds[keep]
        matched_gt = set()
        if preds.numel() and targets.numel():
            pred_xyxy = preds if box_format == "xyxy" else cxcywh_to_xyxy(preds)
            tgt_xyxy = targets if box_format == "xyxy" else cxcywh_to_xyxy(targets)
            ious, _ = box_iou(pred_xyxy, tgt_xyxy)
            for pred_idx in range(preds.size(0)):
                best_gt = int(ious[pred_idx].argmax().item()) if targets.size(0) else -1
                if best_gt >= 0 and ious[pred_idx, best_gt] >= iou_threshold and best_gt not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1
        else:
            fp += int(preds.size(0))
        fn += int(targets.size(0) - len(matched_gt))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def span_f1(gold: Sequence[tuple[int, int, str]], pred: Sequence[tuple[int, int, str]]) -> dict[str, float]:
    gold_set = Counter(gold)
    pred_set = Counter(pred)
    tp = sum((gold_set & pred_set).values())
    fp = sum((pred_set - gold_set).values())
    fn = sum((gold_set - pred_set).values())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"field_precision": precision, "field_recall": recall, "field_f1": f1}


def value_exact(gold_values: Sequence[str], pred_values: Sequence[str]) -> float:
    if not gold_values:
        return 0.0
    return sum(g == p for g, p in zip(gold_values, pred_values)) / len(gold_values)
