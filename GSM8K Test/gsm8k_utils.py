"""Shared GSM8K data and evaluation utilities (no model dependencies)."""
import csv
import json
import random
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
NUM_RE = re.compile(r"(?<![\w.])" + NUMBER + r"(?!\w|\.\d)")


def normalize_number(value):
    """Return canonical numeric string, or None for nonnumeric values."""
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        return None
    try:
        n = Decimal(s)
        return format(n.normalize(), 'f') if n != 0 else '0'
    except InvalidOperation:
        return None


def gold_answer(solution):
    """GSM8K's final answer is after its #### delimiter."""
    if '####' not in solution:
        raise ValueError('GSM8K answer has no #### delimiter')
    value = solution.rsplit('####', 1)[-1].strip()
    normalized = normalize_number(value)
    if normalized is None:
        raise ValueError(f'Unrecognized GSM8K gold answer: {value!r}')
    return normalized


def extract_prediction(text):
    """Prefer explicit ####/final-answer markers, else last number in text.

    Last-number fallback is heuristic: mark its provenance for later auditing.
    """
    if not text:
        return None, 'empty'
    if '####' in text:
        tail = text.rsplit('####', 1)[-1].strip()
        match = NUM_RE.search(tail)
        if match:
            return normalize_number(match.group()), 'hash_delimiter'
    labeled = re.findall(r'(?:final answer|answer is|answer:)\s*\$?\s*(' + NUMBER + r')', text, flags=re.I)
    if labeled:
        return normalize_number(labeled[-1]), 'answer_label'
    matches = NUM_RE.findall(text)
    if matches:
        return normalize_number(matches[-1]), 'last_number_heuristic'
    return None, 'no_number'


def load_questions(split, count, seed):
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=split)
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    if count > len(ds):
        raise ValueError(f'Asked for {count} samples, split has {len(ds)}')
    return [(idx, ds[int(idx)]['question'], gold_answer(ds[int(idx)]['answer']))
            for idx in indices[:count]]


def output_paths(model):
    now = datetime.now()
    folder = ROOT / 'results' / now.strftime('%Y-%m-%d')
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime('%Y%m%d_%H%M%S_%f')
    return folder / f'gsm8k_{model}_{stamp}.csv', folder / f'gsm8k_{model}_{stamp}_summary.json'


def save_results(rows, csv_path, summary_path, config):
    if not rows:
        raise ValueError('No results to save')
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    correct = sum(r['status'] == 'correct' for r in rows)
    wrong = sum(r['status'] == 'wrong' for r in rows)
    invalid = len(rows) - correct - wrong
    summary = dict(config, count=len(rows), correct=correct, wrong=wrong,
                   invalid_or_error=invalid, accuracy=correct / len(rows),
                   mean_seconds=sum(float(r['seconds']) for r in rows) / len(rows))
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def make_distractors(correct, seed, count=3):
    """Deterministic plausible nearby numbers, unique and never correct."""
    gold = Decimal(correct)
    offsets = [Decimal(x) for x in (-10, -5, -3, -2, -1, 1, 2, 3, 5, 10)]
    rng = random.Random(seed)
    rng.shuffle(offsets)
    options = []
    for offset in offsets:
        v = normalize_number(str(gold + offset))
        if v != correct and v not in options:
            options.append(v)
        if len(options) == count:
            return options
    raise RuntimeError('Could not construct distinct distractors')
