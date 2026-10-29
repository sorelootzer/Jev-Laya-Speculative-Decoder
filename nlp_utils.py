"""Shared NLP benchmark dataset adapters, scoring, and persistence."""
import csv
import json
import random
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = ('hellaswag', 'sst2', 'cola', 'boolq', 'piqa')
CONFIG = {
    'hellaswag': ('Rowan/hellaswag', None),
    'sst2': ('nyu-mll/glue', 'sst2'),
    'cola': ('nyu-mll/glue', 'cola'),
    'boolq': ('google/boolq', None),
    'piqa': ('regisss/piqa', None),
}


def parse_record(dataset, row, idx):
    """Return a standardized record, or None for unlabeled examples."""
    if dataset == 'hellaswag':
        label = row.get('label')
        if label is None or str(label).strip() not in ('0', '1', '2', '3'):
            return None
        opts = dict(zip('ABCD', row['endings']))
        prompt = row['ctx']
        gold = 'ABCD'[int(label)]
    elif dataset in ('sst2', 'cola'):
        label = row.get('label')
        if label not in (0, 1):
            return None
        opts = {'A': 'negative', 'B': 'positive'} if dataset == 'sst2' else {
            'A': 'grammatically unacceptable', 'B': 'grammatically acceptable'}
        prompt = row['sentence']
        gold = 'AB'[int(label)]
    elif dataset == 'boolq':
        label = row.get('answer')
        if not isinstance(label, bool):
            return None
        opts = {'A': 'No', 'B': 'Yes'}
        prompt = f"Passage: {row['passage']}\nQuestion: {row['question']}"
        gold = 'B' if label else 'A'
    elif dataset == 'piqa':
        label = row.get('label')
        if label not in (0, 1):
            return None
        opts = {'A': row['sol1'], 'B': row['sol2']}
        prompt = row['goal']
        gold = 'AB'[int(label)]
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    return {'question_id': idx, 'prompt': prompt, 'options': opts, 'gold': gold,
            'gold_text': opts[gold], 'dataset': dataset}


def shuffle_options(example, seed):
    """Deterministically shuffle options, remapping the gold label."""
    labels = list(example['options'])
    texts = list(example['options'].values())
    gold_text = example['gold_text']
    random.Random(seed * 1000003 + example['question_id']).shuffle(texts)
    example = dict(example)
    example['options'] = dict(zip(labels, texts))
    example['gold'] = labels[texts.index(gold_text)]
    return example


def load_examples(dataset, split='validation', count=20, seed=42, question_id=None):
    if dataset not in DATASETS:
        raise ValueError(f'Dataset must be one of {DATASETS}')
    if count < 1:
        raise ValueError('count must be positive')
    from datasets import load_dataset
    path, config = CONFIG[dataset]
    try:
        ds = load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
    except Exception as exc:
        raise RuntimeError(f'Unable to load {dataset} ({path}, {split}): {exc}') from exc
    if question_id is not None:
        if not 0 <= question_id < len(ds):
            raise ValueError(f'question-id must be between 0 and {len(ds)-1}')
        record = parse_record(dataset, ds[question_id], question_id)
        if record is None:
            raise ValueError('Selected example has no public gold label; use validation/train')
        return [shuffle_options(record, seed)]
    indices = list(range(len(ds)))
    random.Random(seed).shuffle(indices)
    selected = []
    for idx in indices:
        record = parse_record(dataset, ds[idx], idx)
        if record is not None:
            selected.append(shuffle_options(record, seed))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} labeled examples available; requested {count}')
    return selected


def prompt_for_choice(example):
    task = {
        'hellaswag': 'Choose the most plausible continuation of the context.',
        'sst2': 'Classify the sentiment of this text.',
        'cola': 'Determine whether this sentence is grammatically acceptable.',
        'boolq': 'Answer the question using the passage.',
        'piqa': 'Choose the more sensible solution to the practical goal.',
    }[example['dataset']]
    choices = '\n'.join(f'{key}. {value}' for key, value in example['options'].items())
    return f'{task}\n\n{example["prompt"]}\n\nChoices:\n{choices}\n\nAnswer with only the option letter.'


def parse_choice(text, valid):
    import re
    s = str(text).strip().upper()
    if s in valid:
        return s
    m = re.fullmatch(r'(?:OPTION|ANSWER\s*:?|CHOICE\s*:?|\()?\s*([A-D])\s*[).:]?', s)
    return m.group(1) if m and m.group(1) in valid else None


def output_paths(dataset, backend):
    now = datetime.now()
    folder = ROOT / 'results' / now.strftime('%Y-%m-%d')
    folder.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime('%Y%m%d_%H%M%S_%f')
    name = f'nlp_{dataset}_{backend}_{stamp}'
    return folder / f'{name}.csv', folder / f'{name}_summary.json'


def save_results(rows, csv_path, summary_path, config):
    if not rows:
        raise ValueError('No results to save')
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    correct = sum(r['status'] == 'correct' for r in rows)
    wrong = sum(r['status'] == 'wrong' for r in rows)
    invalid = sum(r['status'] == 'invalid' for r in rows)
    errors = sum(r['status'] == 'error' for r in rows)
    summary = dict(config, count=len(rows), correct=correct, wrong=wrong,
                   invalid=invalid, errors=errors, accuracy=correct/len(rows),
                   mean_seconds=sum(float(r['seconds']) for r in rows)/len(rows))
    if config['dataset'] == 'cola':
        # CoLA convention: MCC, treating invalid/errors as wrong predictions is
        # not statistically valid; report MCC on valid predictions only.
        valid = [r for r in rows if r['status'] in ('correct', 'wrong')]
        if valid:
            from sklearn.metrics import matthews_corrcoef
            summary['mcc_valid_only'] = float(matthews_corrcoef(
                [int(r['gold'] == 'B') for r in valid],
                [int(r['selected_option'] == 'B') for r in valid]))
            summary['mcc_valid_count'] = len(valid)
        else:
            summary['mcc_valid_only'] = None
            summary['mcc_valid_count'] = 0
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def run_benchmark(examples, backend, predict, config):
    csv_path, summary_path = output_paths(config['dataset'], backend)
    import time
    rows = []
    for i, ex in enumerate(examples, 1):
        start = time.perf_counter()
        choice, raw, confidence, probs, error = None, '', None, {}, ''
        try:
            choice, raw, confidence, probs = predict(ex)
            choice = parse_choice(choice, ex['options'])
        except Exception as exc:
            # Do not persist remote HTTP bodies, credentials, or headers.
            error = f'{type(exc).__name__}: {str(exc)[:250]}'
        status = 'error' if error else ('invalid' if choice is None else
                 ('correct' if choice == ex['gold'] else 'wrong'))
        rows.append({'dataset': ex['dataset'], 'question_id': ex['question_id'],
                     'question': ex['prompt'], 'options': json.dumps(ex['options'], ensure_ascii=False),
                     'gold': ex['gold'], 'gold_text': ex['gold_text'],
                     'selected_option': choice or '',
                     'selected_text': ex['options'].get(choice, '') if choice else '',
                     'status': status, 'raw_output': str(raw)[:3000],
                     'answer_confidence': confidence,
                     'probabilities': json.dumps(probs, ensure_ascii=False),
                     'seconds': round(time.perf_counter()-start, 4), 'error': error})
        print(f'{i}/{len(examples)} ID={ex["question_id"]} {status.upper()} '
              f'gold={ex["gold"]} selected={choice}')
        save_results(rows, csv_path, summary_path, config)
    print('Results:', csv_path, '\nSummary:', summary_path)
    return rows
