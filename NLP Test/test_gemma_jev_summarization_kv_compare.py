"""Paired long-form generation benchmark: Gemma greedy vs Gemma-JEV.

Run from the project root with this file inside 'NLP Test/'. Dataset identifiers
may need overrides as Hugging Face hosting/access policies change.
"""
import argparse
import csv
import json
import random
import re
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / 'gemma_jev.py').exists() else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_IDS = {
    'samsum': ('knkarthick/samsum', None),
    'cnn_dailymail': ('abisee/cnn_dailymail', '3.0.0'),
    'eli5': ('sentence-transformers/eli5', None),
}


def extract_example(dataset, row, idx):
    """Convert common HF dataset schemas; fail rather than silently mislabel."""
    if dataset == 'samsum':
        source = row.get('dialogue')
        reference = row.get('summary')
    elif dataset == 'cnn_dailymail':
        source = row.get('article')
        reference = row.get('highlights')
    elif dataset == 'eli5':
        source = row.get('title') or row.get('question')
        if isinstance(source, dict):
            source = source.get('text')
        answers = row.get('answers') or row.get('answer')
        if isinstance(answers, dict):
            answers = answers.get('text') or answers.get('answers')
        if isinstance(answers, list):
            answers = [x.get('text', '') if isinstance(x, dict) else x for x in answers]
            reference = next((x for x in answers if isinstance(x, str) and x.strip()), None)
        else:
            reference = answers
    else:
        raise ValueError(f'Unsupported dataset: {dataset}')
    if not isinstance(source, str) or not source.strip():
        return None
    if not isinstance(reference, str) or not reference.strip():
        return None
    return {'question_id': idx, 'source': source.strip(), 'reference': reference.strip()}


def load_examples(args):
    from datasets import load_dataset
    dataset_id, config = DATASET_IDS[args.dataset]
    dataset_id = args.hf_dataset or dataset_id
    config = args.hf_config if args.hf_config is not None else config
    try:
        if args.dataset == 'eli5' and not args.hf_dataset:
            raise ValueError('ELI5 hosting/schema varies. Supply --hf-dataset with a labeled ELI5 dataset repository.')
        ds = load_dataset(dataset_id, config, split=args.split) if config else load_dataset(dataset_id, split=args.split)
    except Exception as exc:
        raise RuntimeError(f'Unable to load {dataset_id!r} split={args.split!r}. '
                           'Check access, dataset ID, configuration and split; '
                           'override using --hf-dataset / --hf-config / --split. '
                           f'Original error: {exc}') from exc
    if args.question_id is not None:
        if not 0 <= args.question_id < len(ds):
            raise ValueError(f'--question-id must be in [0,{len(ds)-1}]')
        indices = [args.question_id]
    else:
        indices = list(range(len(ds)))
        random.Random(args.seed).shuffle(indices)
    examples = []
    for idx in indices:
        ex = extract_example(args.dataset, ds[idx], idx)
        if ex:
            examples.append(ex)
        if len(examples) >= args.count:
            break
    if not examples:
        raise ValueError('No labeled examples found; check dataset columns and split.')
    return examples


def make_prompt(dataset, source):
    if dataset == 'samsum':
        return ('Summarize the following conversation accurately in 2-4 complete sentences. '
                'Do not invent details.\n\nConversation:\n' + source + '\n\nSummary:')
    if dataset == 'cnn_dailymail':
        return ('Summarize the following news article accurately in 2-5 complete sentences. '
                'Include the main facts without adding unsupported claims.\n\nArticle:\n' + source + '\n\nSummary:')
    return ('Answer the following question with a clear, factual explanation in one short paragraph. '
            'Do not invent details.\n\nQuestion:\n' + source + '\n\nAnswer:')


def fit_prompt(decoder, dataset, source, max_tokens, n_ctx, max_input_tokens):
    """Token-budget source truncation, preserving the task framing."""
    prompt = make_prompt(dataset, source)
    tokens = decoder.prompt_tokens(prompt)
    allowance = n_ctx - max_tokens - 8
    if allowance <= 0:
        raise ValueError('n_ctx too small for requested max output tokens')
    if len(tokens) <= allowance and len(tokens) <= max_input_tokens:
        return prompt, False, len(tokens)
    # Binary search on source characters, measuring the actual Gemma tokenization.
    lo, hi, best = 0, len(source), None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = make_prompt(dataset, source[:mid].rstrip())
        n = len(decoder.prompt_tokens(candidate))
        if n <= min(allowance, max_input_tokens):
            best = (candidate, n)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        raise ValueError('Task instruction alone exceeds the input budget')
    return best[0], True, best[1]


def rouge_scores(prediction, reference):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return {name: score.fmeasure for name, score in scorer.score(reference, prediction).items()}


def mean(values):
    return sum(values) / len(values) if values else None


def evaluate(args, examples, decoder, selector):
    out = Path(args.results_dir) / datetime.now().strftime('%Y-%m-%d')
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    prefix = out / f'paired_{args.dataset}_{args.decoder}_{Path(args.model).stem}_k{args.k}_{stamp}'
    csv_path = Path(str(prefix) + '.csv')
    summary_path = Path(str(prefix) + '_summary.json')
    rows = []
    bertscore = None
    if args.bertscore:
        from bert_score import score as bertscore
    for num, ex in enumerate(examples, 1):
        row = {'dataset': args.dataset, 'question_id': ex['question_id'],
               'source': ex['source'], 'reference': ex['reference'],
               'model': str(Path(args.model).resolve()), 'k': args.k, 'decoder': args.decoder}
        try:
            prompt, truncated, input_tokens = fit_prompt(decoder, args.dataset, ex['source'],
                                                          args.max_tokens, args.n_ctx, args.max_input_tokens)
            row.update(prompt=prompt, input_truncated=truncated, input_tokens=input_tokens)
            baseline = decoder.greedy_baseline(prompt, max_tokens=args.max_tokens)
            guided = decoder.run(prompt, args.k, selector, max_tokens=args.max_tokens,
                                 verbose=False, force_first_jev=args.force_first_jev)
            row.update(gemma_response=baseline['response'], gemma_jev_response=guided['response'],
                       gemma_seconds=baseline['elapsed_seconds'], gemma_jev_seconds=guided['elapsed_seconds'],
                       gemma_tokens=baseline['output_tokens'], gemma_jev_tokens=guided['output_tokens'],
                       gemma_stop=baseline['termination_reason'], gemma_jev_stop=guided['termination_reason'],
                       jev_calls=guided['jev_calls'], jev_overrides=guided['jev_overrides'],
                       jev_seconds=guided['jev_seconds'], jev_bypass_reasons=json.dumps(guided['jev_bypass_reasons']),
                       kv_metrics=json.dumps(guided.get('kv_metrics', {})),
                       gemma_tps=baseline['output_tokens']/max(baseline['elapsed_seconds'], 1e-9),
                       gemma_jev_tps=guided['output_tokens']/max(guided['elapsed_seconds'], 1e-9))
            for name, text in [('gemma', baseline['response']), ('gemma_jev', guided['response'])]:
                for metric, value in rouge_scores(text, ex['reference']).items():
                    row[f'{name}_{metric}'] = value
            if bertscore is not None:
                for name, text in [('gemma', baseline['response']), ('gemma_jev', guided['response'])]:
                    _, _, f1 = bertscore([text or ' '], [ex['reference']], lang='en', verbose=False)
                    row[f'{name}_bertscore_f1'] = float(f1[0])
            row['status'] = 'ok'
        except Exception as exc:
            row['status'] = 'error'
            row['error'] = f'{type(exc).__name__}: {exc}'
            if args.fail_fast:
                raise
        rows.append(row)
        print(f'[{num}/{len(examples)}] id={ex["question_id"]} status={row["status"]} '
              f'JEV calls={row.get("jev_calls", 0)}')
        # Persist after each example to protect lengthy/paid runs.
        fields = list(dict.fromkeys(k for item in rows for k in item))
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    valid = [r for r in rows if r['status'] == 'ok']
    summary = {'dataset': args.dataset, 'count': len(rows), 'completed': len(valid),
               'errors': len(rows)-len(valid), 'model': str(args.model), 'jev_model': args.jev_model,
               'k': args.k, 'b': 2, 'max_tokens': args.max_tokens, 'decoder': args.decoder,
               'jev_participation_questions': sum(r['jev_calls'] > 0 for r in valid),
               'jev_total_calls': sum(r['jev_calls'] for r in valid),
               'jev_total_overrides': sum(r['jev_overrides'] for r in valid),
               'input_truncated_count': sum(bool(r['input_truncated']) for r in valid),
               'kv_metrics_total': {key: sum(json.loads(r['kv_metrics']).get(key, 0) for r in valid)
                                    for key in (json.loads(valid[0]['kv_metrics']) if valid else {})}}
    for name in ('gemma', 'gemma_jev'):
        for metric in ('rouge1', 'rouge2', 'rougeL', 'seconds', 'tokens', 'tps'):
            summary[f'{name}_mean_{metric}'] = mean([r[f'{name}_{metric}'] for r in valid])
        if args.bertscore:
            summary[f'{name}_mean_bertscore_f1'] = mean([r[f'{name}_bertscore_f1'] for r in valid])
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('CSV:', csv_path, '\nSummary:', summary_path)
    return rows, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=tuple(DATASET_IDS), required=True)
    p.add_argument('--decoder', choices=('kv', 'optimized', 'original'), default='kv',
                   help='Compare KV optimization with original decoder using matched arguments')
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', default='validation')
    p.add_argument('--question-id', type=int)
    p.add_argument('--model', type=Path, default=PROJECT_ROOT / 'models' / 'Gemma4' / 'gemma-4-E4B-it-Q4_K_M.gguf')
    p.add_argument('--jev-model', default='jev-latest')
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--n-ctx', type=int, default=2048)
    p.add_argument('--max-input-tokens', type=int, default=1400)
    p.add_argument('--results-dir', type=Path, default=PROJECT_ROOT / 'results')
    p.add_argument('--hf-dataset', help='Override Hugging Face dataset ID')
    p.add_argument('--hf-config', help='Override Hugging Face dataset configuration')
    p.add_argument('--bertscore', action='store_true', help='Compute optional BERTScore (slow; downloads model)')
    p.add_argument('--force-first-jev', action='store_true', help='Force first eligible JEV decision (off by default)')
    p.add_argument('--fail-fast', action='store_true')
    args = p.parse_args()
    if args.count < 1 or args.k < 1 or 2**args.k > 24 or args.max_tokens < args.k:
        p.error('Require count >= 1, 1 <= k with 2**k <= 24, max-tokens >= k')
    if args.max_tokens > 1024:
        p.error('Maximum output is 1024 tokens')
    if args.n_ctx <= args.max_tokens + 16 or args.max_input_tokens < 64:
        p.error('Insufficient context or input-token budget')
    if not args.model.is_file():
        p.error(f'GGUF model not found: {args.model}')
    if args.decoder == 'kv':
        import gemma_jev_kv as decoder
    elif args.decoder == 'optimized':
        import gemma_jev_kv_optimized as decoder
    else:
        import gemma_jev as decoder
    from jev_selector import JEVSelector
    decoder.N_CTX = args.n_ctx
    decoder.MAX_OUTPUT_TOKENS = args.max_tokens
    decoder.MODEL_PATH = args.model
    decoder.load_gemma(args.model)
    selector = JEVSelector(k=args.k, branching_factor=2, model_name=args.jev_model, timeout=60)
    examples = load_examples(args)
    evaluate(args, examples, decoder, selector)


if __name__ == '__main__':
    main()
