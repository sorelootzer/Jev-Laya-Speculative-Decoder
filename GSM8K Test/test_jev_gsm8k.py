"""JEV GSM8K choice benchmark, paired with test_laya_gsm8k.py sampling/options."""
import argparse
import random
import time
from gsm8k_utils import (load_questions, gold_answer, make_distractors,
                         output_paths, save_results)
from jev_selector import JEVSelector


def predict_choice(selector, question, options):
    # Exact prompt and state format used in the Laya GSM8K benchmark.
    state = {'user_request': question, 'assistant_response_so_far': ''}
    query = {'draft_selection': {
        'type': 'choice',
        'instructions': ('Select the mathematically correct numerical answer to the '
                         'user question. Choose exactly one of the given options.'),
        'criteria': options}}
    result = selector.predict(state, query)
    answer = result['answers']['draft_selection']
    return str(answer['choice']), answer, result


def select_questions(split, count, seed, question_id):
    if question_id is None:
        return load_questions(split, count, seed)
    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split=split)
    if not 0 <= question_id < len(ds):
        raise ValueError(f'question-id must be between 0 and {len(ds)-1}')
    record = ds[question_id]
    return [(question_id, record['question'], gold_answer(record['answer']))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--split', choices=['train', 'test'], default='test')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--question-id', type=int, default=None)
    parser.add_argument('--model', default='jev-1.13')
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    if args.count < 1:
        parser.error('count must be positive')
    questions = select_questions(args.split, args.count, args.seed, args.question_id)
    # JEV is hosted. This prompts once for a hidden API key and reuses it.
    selector = JEVSelector(k=2, branching_factor=2, model_name=args.model,
                           timeout=args.timeout)
    csv_path, summary_path = output_paths('jev')
    rows = []
    for i, (idx, question, expected) in enumerate(questions, 1):
        distractors = make_distractors(expected, seed=args.seed * 100000 + idx)
        choices = [expected, *distractors]
        random.Random(args.seed * 100000 + idx + 1).shuffle(choices)
        options = dict(zip('ABCD', choices))
        start = time.perf_counter()
        selected, predicted, confidence, probabilities, error = '', None, None, {}, ''
        resolved_model = args.model
        try:
            selected, answer, result = predict_choice(selector, question, options)
            if selected not in options:
                raise ValueError(f'JEV returned invalid choice {selected!r}')
            predicted = options[selected]
            confidence = answer.get('answer_confidence', answer.get('confidence'))
            probabilities = answer.get('probabilities', {})
            resolved_model = result.get('model_version', result.get('model', args.model))
        except Exception as exc:
            # Never record HTTP bodies, headers, or secrets.
            error = f'{type(exc).__name__}: {exc}'
        status = 'error' if error else ('correct' if predicted == expected else 'wrong')
        rows.append(dict(question_id=idx, question=question, expected=expected,
                         predicted=predicted or '', status=status,
                         options=str(options), selected_option=selected,
                         answer_confidence=confidence, probabilities=str(probabilities),
                         seconds=round(time.perf_counter()-start, 4), error=error,
                         selector_backend='jev', model_version=resolved_model))
        print(f'{i}/{len(questions)} ID={idx} {status.upper()} expected={expected} got={predicted}')
        save_results(rows, csv_path, summary_path,
                     dict(model='jev_choice', model_version=args.model,
                          split=args.split, seed=args.seed,
                          distractors='offsets from gold, shuffled'))
    print('Saved:', csv_path, '\nSummary:', summary_path)


if __name__ == '__main__':
    main()
