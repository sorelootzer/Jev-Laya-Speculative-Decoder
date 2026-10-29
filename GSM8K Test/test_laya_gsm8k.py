"""Standalone Laya GSM8K multiple-choice benchmark (no Gemma dependency)."""
import argparse
import random
import time
from gsm8k_utils import load_questions, make_distractors, output_paths, save_results


def predict_choice(agent, question, options):
    # Match the current Laya agent.predict(state, questions) choice interface.
    state = {'user_request': question, 'assistant_response_so_far': ''}
    query = {'draft_selection': {
        'type': 'choice',
        'instructions': ('Select the mathematically correct numerical answer to the '
                         'user question. Choose exactly one of the given options.'),
        'criteria': options}}
    result = agent.predict(state, query)
    answer = result['answers']['draft_selection']
    return str(answer['choice']), answer, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', default='convaiinnovations/laya')
    args = p.parse_args()
    if args.count < 1:
        p.error('count must be positive')
    import laya
    questions = load_questions(args.split, args.count, args.seed)
    print('Loading Laya once...')
    agent = laya.load(args.model)
    csv_path, summary_path = output_paths('laya')
    rows = []
    for i, (idx, question, expected) in enumerate(questions, 1):
        distractors = make_distractors(expected, seed=args.seed * 100000 + idx)
        choices = [expected, *distractors]
        random.Random(args.seed * 100000 + idx + 1).shuffle(choices)
        options = dict(zip('ABCD', choices))
        start = time.perf_counter()
        selected, predicted, confidence, probabilities, error = '', None, None, {}, ''
        try:
            selected, answer, result = predict_choice(agent, question, options)
            if selected not in options:
                raise ValueError(f'Laya returned invalid choice {selected!r}')
            predicted = options[selected]
            confidence = answer.get('answer_confidence')
            probabilities = answer.get('probabilities', {})
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        status = 'error' if error else ('correct' if predicted == expected else 'wrong')
        rows.append(dict(question_id=idx, question=question, expected=expected,
                         predicted=predicted or '', status=status,
                         options=str(options), selected_option=selected,
                         answer_confidence=confidence, probabilities=str(probabilities),
                         seconds=round(time.perf_counter()-start, 4), error=error))
        print(f'{i}/{len(questions)} ID={idx} {status.upper()} expected={expected} got={predicted}')
        save_results(rows, csv_path, summary_path,
                     dict(model='laya_choice', split=args.split, seed=args.seed,
                          distractors='offsets from gold, shuffled'))
    print('Saved:', csv_path, '\nSummary:', summary_path)


if __name__ == '__main__':
    main()
