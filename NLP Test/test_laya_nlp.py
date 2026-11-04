"""Laya NLP choice benchmark using identical dataset examples as Gemma/JEV."""
import argparse
from nlp_utils import DATASETS, load_examples, prompt_for_choice, run_benchmark


def predict_laya(agent, example):
    state = {'user_request': prompt_for_choice(example), 'assistant_response_so_far': ''}
    questions = {'draft_selection': {
        'type': 'choice',
        'instructions': 'Choose the single best option for the user request. Return its option label.',
        'criteria': example['options']}}
    result = agent.predict(state, questions)
    answer = result['answers']['draft_selection']
    return (str(answer['choice']), str(answer['choice']),
            answer.get('answer_confidence', answer.get('confidence')),
            answer.get('probabilities', {}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required = True)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--split', default='validation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--question-id', type=int)
    p.add_argument('--model', default='convaiinnovations/laya')
    args = p.parse_args()
    examples = load_examples(args.dataset, args.split, args.count, args.seed, args.question_id)
    import laya
    print('Loading Laya once...')
    agent = laya.load(args.model)
    run_benchmark(examples, 'laya', lambda ex: predict_laya(agent, ex),
                  dict(dataset=args.dataset, backend='laya', split=args.split,
                       seed=args.seed, model=args.model))


if __name__ == '__main__':
    main()
