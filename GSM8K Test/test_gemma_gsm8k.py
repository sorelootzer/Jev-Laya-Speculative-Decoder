"""Standalone Gemma 4 GGUF GSM8K benchmark (no Laya dependency)."""
import argparse
import time
from pathlib import Path
import numpy as np
from gsm8k_utils import load_questions, extract_prediction, output_paths, save_results

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models' / 'Gemma4' / 'gemma-4-E4B-it-Q4_K_M.gguf'


def generate(llm, question, max_tokens, n_ctx):
    prompt = f'<|turn>user\n{question}\nGive your final numerical answer after "####".<turn|>\n<|turn>model\n'
    context = list(llm.tokenize(prompt.encode(), add_bos=True, special=True))
    eot = list(llm.tokenize(b'<turn|>', add_bos=False, special=True))
    if len(eot) != 1:
        raise RuntimeError('Expected single Gemma 4 EOT token')
    stops = {int(llm.token_eos()), int(eot[0])}
    if len(context) + max_tokens > n_ctx:
        raise ValueError('Prompt and max_tokens exceed n_ctx')
    llm.reset()
    llm.eval(context)
    generated = []
    stop = 'max_tokens'
    for _ in range(max_tokens):
        token = int(np.argmax(llm.scores[llm.n_tokens - 1]))
        if token in stops:
            stop = 'end_of_turn' if token == eot[0] else 'eos'
            break
        generated.append(token)
        llm.eval([token])
    return llm.detokenize(generated).decode('utf-8', errors='replace'), stop


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--split', choices=['train', 'test'], default='test')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--n-ctx', type=int, default=2048)
    p.add_argument('--max-tokens', type=int, default=1024)
    p.add_argument('--gpu-layers', type=int, default=0)
    args = p.parse_args()
    if args.count < 1 or args.max_tokens < 1:
        p.error('count and max-tokens must be positive')
    if not args.model.is_file():
        p.error(f'GGUF not found: {args.model}')
    from llama_cpp import Llama
    questions = load_questions(args.split, args.count, args.seed)
    llm = Llama(model_path=str(args.model), n_ctx=args.n_ctx,
               n_gpu_layers=args.gpu_layers, logits_all=True, verbose=False)
    template = (getattr(llm, 'metadata', {}) or {}).get('tokenizer.chat_template', '')
    if '<|turn>' not in template or '<turn|>' not in template:
        raise RuntimeError('Unexpected GGUF chat template; check model format')
    csv_path, summary_path = output_paths('gemma')
    rows = []
    for i, (idx, question, expected) in enumerate(questions, 1):
        start = time.perf_counter()
        response, predicted, method, stop, error = '', None, 'error', '', ''
        try:
            response, stop = generate(llm, question, args.max_tokens, args.n_ctx)
            predicted, method = extract_prediction(response)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        status = 'error' if error else ('invalid' if predicted is None else
                 ('correct' if predicted == expected else 'wrong'))
        rows.append(dict(question_id=idx, question=question, expected=expected,
                         predicted=predicted or '', status=status, response=response,
                         extraction_method=method, stop_reason=stop,
                         seconds=round(time.perf_counter()-start, 4), error=error))
        print(f'{i}/{len(questions)} ID={idx} {status.upper()} expected={expected} got={predicted}')
        save_results(rows, csv_path, summary_path,
                     dict(model='gemma', split=args.split, seed=args.seed,
                          max_tokens=args.max_tokens, n_ctx=args.n_ctx))
    print('Saved:', csv_path, '\nSummary:', summary_path)


if __name__ == '__main__':
    main()
