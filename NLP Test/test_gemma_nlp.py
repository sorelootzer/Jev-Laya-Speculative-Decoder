"""Gemma GGUF NLP multiple-choice benchmark; independent of Laya/JEV."""
import argparse
from pathlib import Path
from nlp_utils import DATASETS, load_examples, prompt_for_choice, run_benchmark

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / 'models' / 'Gemma4' / 'gemma-4-E4B-it-Q4_K_M.gguf'


def predict_gemma(llm, example, max_tokens, n_ctx):
    import numpy as np
    user = prompt_for_choice(example)
    prompt = f'<|turn>user\n{user}<turn|>\n<|turn>model\n'
    ids = list(llm.tokenize(prompt.encode('utf-8'), add_bos=True, special=True))
    eot = list(llm.tokenize(b'<turn|>', add_bos=False, special=True))
    if len(eot) != 1:
        raise RuntimeError('Expected Gemma 4 single-token EOT')
    stops = {int(llm.token_eos()), int(eot[0])}
    if len(ids) + max_tokens > n_ctx:
        raise ValueError(f'Prompt ({len(ids)}) plus max-tokens exceeds n_ctx={n_ctx}')
    llm.reset()
    llm.eval(ids)
    out = []
    for _ in range(max_tokens):
        token = int(np.argmax(llm.scores[llm.n_tokens-1]))
        if token in stops:
            break
        out.append(token)
        llm.eval([token])
    text = llm.detokenize(out).decode('utf-8', errors='replace')
    return text, text, None, {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, default = 'hellaswag')
    p.add_argument('--count', type=int, default=20)
    p.add_argument('--split', default='validation')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--question-id', type=int)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--n-ctx', type=int, default=2048)
    p.add_argument('--max-tokens', type=int, default=16)
    p.add_argument('--gpu-layers', type=int, default=0)
    args = p.parse_args()
    if args.count < 1 or args.max_tokens < 1:
        p.error('count and max-tokens must be positive')
    if not args.model.is_file():
        p.error(f'Model not found: {args.model}. Put scripts beside models/ or pass --model')
    examples = load_examples(args.dataset, args.split, args.count, args.seed, args.question_id)
    from llama_cpp import Llama
    llm = Llama(model_path=str(args.model), n_ctx=args.n_ctx,
               n_gpu_layers=args.gpu_layers, logits_all=True, verbose=False)
    template = (getattr(llm, 'metadata', {}) or {}).get('tokenizer.chat_template', '')
    if '<|turn>' not in template or '<turn|>' not in template:
        raise RuntimeError('Unexpected Gemma GGUF chat template')
    run_benchmark(examples, 'gemma',
                  lambda ex: predict_gemma(llm, ex, args.max_tokens, args.n_ctx),
                  dict(dataset=args.dataset, backend='gemma', split=args.split,
                       seed=args.seed, max_tokens=args.max_tokens,
                       n_ctx=args.n_ctx, model=str(args.model)))


if __name__ == '__main__':
    main()
