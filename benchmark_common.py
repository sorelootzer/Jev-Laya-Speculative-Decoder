"""Shared CLI, GGUF adapter, metrics and decoding algorithms. Python 3.10+.
Greedy only: no repetition penalties, sampling thresholds, or random sampling.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import threading
import time
import uuid
from datetime import datetime, timezone


def decode(target, prompt_ids, stops, max_tokens, draft=None, k=2, timeout=300):
    """Adapter interface: reset(), sync(ids), next_token(), block(proposals).
    block returns predictions *after* each proposed token, in one batched eval.
    Returns partial output and error information rather than dropping failures.
    """
    output, events = [], []
    start = time.perf_counter()
    ttft = None
    draft_s = target_s = 0.0
    proposed_total = accepted_total = 0
    reason, error = 'max_tokens', ''
    try:
        target.reset()
        if draft is not None:
            draft.reset()
        while len(output) < max_tokens:
            if time.perf_counter() - start >= timeout:
                reason = 'timeout'
                break
            context = prompt_ids + output
            tick = time.perf_counter()
            target.sync(context)
            first = target.next_token()
            target_s += time.perf_counter() - tick
            if draft is None:
                committed = [first]
            else:
                tick = time.perf_counter()
                draft.sync(context)
                proposals = []
                for _ in range(min(k, max_tokens - len(output))):
                    token = draft.next_token()
                    proposals.append(token)
                    if token in stops:
                        break
                    # Last proposal need not be evaluated by the draft yet.
                    if len(proposals) < min(k, max_tokens - len(output)):
                        draft.sync(context + proposals)
                ds = time.perf_counter() - tick
                draft_s += ds
                tick = time.perf_counter()
                after = target.block(proposals)
                predictions = [first] + after
                committed, accepted = [], 0
                mismatch = False
                for i, token in enumerate(proposals):
                    if token != predictions[i]:
                        committed.append(predictions[i])
                        mismatch = True
                        break
                    accepted += 1
                    committed.append(token)
                    if token in stops:
                        break
                if not mismatch and committed[-1] not in stops:
                    # Standard greedy speculative bonus token.
                    committed.append(predictions[len(proposals)])
                vs = time.perf_counter() - tick
                target_s += vs
                proposed_total += len(proposals)
                accepted_total += accepted
                events.append(dict(iteration=len(events)+1,
                                   context_tokens=len(context), proposed=len(proposals),
                                   accepted=accepted, rejected_suffix=len(proposals)-accepted,
                                   draft_seconds=ds, verification_seconds=vs,
                                   proposal_ids=proposals, committed_ids=committed[:max_tokens-len(output)]))
            for token in committed:
                if token in stops:
                    reason = 'eog'
                    break
                output.append(token)
                if ttft is None:
                    ttft = time.perf_counter() - start
                if len(output) >= max_tokens:
                    break
            if reason == 'eog':
                break
    except KeyboardInterrupt:
        reason, error = 'interrupted', 'KeyboardInterrupt'
    except Exception as exc:
        reason, error = 'error', f'{type(exc).__name__}: {exc}'
    elapsed = time.perf_counter() - start
    return dict(output_ids=output, stop_reason=reason, error=error,
                response_seconds=elapsed, ttft_seconds=ttft,
                output_tokens=len(output), tokens_per_second=len(output)/elapsed if elapsed else 0,
                draft_seconds=draft_s, target_seconds=target_s,
                proposed_tokens=proposed_total, accepted_tokens=accepted_total,
                acceptance_rate=accepted_total/proposed_total if proposed_total else None,
                events=events)


class GGUF:
    def __init__(self, path, args, all_logits=False):
        from llama_cpp import Llama
        self.llm = Llama(model_path=str(Path(path).resolve()), n_ctx=args.n_ctx,
                         n_batch=min(args.n_batch, args.n_ctx),
                         n_threads=args.threads, n_threads_batch=args.threads,
                         n_gpu_layers=args.gpu_layers, logits_all=all_logits,
                         seed=42, use_mmap=True, verbose=args.verbose)
        self.prefix = []
        self.fallbacks = 0
        self.all_logits = all_logits

    def reset(self):
        self.llm.reset()
        self.prefix = []
        self.fallbacks = 0

    def sync(self, ids):
        """Reuse valid KV prefix; fall back to full prefill if rollback unsupported.
        Private backend adapter is deliberately isolated here. No copied KV states.
        """
        n = 0
        for a, b in zip(self.prefix, ids):
            if a != b:
                break
            n += 1
        if n < len(self.prefix):
            # Re-evaluate final prefix token when otherwise no suffix remains.
            n = min(n, max(0, len(ids)-1))
            recurrent = (getattr(self.llm, '_is_recurrent', False) or
                         getattr(self.llm, '_is_hybrid', False))
            ok = False if recurrent else self.llm._ctx.kv_cache_seq_rm(-1, n, -1)
            if ok:
                self.llm.n_tokens = n
            else:
                self.llm.reset()
                n = 0
                self.fallbacks += 1
            self.prefix = self.prefix[:n]
        if n < len(ids):
            self.llm.eval(ids[n:])
            self.prefix = list(ids)

    def next_token(self):
        # Llama.sample reads current backend logits even with logits_all=False.
        return int(self.llm.sample(temp=0.0, repeat_penalty=1.0,
                                   frequency_penalty=0.0, presence_penalty=0.0))

    def block(self, ids):
        import numpy as np
        if not self.all_logits:
            raise RuntimeError('Target block verification requires logits_all=True')
        n = len(self.prefix)
        self.llm.eval(ids)
        self.prefix.extend(ids)
        rows = self.llm.scores[n:n+len(ids)]
        if rows.shape[0] != len(ids) or not np.isfinite(rows).all():
            raise RuntimeError('Invalid verification logits; check llama-cpp-python compatibility')
        return [int(x) for x in np.argmax(rows, axis=1)]

    def render(self, text, template_file=None):
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        template = (Path(template_file).read_text(encoding='utf-8') if template_file
                    else self.llm.metadata.get('tokenizer.chat_template'))
        if not template:
            raise ValueError('No GGUF chat template. Supply a verified --template-file .jinja')
        def token_text(t):
            return self.llm._model.token_get_text(t) if t >= 0 else ''
        formatter = Jinja2ChatFormatter(template=template,
                    eos_token=token_text(self.llm.token_eos()),
                    bos_token=token_text(self.llm.token_bos()), add_generation_prompt=True)
        formatted = formatter(messages=[{'role':'user', 'content':text}]).prompt
        # Chat template already owns BOS. Do not add a second one.
        ids = self.llm.tokenize(formatted.encode('utf-8'), add_bos=False, special=True)
        if not ids:
            raise ValueError('Chat template produced an empty prompt')
        return formatted, ids, hashlib.sha256(template.encode()).hexdigest()

    def stops(self, extra):
        import llama_cpp
        model = self.llm._model.model
        if hasattr(llama_cpp, 'llama_model_get_vocab') and hasattr(llama_cpp, 'llama_vocab_is_eog'):
            vocab = llama_cpp.llama_model_get_vocab(model)
            result = {i for i in range(self.llm.n_vocab()) if llama_cpp.llama_vocab_is_eog(vocab, i)}
        elif hasattr(llama_cpp, 'llama_token_is_eog'):
            result = {i for i in range(self.llm.n_vocab()) if llama_cpp.llama_token_is_eog(model, i)}
        else:
            raise RuntimeError('Backend lacks EOG detection. Install a compatible llama-cpp-python build.')
        result.add(self.llm.token_eos())
        for word in extra:
            ids = self.llm.tokenize(word.encode(), add_bos=False, special=True)
            if len(ids) != 1:
                raise ValueError(f'Extra stop token {word!r} is not a single special token')
            result.add(ids[0])
        return result

    def text(self, ids):
        return self.llm.detokenize(ids).decode('utf-8', errors='replace')


def compatible(small, target, extra_stops=()):
    """Require shared token IDs and stopping policy, not identical default EOS.

    Chat templates may differ: the target renders the shared prompt. Vocabulary
    strings, remaining tokenizer metadata, probes and actual prompts are checked.
    This is a conservative operational check, not a proof for every input string.
    """
    a, b = small.llm, target.llm
    report = dict(small_bos=a.token_bos(), target_bos=b.token_bos(),
                  small_eos=a.token_eos(), target_eos=b.token_eos(),
                  vocabulary_size=a.n_vocab())
    if a.n_vocab() != b.n_vocab():
        raise ValueError('Small and large vocabulary sizes differ; use a compatible model pair.')
    if a.token_bos() != b.token_bos():
        raise ValueError(f'Small and large BOS IDs differ: {report}')
    for i in range(a.n_vocab()):
        if a._model.token_get_text(i) != b._model.token_get_text(i):
            raise ValueError(f'Token vocabulary mismatch at ID {i}.')
    for key in set(a.metadata) | set(b.metadata):
        # Only this default may differ; the effective stop sets below must match.
        if key == 'tokenizer.ggml.eos_token_id':
            continue
        if key.startswith('tokenizer.ggml.') and a.metadata.get(key) != b.metadata.get(key):
            raise ValueError(f'Tokenizer metadata mismatch: {key}')
    small_stops, target_stops = small.stops(extra_stops), target.stops(extra_stops)
    if small_stops != target_stops:
        raise ValueError('Small and large stop token sets differ: '
                         f'{sorted(small_stops)} vs {sorted(target_stops)}')
    for name, model in [('small', a), ('target', b)]:
        if not 0 <= model.token_eos() < model.n_vocab():
            raise ValueError(f'{name} has no valid default EOS ID')
    for text in [' Paris', 'Hello\nworld!', '你好 日本語', '12.50 + 3 =', ' a  b\tç',
                 '<bos><|turn>user\nTest<turn|>\n<|turn>model\n']:
        for special in (False, True):
            if a.tokenize(text.encode(), add_bos=False, special=special) != b.tokenize(
                    text.encode(), add_bos=False, special=special):
                raise ValueError(f'Tokenizer probe mismatch (special={special}).')
    report.update(stop_token_ids=sorted(target_stops),
                  differing_default_eos_allowed=a.token_eos() != b.token_eos(),
                  vocabulary_checked=True)
    return report


def shared_prompt(target, small, text, template_file=None):
    """Use exactly the large-only target prompt, then verify draft tokenization."""
    rendered, tokens, template_hash = target.render(text, template_file)
    if small is not None:
        draft_tokens = small.llm.tokenize(rendered.encode('utf-8'), add_bos=False, special=True)
        if draft_tokens != tokens:
            raise ValueError('Draft tokenization of target-formatted prompt differs.')
    return rendered, tokens, template_hash


class MemoryMonitor:
    """Sample process-tree RSS and system-used RAM every 50 ms; no VRAM metrics.
    RSS summed across processes can double-count shared pages. Peaks are sampled.
    """
    def __enter__(self):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.peak_rss = self.peak_system = 0
        self.done = threading.Event()
        self.sample()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        try:
            processes = [self.process] + self.process.children(recursive=True)
            rss = 0
            for p in processes:
                try:
                    rss += p.memory_info().rss
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            mem = self.psutil.virtual_memory()
            self.peak_system = max(self.peak_system, mem.total - mem.available)
        except self.psutil.Error:
            pass

    def loop(self):
        while not self.done.wait(0.05):
            self.sample()

    def __exit__(self, *args):
        self.done.set()
        self.thread.join()
        self.sample()


def main(mode):
    parser = argparse.ArgumentParser(description=f'{mode}: greedy GGUF benchmark, CPU by default')
    if mode == 'small_large':
        parser.add_argument('--small-model', required=True)
        parser.add_argument('--large-model', required=True)
    else:
        parser.add_argument('--model', required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--prompt')
    source.add_argument('--prompts', help='JSON list of {id, prompt, reference, category}')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=128)
    parser.add_argument('--n-ctx', type=int, default=2048)
    parser.add_argument('--n-batch', type=int, default=256)
    parser.add_argument('--threads', type=int, default=max(1, (os.cpu_count() or 2)//2))
    parser.add_argument('--gpu-layers', type=int, default=0)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--timeout', type=float, default=300, help='Soft limit checked between iterations')
    parser.add_argument('--template-file', help='Verified Jinja chat template override (all methods)')
    parser.add_argument('--stop-token', action='append', default=[])
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    for key in ['repeats','max_tokens','n_ctx','n_batch','threads','k','timeout']:
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_","-")} must be positive')
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    prompts = (json.loads(Path(args.prompts).read_text(encoding='utf-8-sig')) if args.prompts else
               [dict(id='manual', prompt=args.prompt if args.prompt is not None else input('Prompt: '))])
    if not isinstance(prompts, list) or not prompts:
        parser.error('Prompt JSON must be a nonempty list')
    ids = set()
    for item in prompts:
        if not isinstance(item, dict) or not isinstance(item.get('prompt'), str) or not item['prompt'].strip():
            parser.error('Each prompt needs a nonempty prompt string and unique id')
        if 'id' not in item or str(item['id']) in ids:
            parser.error('Missing or duplicate prompt id')
        ids.add(str(item['id']))
    out = Path(args.out_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+mode+'_'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True)
    def dump(name, obj):
        (out/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = dict(mode=mode, settings=vars(args), platform=platform.platform(), cpu=platform.processor(),
                    python=platform.python_version(), algorithm='greedy; no penalties',
                    metrics_note='TTFT is time to first committed non-EOG token; no streaming/printing in timer')
    dump('manifest.json', manifest)
    small = target = None
    try:
        import psutil
        import llama_cpp
        manifest['llama_cpp_version'] = importlib.metadata.version('llama-cpp-python')
        manifest['physical_ram_gib'] = psutil.virtual_memory().total / 2**30
        paths = [args.small_model, args.large_model] if mode == 'small_large' else [args.model]
        manifest['models'] = [dict(path=str(Path(p).resolve()), bytes=Path(p).stat().st_size,
                                    modified_ns=Path(p).stat().st_mtime_ns) for p in paths]
        tick = time.perf_counter()
        with MemoryMonitor() as loading_memory:
            if mode == 'small_large':
                small = GGUF(args.small_model, args)
                target = GGUF(args.large_model, args, all_logits=True)
            else:
                target = GGUF(args.model, args)
        load_seconds = time.perf_counter()-tick
        manifest.update(model_load_seconds=load_seconds, load_peak_rss_gib=loading_memory.peak_rss/2**30,
                        target_metadata=target.llm.metadata,
                        small_metadata=small.llm.metadata if small else None)
        stops = target.stops(args.stop_token)
        if small:
            print('Checking small/large tokenizer compatibility...')
            manifest['compatibility'] = compatible(small, target, args.stop_token)
            print('Tokenizer compatibility passed:', manifest['compatibility'])
            manifest['prompt_policy'] = 'target chat template shared by target and draft'
        prepared = []
        for item in prompts:
            rendered, tokens, template_hash = shared_prompt(
                target, small, item['prompt'], args.template_file)
            if len(tokens)+args.max_tokens > args.n_ctx:
                raise ValueError(f'Prompt {item["id"]}: input + max output exceeds --n-ctx; shorten or increase it.')
            prepared.append((item, tokens))
            item = dict(item, formatted_prompt=rendered, input_ids=tokens, template_sha256=template_hash)
            manifest.setdefault('prompts', []).append(item)
        manifest['stop_token_ids'] = sorted(stops)
        manifest['target_logits_all'] = bool(small)
        dump('manifest.json', manifest)
        print('Warming up (not included in CSV)...')
        for _ in range(args.warmup):
            warm = decode(target, prepared[0][1], stops, min(16, args.max_tokens), small, args.k, args.timeout)
            if warm['stop_reason'] in ('error','interrupted','timeout'):
                raise RuntimeError('Warm-up failed: '+str(warm))
        columns = ['run_id','mode','prompt_id','repeat','category','prompt','reference','k',
                   'input_tokens','output_tokens','response_seconds','ttft_seconds','tokens_per_second',
                   'draft_seconds','target_seconds','proposed_tokens','accepted_tokens','acceptance_rate',
                   'peak_rss_gib','peak_system_used_gib','model_load_seconds','stop_reason','error',
                   'target_cache_fallbacks','draft_cache_fallbacks','answer','quality_score']
        columns += ['target_prefill_seconds', 'target_sync_seconds', 'target_prediction_seconds', 'draft_prefill_seconds', 'draft_sync_seconds', 'draft_prediction_seconds', 'verification_eval_seconds', 'verification_decision_seconds', 'verification_calls', 'rejected_tokens', 'rejection_rate']
        with (out/'summary.csv').open('w',newline='',encoding='utf-8-sig') as f, \
             (out/'details.jsonl').open('w',encoding='utf-8') as detail:
            writer = csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for repeat in range(1,args.repeats+1):
                for item,tokens in prepared:
                    with MemoryMonitor() as memory:
                        result = decode(target,tokens,stops,args.max_tokens,small,args.k,args.timeout)
                    row = {key:result[key] for key in columns if key in result}
                    answer = target.text(result['output_ids'])
                    row.update(run_id=uuid.uuid4().hex, mode=mode, prompt_id=item['id'], repeat=repeat,
                               category=item.get('category',''), prompt=item['prompt'], reference=item.get('reference',''),
                               k=args.k if small else '', input_tokens=len(tokens),
                               peak_rss_gib=memory.peak_rss/2**30, peak_system_used_gib=memory.peak_system/2**30,
                               model_load_seconds=load_seconds,target_cache_fallbacks=target.fallbacks,
                               draft_cache_fallbacks=small.fallbacks if small else 0, answer=answer, quality_score='')
                    writer.writerow(row)
                    f.flush()
                    detail.write(json.dumps(dict(run_id=row['run_id'], **result),ensure_ascii=False)+'\n')
                    detail.flush()
                    print(f'{item["id"]} repeat {repeat}: {result["response_seconds"]:.2f}s, '
                          f'{result["tokens_per_second"]:.2f} tok/s, {result["stop_reason"]}')
                    if len(prepared)==1:
                        print(answer)
                    if result['stop_reason'] in ('error','interrupted'):
                        raise RuntimeError('Run stopped; partial result saved: '+result['error'])
        print('Results:',out.resolve())
    except Exception as exc:
        dump('manifest.json', manifest)
        dump('failure.json', dict(error=f'{type(exc).__name__}: {exc}'))
        raise SystemExit(f'Failed: {exc}\nDiagnostics: {out.resolve()}')
    finally:
        for adapter in [small,target]:
            if adapter is not None:
                adapter.llm.close()

# ESP_METRICS_V3: timing only; use external GPU monitor.
_old_reset, _old_sync = GGUF.reset, GGUF.sync
_old_next, _old_block = GGUF.next_token, GGUF.block

def _reset(self):
    _old_reset(self)
    self._times = dict(prefill=0., sync=0., prediction=0., block=0.)
    self._sync_count = 0

def _sync(self, ids):
    t = time.perf_counter()
    try:
        return _old_sync(self, ids)
    finally:
        key = 'prefill' if self._sync_count == 0 else 'sync'
        self._times[key] += time.perf_counter() - t
        self._sync_count += 1

def _next(self):
    t = time.perf_counter()
    try:
        return _old_next(self)
    finally:
        self._times['prediction'] += time.perf_counter() - t

def _block(self, ids):
    t = time.perf_counter()
    try:
        return _old_block(self, ids)
    finally:
        self._times['block'] += time.perf_counter() - t

GGUF.reset, GGUF.sync = _reset, _sync
GGUF.next_token, GGUF.block = _next, _block
_old_decode = decode

def decode(target, prompt_ids, stops, max_tokens,
           draft=None, k=2, timeout=300):
    r = _old_decode(
        target, prompt_ids, stops, max_tokens, draft, k, timeout
    )
    zero = dict(prefill=0., sync=0., prediction=0., block=0.)
    a = getattr(target, '_times', zero)
    b = getattr(draft, '_times', zero)

    r.update(
        target_prefill_seconds=a['prefill'],
        target_sync_seconds=a['sync'],
        target_prediction_seconds=a['prediction'],
        draft_prefill_seconds=b['prefill'],
        draft_sync_seconds=b['sync'],
        draft_prediction_seconds=b['prediction'],
        verification_eval_seconds=a['block'],
        verification_decision_seconds=max(
            0.,
            sum(e['verification_seconds'] for e in r['events'])
            - a['block']
        ),
        verification_calls=len(r['events']),
        rejected_tokens=r['proposed_tokens']-r['accepted_tokens'],
        rejection_rate=(
            1-r['acceptance_rate']
            if r['acceptance_rate'] is not None else None
        ),
    )
    return r
