"""Gemma-JEV speculative decoder, configurable for paired NLP benchmarking."""
from pathlib import Path
import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
import numpy as np
from llama_cpp import Llama

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
MODEL_PATH = ROOT / "models" / "Gemma4" / "gemma-4-E4B-it-Q4_K_M.gguf"
# Unique per-run filenames are assigned when the prompt is executed.
CSV_PATH = None
BASELINE_PATH = None


def new_output_paths():
    now = datetime.now()
    day_dir = RESULTS_DIR / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    return (day_dir / f"decoder_results_{stamp}.csv",
            day_dir / f"gemma_greedy_baseline_{stamp}.txt")
N_CTX = 2048
BRANCHES = 2
DEFAULT_K = 3
FORCE_FIRST_JEV_DECISION = True
MAX_DRAFTS = 24  # JEV choice API supports at most 24 options
MAX_OUTPUT_TOKENS = 1024
MAX_ALLOWED_OUTPUT_TOKENS = 1024

# Experimental, format-aware completion policy. Does not verify correctness.
ENABLE_COMPLETION_CONTROL = True
COMPLETION_THRESHOLD = 0.80

# Optional experimental override. None disables it.
EOS_THRESHOLD = 0.90  # root-level EOS override; set None to disable
EOT_THRESHOLD = 0.50  # root-level Gemma turn-end override; set None to disable
# Only force completion when the entire greedy continuation matches a marker.
CONTROL_PROBE_THRESHOLD = 0.50
# Stop if Gemma itself strongly favors a verified end-of-turn sequence.
# Set to None to disable this policy.
CONTROL_STOP_THRESHOLD = 0.50

llm = None
EOS_ID = None
EOT_ID = None
STOP_IDS = set()
MARKERS = []


def decode(tokens):
    return llm.detokenize(list(tokens)).decode("utf-8", errors="replace")


def load_gemma(model_path=None):
    global llm, EOS_ID, EOT_ID, STOP_IDS, MARKERS
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    t = time.perf_counter()
    llm = Llama(model_path=str(path), n_ctx=N_CTX,
                logits_all=True, n_gpu_layers=0, verbose=False)
    EOS_ID = int(llm.token_eos())
    eot_ids = list(llm.tokenize(b"<turn|>", add_bos=False, special=True))
    if len(eot_ids) != 1:
        raise RuntimeError(f"Expected one Gemma turn-end token, got {eot_ids}")
    EOT_ID = int(eot_ids[0])
    STOP_IDS = {EOS_ID, EOT_ID}
    MARKERS = [("<turn|>", [EOT_ID])]
    if EOS_ID == EOT_ID:
        raise RuntimeError("EOS and EOT unexpectedly share an ID")
    template = getattr(llm, "metadata", {}).get("tokenizer.chat_template", "")
    if "<|turn>" not in template or "<turn|>" not in template:
        raise RuntimeError("GGUF does not advertise the expected Gemma 4 chat template")
    print(f"Gemma loaded in {time.perf_counter()-t:.3f}s; "
          f"EOS={EOS_ID}, EOT={EOT_ID}, stop IDs={sorted(STOP_IDS)}")


def prompt_tokens(prompt):
    # Matches the simple user/model turns in the embedded Gemma 4 template.
    s = f"<|turn>user\n{prompt.strip()}<turn|>\n<|turn>model\n"
    return list(llm.tokenize(s.encode(), add_bos=True, special=True))


def softmax(logits):
    a = np.asarray(logits, dtype=np.float64)
    z = np.exp(a - np.max(a))
    return z / z.sum()


def top_tokens(logits):
    probs = softmax(logits)
    ids = np.argpartition(probs, -BRANCHES)[-BRANCHES:]
    ids = ids[np.argsort(probs[ids])[::-1]]
    return [{"id": int(i), "probability": float(probs[i]),
             "logit": float(logits[i]), "text": decode([int(i)])} for i in ids]


def stop_reason(ids):
    if ids and ids[-1] == EOT_ID:
        return "end_of_turn"
    if ids and ids[-1] == EOS_ID:
        return "eos_token"
    return None


def marker_prefix_length(ids):
    """Longest suffix of token IDs matching the start of a control marker."""
    longest = 0
    for _, seq in MARKERS:
        for n in range(1, min(len(seq), len(ids)) + 1):
            if ids[-n:] == seq[:n]:
                longest = max(longest, n)
    return longest


def visible_token_ids(draft):
    """Return content tokens before the first EOS/EOT, excluding the marker."""
    ids = draft["token_ids"]
    for i, token in enumerate(ids):
        if token in STOP_IDS:
            return ids[:i]
    return ids


def control_draft(accepted, draft):
    """Reject protocol-only or malformed drafts, NOT completed answers.

    A draft like [token_for_22, EOT] is a valid terminal answer.
    A draft like [EOT] is control-only and must not be sent to Laya.
    """
    ids = draft["token_ids"]
    visible = visible_token_ids(draft)
    if not visible:
        return True
    # Reject any tokens after the first stop marker, rather than letting
    # a candidate silently contain material from the next turn.
    if len(visible) < len(ids) - 1:
        return True
    # Exclude dedicated channel/turn/tool protocol IDs and textual fragments.
    text = decode(visible)
    if any(x in text for x in ("<|channel>", "<|turn>", "<|tool", "<start_", "<end_")):
        return True
    return False


def probe_greedy_stop(context):
    """Verify that greedy Gemma continues directly to a known stop marker."""
    llm.reset()
    llm.eval(context)
    tokens = []
    max_steps = max((len(seq) for _, seq in MARKERS), default=0)
    for _ in range(max_steps):
        logits = llm.scores[llm.n_tokens - 1]
        token = int(np.argmax(logits))
        tokens.append(token)
        if token in STOP_IDS:
            return stop_reason(tokens)
        reason = stop_reason(tokens)
        if reason:
            return reason
        if not any(seq[:len(tokens)] == tokens for _, seq in MARKERS):
            return None
        llm.eval([token])
    return None


class PrefixKVCache:
    """Own the accepted-prefix state; never retain speculative branch state."""
    def __init__(self):
        self.tokens = []
        self.state = None
        self.live_prefix = False
        self.metrics = {"prompt_eval_seconds": 0.0, "accepted_eval_seconds": 0.0,
                        "state_save_seconds": 0.0, "state_load_seconds": 0.0,
                        "branch_eval_seconds": 0.0, "state_saves": 0,
                        "state_loads": 0, "prefix_cache_hits": 0,
                        "prefix_tokens_reused": 0, "accepted_tokens_evaluated": 0,
                        "load_elisions": 0, "parent_states_skipped": 0,
                        "branch_tokens_evaluated": 0}

    def save(self):
        t = time.perf_counter()
        state = llm.save_state()
        self.metrics["state_save_seconds"] += time.perf_counter() - t
        self.metrics["state_saves"] += 1
        return state

    def load(self, state):
        t = time.perf_counter()
        llm.load_state(state)
        self.metrics["state_load_seconds"] += time.perf_counter() - t
        self.metrics["state_loads"] += 1
        self.live_prefix = (state is self.state)

    def initialize(self, context):
        t = time.perf_counter()
        llm.reset()
        llm.eval(context)
        self.metrics["prompt_eval_seconds"] += time.perf_counter() - t
        self.tokens = list(context)
        self.state = self.save()
        self.live_prefix = True

    def restore(self, context):
        if self.state is None:
            self.initialize(context)
            return
        if list(context) != self.tokens:
            raise RuntimeError("KV accepted-prefix tokens diverged from decoder context")
        self.metrics["prefix_cache_hits"] += 1
        self.metrics["prefix_tokens_reused"] += len(context)
        if self.live_prefix:
            self.metrics["load_elisions"] += 1
        else:
            self.load(self.state)

    def advance(self, context):
        """Restore accepted state, evaluate only newly committed content tokens."""
        if self.state is None or list(context[:len(self.tokens)]) != self.tokens:
            raise RuntimeError("KV advance received a mismatched accepted prefix")
        delta = list(context[len(self.tokens):])
        if any(token in STOP_IDS for token in delta):
            # EOS/EOT is a terminal marker and is not needed in a future KV state.
            raise RuntimeError("Cannot advance KV state beyond EOS/EOT")
        if not delta:
            return
        self.load(self.state)
        t = time.perf_counter()
        llm.eval(delta)
        self.metrics["accepted_eval_seconds"] += time.perf_counter() - t
        self.metrics["accepted_tokens_evaluated"] += len(delta)
        self.tokens = list(context)
        self.state = self.save()
        self.live_prefix = True


def make_drafts(context, k, allow_preemptive_stop=True, kv_cache=None):
    """Expand a binary tree using persistent accepted-prefix and parent KV states."""
    start = time.perf_counter()
    if kv_cache is None:
        kv_cache = PrefixKVCache()
    kv_cache.restore(context)
    prefix_time = time.perf_counter() - start
    logits = llm.scores[llm.n_tokens - 1].copy()
    root_probs = softmax(logits)
    root_eos_p = float(root_probs[EOS_ID])
    root_eot_p = float(root_probs[EOT_ID])
    root_top = top_tokens(logits)[0]
    if ((allow_preemptive_stop or root_top["id"] in STOP_IDS) and
            ((EOS_THRESHOLD is not None and root_eos_p >= EOS_THRESHOLD) or
             (EOT_THRESHOLD is not None and root_eot_p >= EOT_THRESHOLD))):
        stop_name = ("end_turn_threshold" if EOT_THRESHOLD is not None and
                     root_eot_p >= EOT_THRESHOLD else "eos_threshold")
        return [], {"total_seconds": time.perf_counter()-start,
                    "prefix_seconds": prefix_time, "level_seconds": [],
                    "root_eos_probability": root_eos_p,
                    "root_eot_probability": root_eot_p,
                    "root_top": root_top, "preemptive_stop": stop_name}

    leaves = []
    level_times = [0.0] * k

    def expand(node, node_logits, depth):
        t = time.perf_counter()
        children = top_tokens(node_logits)
        level_times[depth] += time.perf_counter() - t
        if depth == k-1:
            for rank, token in enumerate(children, 1):
                child = {"name": node["name"] + str(rank),
                         "token_ids": node["token_ids"] + [token["id"]],
                         "tokens": node["tokens"] + [token],
                         "joint_log_probability": node["joint_log_probability"] +
                             math.log(max(token["probability"], 1e-45))}
                child["terminal"] = token["id"] in STOP_IDS
                child["is_terminal"] = child["terminal"]
                leaves.append(child)
            return
        # Snapshot only at genuine forks. First child uses the currently live
        # parent KV state; only later siblings need a restoration.
        nonterminal = [(rank, token) for rank, token in enumerate(children, 1)
                       if token["id"] not in STOP_IDS]
        parent_state = kv_cache.save() if len(nonterminal) > 1 else None
        if parent_state is None:
            kv_cache.metrics["parent_states_skipped"] += 1
        traversed = 0
        for rank, token in enumerate(children, 1):
            child = {"name": node["name"] + str(rank),
                     "token_ids": node["token_ids"] + [token["id"]],
                     "tokens": node["tokens"] + [token],
                     "joint_log_probability": node["joint_log_probability"] +
                         math.log(max(token["probability"], 1e-45))}
            child["terminal"] = token["id"] in STOP_IDS
            child["is_terminal"] = child["terminal"]
            if child["terminal"]:
                leaves.append(child)
                continue
            if traversed:
                kv_cache.load(parent_state)
            else:
                kv_cache.metrics["load_elisions"] += 1
            traversed += 1
            t = time.perf_counter()
            llm.eval([token["id"]])
            kv_cache.metrics["branch_eval_seconds"] += time.perf_counter()-t
            kv_cache.metrics["branch_tokens_evaluated"] += 1
            kv_cache.live_prefix = False
            expand(child, llm.scores[llm.n_tokens-1].copy(), depth+1)

    expand({"name": "", "token_ids": [], "tokens": [],
            "joint_log_probability": 0.0}, logits, 0)
    for draft in leaves:
        draft["text"] = decode(visible_token_ids(draft))
        draft["joint_probability"] = math.exp(draft["joint_log_probability"])
        draft["average_log_probability"] = draft["joint_log_probability"]/len(draft["token_ids"])
        draft["termination_reason"] = stop_reason(draft["token_ids"])
        draft["content_token_count"] = len(visible_token_ids(draft))
    return leaves, {"total_seconds": time.perf_counter()-start,
                    "prefix_seconds": prefix_time, "level_seconds": level_times,
                    "root_eos_probability": root_eos_p,
                    "root_eot_probability": root_eot_p, "root_top": root_top}


def commit(draft, context, output):
    for token in draft["token_ids"]:
        context.append(token)
        output.append(token)
        reason = stop_reason(output)
        if reason:
            return reason
    return None


def detect_output_format(prompt):
    text = prompt.lower()
    number_patterns = (
        r"\b(?:one|1|single|only|just)\s+(?:two[- ]digit\s+)?(?:number|integer|numeric(?:al)?\s+answer)\b",
        r"\b(?:two[- ]digit|2[- ]digit)\s+(?:number|answer)\b",
        r"\bonly\s+the\s+(?:number|digits?)\b",
    )
    if any(re.search(pattern, text) for pattern in number_patterns):
        return "number"
    if "reply with only the option letter" in text:
        return "choice"
    if any(phrase in text for phrase in (
        "one sentence", "single sentence", "as few words as possible",
        "brief answer", "short answer", "concise answer",
    )):
        return "short_sentence"
    return None


def response_is_complete(response, output_format):
    response = response.strip()
    if not response or output_format is None:
        return False
    if output_format == "choice":
        return re.fullmatch(r"[A-D]", response, flags=re.IGNORECASE) is not None
    if output_format == "number":
        return re.fullmatch(r"[+-]?\d+(?:\.\d+)?", response) is not None
    if output_format == "short_sentence":
        # Only use unambiguous final punctuation. This is a format heuristic,
        # not a mathematical/factual correctness check.
        return bool(re.search(r"[.!?]$", response))
    return False


def completion_candidate(drafts, output, output_format):
    """Only override with a HIGH-PROBABILITY TERMINAL complete answer."""
    if not ENABLE_COMPLETION_CONTROL or output_format is None or not drafts:
        return None
    best = max(drafts, key=lambda d: d["joint_probability"])
    if (best["joint_probability"] >= COMPLETION_THRESHOLD
            and best.get("terminal", False)
            and response_is_complete(decode(output) + best["text"], output_format)):
        return best
    return None


def run(prompt, k, selector, csv_path=None, max_tokens=None, verbose=True,
        force_first_jev=FORCE_FIRST_JEV_DECISION):
    max_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
    if max_tokens < k or max_tokens > MAX_ALLOWED_OUTPUT_TOKENS:
        raise ValueError("max_tokens must be >= k and <= 1024")
    context = prompt_tokens(prompt)
    kv_cache = PrefixKVCache()
    output = []
    rows = []
    selection_results = []
    candidate_evaluations = {}
    output_format = detect_output_format(prompt)
    started = time.perf_counter()
    gemma_time = laya_time = 0.0
    candidate_evaluations = {}
    overrides = 0
    jev_overrides = 0
    bypass_reasons = []
    reason = None
    iteration = 0
    if ENABLE_COMPLETION_CONTROL and response_is_complete(decode(output), output_format):
        reason = "format_complete"
    while reason is None and len(context)+k <= N_CTX and len(output)+k <= max_tokens:
        iteration += 1
        drafts, metrics = make_drafts(
            context, k, allow_preemptive_stop=not (force_first_jev and not selection_results),
            kv_cache=kv_cache)
        gemma_time += metrics["total_seconds"]
        if metrics.get("preemptive_stop"):
            reason = metrics["preemptive_stop"]
            overrides += 1
            bypass_reasons.append(reason)
            print(f"Iteration {iteration}: {reason}; "
                  f"EOS P={metrics['root_eos_probability']:.4f}, "
                  f"EOT P={metrics['root_eot_probability']:.4f}; stopping before JEV.")
            rows.append({
                "iteration": iteration, "k": k, "draft_name": "", "draft_text": "",
                "gemma_probability": None, "terminal": True,
                "terminal_reason": metrics["preemptive_stop"],
                "content_token_count": 0,
                "selected_draft": "", "jev_probability": None,
                "selection_mode": reason, "root_eos_probability": metrics["root_eos_probability"],
                "root_eot_probability": metrics["root_eot_probability"],
                "gemma_seconds": metrics["total_seconds"],
                "gemma_level_seconds": "[]", "jev_seconds": 0.0,
                "iteration_termination_reason": reason,
                "output_format": output_format or "unrestricted",
                "completion_override": False,
                "completion_candidate": "",
                "completion_probability": None,
                "selector_backend": selector.backend_name,
                "selector_model_version": selector.model_name,
            })
            break
        if not drafts:
            reason = "no_valid_drafts"
            break
        selected = None
        candidate_evaluations = {}
        laya = {"decision_seconds": 0, "probabilities": {}}
        mode = "jev"
        eos_p = metrics["root_eos_probability"]
        eot_p = metrics["root_eot_probability"]
        top = metrics["root_top"]
        semantic = [d for d in drafts if not control_draft(output, d)]
        # In benchmark mode, the first eligible tree must reach JEV before
        # optional Gemma control/completion overrides. Real EOS/EOT remains valid.
        first_eligible = bool(force_first_jev and not selection_results and len(semantic) >= 2)
        if not semantic:
            bypass_reasons.append("no_semantic_candidates")
        elif len(semantic) < 2:
            bypass_reasons.append("fewer_than_two_semantic_candidates")
        # A verified high-probability turn closure takes precedence only
        # after the first eligible JEV evaluation.

        if (not first_eligible and CONTROL_STOP_THRESHOLD is not None
                and top["probability"] >= CONTROL_STOP_THRESHOLD
                and (top["id"] in STOP_IDS or any(
                    seq and top["id"] == seq[0] for _, seq in MARKERS
                ))):
            probed = probe_greedy_stop(context)
            kv_cache.live_prefix = False
            if probed:
                reason = "verified_greedy_" + probed
                mode = "gemma_control_stop"
                overrides += 1
        # After the first eligible JEV selection, normal completion controls apply.
        completion = (completion_candidate(drafts, output, output_format)
                      if reason is None and not first_eligible else None)
        if completion is not None:
            selected = completion
            mode = "completion_override"
            overrides += 1
            reason = commit(selected, context, output) or "completion_override"
        elif reason is None:
            if len(semantic) >= 2:
                gemma_best = max(semantic, key=lambda d: d["joint_probability"])
                selection = selector.select(
                    user_prompt=prompt, assistant_response=decode(output), drafts=semantic
                )
                selected = selection["selected_draft"]
                laya = selection["metadata"]
                candidate_evaluations = selection["candidate_evaluations"]
                raw_laya_result = selection["raw_result"]
                selection_results.append(selection)
                if selected["name"] != gemma_best["name"]:
                    jev_overrides += 1
                mode = "jev_filtered"  # Shared label for existing CSV readers
                laya_time += laya.get("decision_seconds", 0.0)
                reason = commit(selected, context, output)
                if (reason is None and ENABLE_COMPLETION_CONTROL
                        and response_is_complete(
                            decode([t for t in output if t not in STOP_IDS]),
                            output_format)):
                    reason = "format_complete"
            else:
                # Do not append '<' or malformed protocol fragments merely
                # because all semantic candidates have been filtered out.
                reason = "insufficient_semantic_drafts"
                mode = "control_blocked"
                overrides += 1
                bypass_reasons.append(reason)
        print(f"Iteration {iteration}: {mode}, EOS P={eos_p:.4f}, EOT P={eot_p:.4f}, "
              f"selected={selected['name'] if selected else 'STOP'}, "
              f"text={selected['text']!r}" if selected else
              f"Iteration {iteration}: {mode}, EOS P={eos_p:.4f}, "
              f"EOT P={eot_p:.4f}, STOP")
        print("Candidate drafts (Gemma / JEV):")
        for d in sorted(drafts, key=lambda x: x["joint_probability"], reverse=True):
            evaluation = (candidate_evaluations.get(d["name"], {})
                          if mode == "jev_filtered" else {})
            lp = evaluation.get("jev_probability")
            label = "JEV" if evaluation else "FILTERED/CONTROL"
            chosen = " [SELECTED]" if selected and selected["name"] == d["name"] else ""
            lp_text = f" Selector P={float(lp):.6g}" if lp is not None else ""
            print(f"  {d['name']} {d['text']!r} Gemma P={d['joint_probability']:.6g}"
                  f"{lp_text} [{label}]{'[EOT]' if d.get('terminal') else ''}{chosen}")
        for draft in drafts:
            rows.append({
                "iteration": iteration, "k": k,
                "draft_name": draft["name"], "draft_text": draft["text"],
                "gemma_probability": draft["joint_probability"],
                "terminal": draft["terminal"],
                "terminal_reason": draft.get("termination_reason"),
                "content_token_count": draft.get("content_token_count"),
                "selected_draft": selected["name"] if selected else "",
                "jev_probability": (
                    candidate_evaluations.get(draft["name"], {}).get("jev_probability")
                    if mode == "jev_filtered" else None
                ),
                "selection_mode": mode, "root_eos_probability": eos_p,
                "root_eot_probability": eot_p,
                "gemma_seconds": metrics["total_seconds"],
                "gemma_level_seconds": json.dumps(metrics["level_seconds"]),
                "jev_seconds": laya.get("decision_seconds", 0.0),
                "iteration_termination_reason": reason or "",
                "output_format": output_format or "unrestricted",
                "completion_override": mode == "completion_override",
                "completion_candidate": selected["name"] if mode == "completion_override" else "",
                "completion_probability": selected["joint_probability"] if mode == "completion_override" else None,
                "selector_backend": selector.backend_name,
                "jev_called": mode == "jev_filtered",
                "jev_override": mode == "jev_filtered" and selected is not None and selected["name"] != gemma_best["name"],
                "jev_bypass_reason": "" if mode == "jev_filtered" else mode,
                "selector_model_version": laya.get("model_version", selector.model_name),
            })
        if reason:
            break
        kv_cache.advance(context)
    if reason is None:
        reason = "max_tokens" if len(output)+k > max_tokens else "context_limit"
    response = decode([token for token in output if token not in STOP_IDS])
    for marker, _ in MARKERS:
        response = response.split(marker, 1)[0]
    elapsed = time.perf_counter()-started
    for row in rows:
        row.update(final_response=response, termination_reason=reason,
                   control_overrides=overrides, elapsed_seconds=elapsed)
    for row in rows:
        row.update(kv_metrics=json.dumps(kv_cache.metrics))
    if rows and csv_path is not None:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
    if verbose:
        print("\nFINAL RESPONSE:\n" + response)
    if verbose:
        print(f"Termination={reason}; iterations={iteration}; output tokens={len(output)}")
        print(f"Gemma={gemma_time:.3f}s; Selector={laya_time:.3f}s; "
          f"total={elapsed:.3f}s; throughput={len(output)/elapsed:.3f} tokens/s")
    if verbose:
        print(f"KV cache: {kv_cache.metrics}")
        print(f"Control overrides={overrides}; CSV={csv_path}")
    return {"response": response, "rows": rows,
            "jev_selections": selection_results, "termination_reason": reason,
            "output_tokens": len(output), "elapsed_seconds": elapsed,
            "jev_seconds": laya_time, "gemma_seconds": gemma_time,
            "jev_calls": len(selection_results), "jev_overrides": jev_overrides,
            "jev_bypassed": len(selection_results) == 0,
            "jev_bypass_reasons": bypass_reasons or ([reason] if not selection_results else []),
            "force_first_jev": force_first_jev, "kv_metrics": dict(kv_cache.metrics)}



def greedy_baseline(prompt, baseline_path=None, max_tokens=None):
    """Greedy Gemma with a persistent llama.cpp KV cache for each prompt."""
    max_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    context = prompt_tokens(prompt)
    generated = []
    start = time.perf_counter()
    reason = "context_limit"
    if len(context) < N_CTX:
        llm.reset()
        llm.eval(context)  # Evaluate prompt only once.
        while len(generated) < max_tokens:
            token = int(np.argmax(llm.scores[llm.n_tokens - 1]))
            if token in STOP_IDS:
                reason = stop_reason([token]) or "stop_token"
                break
            generated.append(token)
            if len(context) + len(generated) >= N_CTX:
                reason = "context_limit"
                break
            # Append only the new token; do not reset/replay the prompt.
            llm.eval([token])
        else:
            reason = "max_tokens"
    result = decode(generated)
    duration = time.perf_counter() - start
    if baseline_path is not None:
        baseline_path.write_text(
            f"Prompt: {prompt}\nAnswer: {result}\nStop reason: {reason}\n"
            f"Tokens: {len(generated)}\nSeconds: {duration:.3f}\n",
            encoding="utf-8")
    print(f"Gemma-only baseline: {result!r} ({reason}, {duration:.2f}s)")
    return {"response": result, "termination_reason": reason,
            "output_tokens": len(generated), "elapsed_seconds": duration}


def main():
    parser = argparse.ArgumentParser(description="Gemma-JEV speculative decoder")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Gemma GGUF path")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--no-force-first-jev", action="store_true",
                        help="Disable first eligible tree JEV evaluation")
    parser.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()
    if args.k < 1 or BRANCHES ** args.k > MAX_DRAFTS:
        parser.error(f"k must satisfy 2^k <= {MAX_DRAFTS}")
    load_gemma(args.model)
    from jev_selector import JEVSelector
    selector = JEVSelector(k=args.k, branching_factor=BRANCHES, model_name=args.jev_model)
    selector.backend_name = "jev"
    while True:
        try:
            prompt = input("\nEnter prompt (or exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.lower() in ("exit", "quit"):
            break
        if not prompt:
            continue
        csv_path, baseline_path = new_output_paths()
        greedy_baseline(prompt, baseline_path, max_tokens=args.max_tokens)
        run(prompt, args.k, selector, csv_path, max_tokens=args.max_tokens,
            force_first_jev=not args.no_force_first_jev)


if __name__ == "__main__":
    main()
