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
DEFAULT_K = 2
MAX_DRAFTS = 24  # JEV choice API supports at most 24 options
MAX_OUTPUT_TOKENS = 64

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


def make_drafts(context, k):
    start = time.perf_counter()
    llm.reset()
    llm.eval(context)
    logits = llm.scores[llm.n_tokens - 1].copy()
    root_probs = softmax(logits)
    root_eos_p = float(root_probs[EOS_ID])
    root_eot_p = float(root_probs[EOT_ID])
    root_top = top_tokens(logits)[0]
    # Check the root EOS/EOT distribution BEFORE expanding any speculative tree.
    # This is an experimental override, distinct from normal EOS detection.
    if ((EOS_THRESHOLD is not None and root_eos_p >= EOS_THRESHOLD) or
            (EOT_THRESHOLD is not None and root_eot_p >= EOT_THRESHOLD)):
        stop_name = ("end_turn_threshold" if EOT_THRESHOLD is not None and
                     root_eot_p >= EOT_THRESHOLD else "eos_threshold")
        return [], {
            "total_seconds": time.perf_counter() - start,
            "prefix_seconds": time.perf_counter() - start,
            "level_seconds": [],
            "root_eos_probability": root_eos_p,
            "root_eot_probability": root_eot_p,
            "root_top": root_top,
            "preemptive_stop": stop_name,
        }
    state = llm.save_state()
    prefix_time = time.perf_counter() - start
    frontier = [{"name": "", "token_ids": [], "tokens": [],
                 "joint_log_probability": 0.0}]
    leaves = []
    level_times = []
    for depth in range(k):
        t = time.perf_counter()
        next_frontier = []
        for node in frontier:
            llm.load_state(state)
            if node["token_ids"]:
                llm.eval(node["token_ids"])
            for rank, token in enumerate(top_tokens(llm.scores[llm.n_tokens-1].copy()), 1):
                ids = node["token_ids"] + [token["id"]]
                child = {
                    "name": node["name"] + str(rank),
                    "token_ids": ids,
                    "tokens": node["tokens"] + [token],
                    "joint_log_probability": node["joint_log_probability"] +
                        math.log(max(token["probability"], 1e-45)),
                }
                child["terminal"] = bool(stop_reason(context + ids))
                child["is_terminal"] = child["terminal"]
                if child["terminal"] or depth == k-1:
                    leaves.append(child)
                else:
                    next_frontier.append(child)
        level_times.append(time.perf_counter()-t)
        frontier = next_frontier
        if not frontier:
            break
    for draft in leaves:
        draft["text"] = decode(visible_token_ids(draft))
        draft["joint_probability"] = math.exp(draft["joint_log_probability"])
        draft["average_log_probability"] = draft["joint_log_probability"]/len(draft["token_ids"])
        draft["termination_reason"] = stop_reason(draft["token_ids"])
        draft["content_token_count"] = len(visible_token_ids(draft))
    return leaves, {
        "total_seconds": time.perf_counter()-start,
        "prefix_seconds": prefix_time, "level_seconds": level_times,
        "root_eos_probability": root_eos_p,
        "root_eot_probability": root_eot_p, "root_top": root_top
    }


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


def run(prompt, k, selector, csv_path=None, max_tokens=None, verbose=True):
    max_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
    if max_tokens < k:
        raise ValueError("max_tokens must be >= k")
    context = prompt_tokens(prompt)
    output = []
    rows = []
    selection_results = []
    candidate_evaluations = {}
    output_format = detect_output_format(prompt)
    started = time.perf_counter()
    gemma_time = laya_time = 0.0
    candidate_evaluations = {}
    overrides = 0
    reason = None
    iteration = 0
    if ENABLE_COMPLETION_CONTROL and response_is_complete(decode(output), output_format):
        reason = "format_complete"
    while reason is None and len(context)+k <= N_CTX and len(output)+k <= max_tokens:
        iteration += 1
        drafts, metrics = make_drafts(context, k)
        gemma_time += metrics["total_seconds"]
        if metrics.get("preemptive_stop"):
            reason = metrics["preemptive_stop"]
            overrides += 1
            print(f"Iteration {iteration}: {reason}; "
                  f"EOS P={metrics['root_eos_probability']:.4f}, "
                  f"EOT P={metrics['root_eot_probability']:.4f}; stopping before Laya.")
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
        # A verified high-probability turn closure takes precedence over Laya.
        if (CONTROL_STOP_THRESHOLD is not None
                and top["probability"] >= CONTROL_STOP_THRESHOLD
                and (top["id"] in STOP_IDS or any(
                    seq and top["id"] == seq[0] for _, seq in MARKERS
                ))):
            probed = probe_greedy_stop(context)
            if probed:
                reason = "verified_greedy_" + probed
                mode = "gemma_control_stop"
                overrides += 1
        # A complete terminal answer from Gemma may close a concise response
        # without giving Laya another opportunity to append an explanation.
        completion = completion_candidate(drafts, output, output_format) if reason is None else None
        if completion is not None:
            selected = completion
            mode = "completion_override"
            overrides += 1
            reason = commit(selected, context, output) or "completion_override"
        elif reason is None:
            semantic = [d for d in drafts if not control_draft(output, d)]
            if semantic:
                selection = selector.select(
                    user_prompt=prompt, assistant_response=decode(output), drafts=semantic
                )
                selected = selection["selected_draft"]
                laya = selection["metadata"]
                candidate_evaluations = selection["candidate_evaluations"]
                raw_laya_result = selection["raw_result"]
                selection_results.append(selection)
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
                reason = "no_semantic_drafts_control_blocked"
                mode = "control_blocked"
                overrides += 1
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
                "selector_model_version": laya.get("model_version", selector.model_name),
            })
        if reason:
            break
    if reason is None:
        reason = "max_tokens" if len(output)+k > max_tokens else "context_limit"
    response = decode([token for token in output if token not in STOP_IDS])
    for marker, _ in MARKERS:
        response = response.split(marker, 1)[0]
    elapsed = time.perf_counter()-started
    for row in rows:
        row.update(final_response=response, termination_reason=reason,
                   control_overrides=overrides, elapsed_seconds=elapsed)
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
        print(f"Control overrides={overrides}; CSV={csv_path}")
    return {"response": response, "rows": rows,
            "jev_selections": selection_results, "termination_reason": reason,
            "output_tokens": len(output), "elapsed_seconds": elapsed,
            "jev_seconds": laya_time, "gemma_seconds": gemma_time}



def greedy_baseline(prompt, baseline_path=None, max_tokens=None):
    """Run Gemma alone with identical prompt framing and context budget."""
    max_tokens = MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
    context = prompt_tokens(prompt)
    generated = []
    start = time.perf_counter()
    reason = "context_limit"
    while len(context) < N_CTX and len(generated) < max_tokens:
        llm.reset()
        llm.eval(context)
        logits = llm.scores[llm.n_tokens - 1]
        token = int(np.argmax(logits))
        if token in STOP_IDS:
            reason = stop_reason([token]) or "stop_token"
            break
        generated.append(token)
        context.append(token)
    if reason == "context_limit" and len(generated) >= max_tokens:
        reason = "max_tokens"
    result = decode(generated)
    duration = time.perf_counter() - start
    if baseline_path is not None:
        baseline_path.write_text(
        f"Prompt: {prompt}\nAnswer: {result}\nStop reason: {reason}\n"
        f"Tokens: {len(generated)}\nSeconds: {duration:.3f}\n",
        encoding="utf-8"
    )
    print(f"Gemma-only baseline: {result!r} ({reason}, {duration:.2f}s)")
    return {"response": result, "termination_reason": reason,
            "output_tokens": len(generated), "elapsed_seconds": duration}

def main():
    parser = argparse.ArgumentParser(description="Gemma-JEV speculative decoder")
    parser.add_argument("--model", default=str(MODEL_PATH), help="Gemma GGUF path")
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
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
        run(prompt, args.k, selector, csv_path, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
