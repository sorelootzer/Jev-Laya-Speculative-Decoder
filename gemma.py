from llama_cpp import Llama
from laya_selector import LayaSelector
import csv
import json
import math
import time
from pathlib import Path

import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent
MODEL_PATH = Path(r"C:\Users\lootz\Models\Gemma4\gemma-4-E4B-it-Q4_K_M.gguf")

N_CTX = 2048
DEFAULT_K = 2
BRANCHES = 2
CSV_FILE = SCRIPT_DIR / "decoder_results.csv"


# ============================================================
# MODEL STATE
# ============================================================

llm = None
MODEL_LOAD_TIME = 0.0


def load_gemma():
    global llm, MODEL_LOAD_TIME

    if llm is not None:
        return

    print("Loading Gemma...")
    load_start = time.perf_counter()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        logits_all=True,
        n_gpu_layers=0,
        verbose=False,
    )

    MODEL_LOAD_TIME = time.perf_counter() - load_start
    print(f"Gemma loaded in {MODEL_LOAD_TIME:.3f} s")


# ============================================================
# LOGIT UTILITIES
# ============================================================

def softmax(logits):
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)


def top_tokens(logits, n=BRANCHES):
    probabilities = softmax(logits)
    indices = np.argpartition(probabilities, -n)[-n:]
    indices = indices[np.argsort(probabilities[indices])[::-1]]

    candidates = []
    for token_id in indices:
        token_id = int(token_id)
        text = llm.detokenize([token_id]).decode("utf-8", errors="replace")
        candidates.append({
            "id": token_id,
            "text": text,
            "logit": float(logits[token_id]),
            "probability": float(probabilities[token_id]),
        })

    return candidates


# ============================================================
# CHAT / STOP TOKENS
# ============================================================

def create_chat_context(user_prompt):
    formatted_prompt = (
        "<start_of_turn>user\n"
        + user_prompt
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )

    tokens = llm.tokenize(
        formatted_prompt.encode("utf-8"),
        add_bos=True,
        special=True,
    )

    return formatted_prompt, tokens


def get_stop_token_ids():
    stop_ids = {llm.token_eos()}
    possible_stop_tokens = [
        
    b"<start_end_turn/>",
    b"<eos>",
    ]
    for token_text in (b"<end_of_turn>", b"<eos>"):
        ids = llm.tokenize(token_text, add_bos=False, special=True)
        if len(ids) == 1:
            stop_ids.add(ids[0])

    return stop_ids


STOP_TOKEN_IDS = get_stop_token_ids()

print("\n========== STOP TOKEN DIAGNOSTIC ==========")

# Normal EOS
print("EOS token ID:", llm.token_eos())

# How this tokenizer represents <end_of_turn>
eot_tokens = llm.tokenize(
    b"<start_end_turn/>",
    add_bos=False,
    special=True
)

print("<end_of_turn> token IDs:", eot_tokens)

print("\nIndividual token decoding:")

for token_id in eot_tokens:
    decoded = llm.detokenize(
        [token_id]
    ).decode(
        "utf-8",
        errors="replace"
    )

    print(
        f"ID {token_id}: {repr(decoded)}"
    )

# Verify the entire sequence decodes correctly
full_decoded = llm.detokenize(
    eot_tokens
).decode(
    "utf-8",
    errors="replace"
)

print(
    "\nFull sequence:",
    repr(full_decoded)
)

print(
    "\nDetected single-token stop IDs:",
    STOP_TOKEN_IDS
)

print("===========================================\n")

# ============================================================
# GENERIC TOKEN TREE
# ============================================================

def generate_drafts(context_tokens, k, branching_factor=BRANCHES):
    """
    Generate a b^k candidate tree while evaluating the persistent context once.

    The root context is evaluated once and saved. Each tree node restores that
    root state and evaluates only its speculative path before reading logits.
    This avoids re-evaluating the full prompt/accepted response for every node.

    Terminal paths are not expanded further. They are returned as terminal
    candidates and may contain fewer than k visible tokens.
    """

    if k < 1:
        raise ValueError("k must be at least 1")
    if branching_factor < 1:
        raise ValueError("branching_factor must be at least 1")
    if len(context_tokens) + k > N_CTX:
        raise RuntimeError(
            f"Insufficient context capacity: {len(context_tokens)} used, "
            f"k={k}, n_ctx={N_CTX}."
        )

    generation_start = time.perf_counter()
    level_times = [0.0 for _ in range(k)]
    node_counts = [0 for _ in range(k)]

    # Evaluate persistent prefix once.
    prefix_start = time.perf_counter()
    llm.reset()
    llm.eval(context_tokens)
    root_state = llm.save_state()
    prefix_seconds = time.perf_counter() - prefix_start

    leaves = []

    # Node representation:
    # path = branch-name string, e.g. "121"
    # token_ids / token records are the speculative path from the root.
    frontier = [{
        "name": "",
        "token_ids": [],
        "tokens": [],
        "joint_log_probability": 0.0,
        "terminal": False,
    }]

    for depth in range(k):
        level_start = time.perf_counter()
        next_frontier = []

        for node in frontier:
            if node["terminal"]:
                leaves.append(node)
                continue

            llm.load_state(root_state)

            if node["token_ids"]:
                llm.eval(node["token_ids"])

            logits = llm.scores[llm.n_tokens - 1].copy()
            candidates = top_tokens(logits, branching_factor)
            node_counts[depth] += 1

            for branch_index, candidate in enumerate(candidates, start=1):
                child_token_ids = node["token_ids"] + [candidate["id"]]
                child_tokens = node["tokens"] + [candidate]
                child_name = node["name"] + str(branch_index)
                child_log_probability = (
                    node["joint_log_probability"]
                    + math.log(max(candidate["probability"], 1e-45))
                )
                terminal = candidate["id"] in STOP_TOKEN_IDS

                child = {
                    "name": child_name,
                    "token_ids": child_token_ids,
                    "tokens": child_tokens,
                    "joint_log_probability": child_log_probability,
                    "terminal": terminal,
                }

                if terminal or depth == k - 1:
                    leaves.append(child)
                else:
                    next_frontier.append(child)

        level_times[depth] = time.perf_counter() - level_start
        frontier = next_frontier

        if not frontier:
            break

    drafts = []

    for leaf in leaves:
        text = llm.detokenize(leaf["token_ids"]).decode(
            "utf-8", errors="replace"
        )

        drafts.append({
            "name": leaf["name"],
            "token_ids": leaf["token_ids"],
            "tokens": leaf["tokens"],
            "text": text,
            "joint_probability": math.exp(leaf["joint_log_probability"]),
            "joint_log_probability": leaf["joint_log_probability"],
            "average_log_probability": (
                leaf["joint_log_probability"] / len(leaf["token_ids"])
            ),
            "terminal": leaf["terminal"],
            "draft_length": len(leaf["token_ids"]),
        })

    return drafts, {
        "prefix_seconds": prefix_seconds,
        "level_seconds": level_times,
        "node_counts": node_counts,
        "total_seconds": time.perf_counter() - generation_start,
        "draft_count": len(drafts),
    }


# ============================================================
# DISPLAY / COMMIT
# ============================================================

def print_drafts(drafts):
    ranked = sorted(
        drafts,
        key=lambda draft: draft["joint_log_probability"],
        reverse=True,
    )

    print("\nCandidate drafts:")
    for draft in ranked:
        terminal = " [STOP]" if draft["terminal"] else ""
        print(
            f"  {draft['name']} | {repr(draft['text'])} | "
            f"P={draft['joint_probability']:.6f}{terminal}"
        )


def commit_selected_draft(selected, context_tokens, generated_tokens):
    committed = []

    for token_id in selected["token_ids"]:
        if token_id in STOP_TOKEN_IDS:
            return committed, True

        context_tokens.append(token_id)
        generated_tokens.append(token_id)
        committed.append(token_id)

    return committed, bool(selected.get("terminal", False))


# ============================================================
# ITERATIVE DECODER
# ============================================================

def generate_response(user_prompt, k):
    if k < 1:
        raise ValueError("k must be at least 1")

    selector = LayaSelector(k=k, branching_factor=BRANCHES)

    total_start = time.perf_counter()
    _, context_tokens = create_chat_context(user_prompt)
    initial_context_tokens = len(context_tokens)
    generated_tokens = []
    history = []
    iteration = 0
    total_gemma_time = 0.0
    total_laya_time = 0.0

    print("\n========================================")
    print("USER")
    print("========================================")
    print(user_prompt)
    print("\n========================================")
    print("ASSISTANT")
    print("========================================")

    try:
        while True:
            # Natural hard ceiling: enough room must remain for a k-token draft.
            if len(context_tokens) + k > N_CTX:
                print("\nContext window exhausted. Ending generation.")
                break

            iteration += 1

            drafts, gemma_metrics = generate_drafts(
                context_tokens=context_tokens,
                k=k,
                branching_factor=BRANCHES,
            )
            total_gemma_time += gemma_metrics["total_seconds"]

            if not drafts:
                print("\nNo candidate drafts produced. Ending generation.")
                break

            print(f"\n--- Iteration {iteration} ---")
            print_drafts(drafts)

            # If the highest-probability first-step continuation is already a
            # terminal candidate, avoid forcing Laya to choose among paths that
            # conceptually continue after the model has ended its turn.
            terminal_drafts = [d for d in drafts if d["terminal"]]

            current_response = llm.detokenize(generated_tokens).decode(
                "utf-8", errors="replace"
            )

            # LayaSelector currently requires exactly b^k full-length drafts.
            # Until it accepts early-terminal paths, use Laya only when the tree
            # is complete. A terminal-only tree ends immediately.
            full_length_drafts = [d for d in drafts if len(d["token_ids"]) == k]

            if not full_length_drafts and terminal_drafts:
                selected = max(
                    terminal_drafts,
                    key=lambda d: d["joint_log_probability"],
                )
                laya_metadata = {
                    "decision_seconds": 0.0,
                    "selected_name": selected["name"],
                    "probabilities": {},
                    "confidence": None,
                    "answer_confidence": None,
                    "gemma_laya_agree": True,
                    "selection_mode": "terminal_short_circuit",
                }
            else:
                # A complete non-terminal tree contains exactly b^k leaves.
                selected, laya_metadata = selector.select(
                    user_prompt=user_prompt,
                    assistant_response=current_response,
                    drafts=full_length_drafts,
                )
                laya_metadata["selection_mode"] = "laya"

            total_laya_time += laya_metadata["decision_seconds"]

            print(
                f"\nSelected: {selected['name']} "
                f"{repr(selected['text'])}"
            )

            gemma_best = max(
                drafts,
                key=lambda draft: draft["joint_log_probability"],
            )

            history.append({
                "iteration": iteration,
                "context_tokens": len(context_tokens),
                "drafts": {
                    draft["name"]: {
                        "text": draft["text"],
                        "gemma_probability": draft["joint_probability"],
                        "average_log_probability": draft["average_log_probability"],
                        "terminal": draft["terminal"],
                    }
                    for draft in drafts
                },
                "gemma_best_draft": gemma_best["name"],
                "selected_draft": selected["name"],
                "selected_text": selected["text"],
                "gemma_prefix_seconds": gemma_metrics["prefix_seconds"],
                "gemma_level_seconds": gemma_metrics["level_seconds"],
                "gemma_total_seconds": gemma_metrics["total_seconds"],
                "laya_seconds": laya_metadata["decision_seconds"],
                "laya_probabilities": laya_metadata.get("probabilities", {}),
                "laya_confidence": laya_metadata.get("confidence"),
                "laya_answer_confidence": laya_metadata.get("answer_confidence"),
                "gemma_laya_agree": laya_metadata.get("gemma_laya_agree"),
                "selection_mode": laya_metadata.get("selection_mode"),
            })

            committed, should_stop = commit_selected_draft(
                selected,
                context_tokens,
                generated_tokens,
            )

            if should_stop:
                print("\n<end_of_turn>/EOS selected. Ending generation.")
                break

            if not committed:
                print("\nNo tokens committed. Ending generation.")
                break

    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user.")

    final_response = llm.detokenize(generated_tokens).decode(
        "utf-8", errors="replace"
    )
    final_response = final_response.split("<end_of_turn>", 1)[0]

    total_seconds = time.perf_counter() - total_start
    output_token_count = len(generated_tokens)
    throughput = output_token_count / total_seconds if total_seconds else 0.0

    metrics = {
        "k": k,
        "branching_factor": BRANCHES,
        "expected_drafts": BRANCHES ** k,
        "iterations": iteration,
        "generated_tokens": output_token_count,
        "initial_context_tokens": initial_context_tokens,
        "gemma_seconds": total_gemma_time,
        "laya_seconds": total_laya_time,
        "total_seconds": total_seconds,
        "tokens_per_second": throughput,
    }

    print("\n\n========================================")
    print("FINAL RESPONSE")
    print("========================================")
    print(final_response)
    print("\n========================================")
    print("METRICS")
    print("========================================")
    print(f"k:                      {k}")
    print(f"Expected drafts:        {BRANCHES ** k}")
    print(f"Iterations:             {iteration}")
    print(f"Generated tokens:       {output_token_count}")
    print(f"Gemma draft time:       {total_gemma_time:.3f} s")
    print(f"Laya decision time:     {total_laya_time:.3f} s")
    print(f"End-to-end time:        {total_seconds:.3f} s")
    print(f"Effective throughput:   {throughput:.3f} tokens/s")
    print("========================================")

    return final_response, history, metrics


# ============================================================
# CSV
# ============================================================

def save_history(user_prompt, final_response, history, metrics):
    fieldnames = [
        "prompt",
        "final_response",
        "k",
        "branching_factor",
        "iteration",
        "context_tokens",
        "draft_name",
        "draft_text",
        "gemma_probability",
        "average_log_probability",
        "terminal",
        "gemma_best_draft",
        "selected_draft",
        "selected_text",
        "laya_probability",
        "laya_confidence",
        "laya_answer_confidence",
        "gemma_laya_agree",
        "selection_mode",
        "gemma_prefix_seconds",
        "gemma_level_seconds",
        "gemma_total_seconds",
        "laya_seconds",
        "total_generated_tokens",
        "total_end_to_end_seconds",
        "effective_tokens_per_second",
    ]

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for iteration in history:
            for draft_name, draft in iteration["drafts"].items():
                writer.writerow({
                    "prompt": user_prompt,
                    "final_response": final_response,
                    "k": metrics["k"],
                    "branching_factor": metrics["branching_factor"],
                    "iteration": iteration["iteration"],
                    "context_tokens": iteration["context_tokens"],
                    "draft_name": draft_name,
                    "draft_text": draft["text"],
                    "gemma_probability": draft["gemma_probability"],
                    "average_log_probability": draft["average_log_probability"],
                    "terminal": draft["terminal"],
                    "gemma_best_draft": iteration["gemma_best_draft"],
                    "selected_draft": iteration["selected_draft"],
                    "selected_text": iteration["selected_text"],
                    "laya_probability": iteration["laya_probabilities"].get(draft_name),
                    "laya_confidence": iteration["laya_confidence"],
                    "laya_answer_confidence": iteration["laya_answer_confidence"],
                    "gemma_laya_agree": iteration["gemma_laya_agree"],
                    "selection_mode": iteration["selection_mode"],
                    "gemma_prefix_seconds": iteration["gemma_prefix_seconds"],
                    "gemma_level_seconds": json.dumps(iteration["gemma_level_seconds"]),
                    "gemma_total_seconds": iteration["gemma_total_seconds"],
                    "laya_seconds": iteration["laya_seconds"],
                    "total_generated_tokens": metrics["generated_tokens"],
                    "total_end_to_end_seconds": metrics["total_seconds"],
                    "effective_tokens_per_second": metrics["tokens_per_second"],
                })

    print(f"\nResults saved to:\n{CSV_FILE.resolve()}")


# ============================================================
# ENTRY POINT
# ============================================================

def read_k():
    raw = input(f"Draft length k [{DEFAULT_K}]: ").strip()
    if not raw:
        return DEFAULT_K

    k = int(raw)
    if k < 1:
        raise ValueError("k must be at least 1")

    # Guard against accidental exponential explosions during development.
    if BRANCHES ** k > 64:
        raise ValueError(
            f"k={k} would create {BRANCHES ** k} drafts. "
            "Current development limit is 64 drafts."
        )

    return k


def main():
    k = read_k()
    user_prompt = input("\nEnter prompt: ")

    # Ask for all runtime configuration before loading either model.
    load_gemma()

    final_response, history, metrics = generate_response(user_prompt, k)
    save_history(user_prompt, final_response, history, metrics)


if __name__ == "__main__":
    main()
