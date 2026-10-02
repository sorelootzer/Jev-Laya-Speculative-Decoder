from llama_cpp import Llama
import numpy as np
import time
import csv
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"C:\Users\lootz\Models\Gemma4"
    r"\gemma-4-E4B-it-Q4_K_M.gguf"
)

# Draft configuration
K = 2
BRANCHES = 2

# Development failsafe.
# This is NOT the speculative draft length.
MAX_OUTPUT_TOKENS = 100

SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "decoder_results.csv"


# ============================================================
# LOAD GEMMA
# ============================================================

print("Loading Gemma...")

load_start = time.perf_counter()

llm = Llama(
    model_path=MODEL_PATH,

    # More than enough for development/testing.
    n_ctx=2048,

    # Required for direct access to logits.
    logits_all=True,

    # CPU inference
    n_gpu_layers=0,

    verbose=False,
)

load_end = time.perf_counter()

MODEL_LOAD_TIME = load_end - load_start

print(
    f"Model loaded in "
    f"{MODEL_LOAD_TIME:.3f} s"
)


# ============================================================
# LOGIT UTILITIES
# ============================================================

def softmax(logits):
    """
    Convert raw logits into probabilities.
    """

    shifted = logits - np.max(logits)

    exp_logits = np.exp(shifted)

    return exp_logits / np.sum(exp_logits)


def top_tokens(logits, n=2):
    """
    Extract the top-n candidate tokens from a logit vector.

    Returns:
        [
            {
                "id": ...,
                "text": ...,
                "logit": ...,
                "probability": ...
            },
            ...
        ]
    """

    probabilities = softmax(logits)

    # Find top n without sorting the entire vocabulary.
    indices = np.argpartition(
        probabilities,
        -n
    )[-n:]

    # Sort the selected candidates highest -> lowest.
    indices = indices[
        np.argsort(
            probabilities[indices]
        )[::-1]
    ]

    results = []

    for token_id in indices:

        token_id = int(token_id)

        token_text = llm.detokenize(
            [token_id]
        ).decode(
            "utf-8",
            errors="replace"
        )

        results.append({
            "id": token_id,

            "text": token_text,

            "logit": float(
                logits[token_id]
            ),

            "probability": float(
                probabilities[token_id]
            )
        })

    return results


# ============================================================
# GEMMA CHAT FORMAT
# ============================================================

def create_chat_context(user_prompt):
    """
    Format the user's input as a Gemma conversation.

    Result conceptually:

        <start_of_turn>user
        USER PROMPT<end_of_turn>
        <start_of_turn>model

    Generation then begins after the model turn marker.
    """

    formatted_prompt = (
        "<start_of_turn>user\n"
        + user_prompt
        + "<end_of_turn>\n"
        + "<start_of_turn>model\n"
    )

    context_tokens = llm.tokenize(
        formatted_prompt.encode("utf-8"),

        # Add beginning-of-sequence token.
        add_bos=True,

        # Critical:
        # recognise Gemma control tokens as special tokens.
        special=True
    )

    return formatted_prompt, context_tokens


# ============================================================
# STOP TOKENS
# ============================================================

def get_stop_token_ids():
    """
    Find known termination tokens recognised by this tokenizer.

    EOS is always included.

    We also attempt to identify Gemma's end-of-turn token.
    """

    stop_ids = {
        llm.token_eos()
    }

    possible_stop_tokens = [
        b"<end_of_turn>",
        b"<eos>",
    ]

    for token_text in possible_stop_tokens:

        ids = llm.tokenize(
            token_text,
            add_bos=False,
            special=True
        )

        # Only consider it a special termination token
        # if the tokenizer represents it as ONE token.
        if len(ids) == 1:
            stop_ids.add(ids[0])

    return stop_ids


STOP_TOKEN_IDS = get_stop_token_ids()

print(
    "Stop token IDs:",
    STOP_TOKEN_IDS
)


# ============================================================
# EMPTY-DRAFT CHECKS
# ============================================================

def is_empty_draft(draft):
    """
    True when a selected draft has no visible output.
    """

    text = draft.get(
        "text",
        ""
    )

    if text is None:
        return True

    return text.strip() == ""


def all_drafts_empty(drafts):
    """
    True when every candidate draft contains
    no visible output.
    """

    for draft in drafts:

        text = draft.get(
            "text",
            ""
        )

        if text and text.strip():
            return False

    return True


# ============================================================
# GENERATE k=2, b=2 TOKEN TREE
# ============================================================

def generate_drafts(context_tokens):
    """
    Generate:

                        context
                           |
                    ----------------
                    |              |
                   T1              T2
                  /  \\            /  \\
                T11 T12         T21 T22


    Produces exactly four two-token drafts:

        11
        12
        21
        22
    """

    generation_start = time.perf_counter()


    # ========================================================
    # LEVEL 1
    # ========================================================

    level1_start = time.perf_counter()

    llm.reset()

    llm.eval(
        context_tokens
    )

    # Last evaluated position predicts the NEXT token.
    level1_logits = llm.scores[
        llm.n_tokens - 1
    ].copy()

    first_candidates = top_tokens(
        level1_logits,
        BRANCHES
    )

    level1_end = time.perf_counter()


    # ========================================================
    # LEVEL 2
    # ========================================================

    level2_start = time.perf_counter()

    drafts = []

    for first_index, first in enumerate(
        first_candidates,
        start=1
    ):

        # ----------------------------------------------------
        # Evaluate:
        #
        # context + T1
        #
        # and separately:
        #
        # context + T2
        #
        # ----------------------------------------------------

        llm.reset()

        branch_context = (
            context_tokens
            + [first["id"]]
        )

        llm.eval(
            branch_context
        )

        second_logits = llm.scores[
            llm.n_tokens - 1
        ].copy()

        second_candidates = top_tokens(
            second_logits,
            BRANCHES
        )

        # ----------------------------------------------------
        # Construct:
        #
        # T11 / T12
        #
        # or:
        #
        # T21 / T22
        # ----------------------------------------------------

        for second_index, second in enumerate(
            second_candidates,
            start=1
        ):

            draft_name = (
                f"{first_index}"
                f"{second_index}"
            )

            token_ids = [
                first["id"],
                second["id"]
            ]

            draft_text = llm.detokenize(
                token_ids
            ).decode(
                "utf-8",
                errors="replace"
            )

            # ------------------------------------------------
            # Joint probability
            #
            # P(T1,T11)
            #
            # =
            #
            # P(T1 | context)
            #
            # *
            #
            # P(T11 | context,T1)
            # ------------------------------------------------

            joint_probability = (
                first["probability"]
                *
                second["probability"]
            )

            # Log probability is numerically preferable
            # when sequence lengths increase.
            joint_log_probability = (
                np.log(
                    max(
                        first["probability"],
                        1e-45
                    )
                )
                +
                np.log(
                    max(
                        second["probability"],
                        1e-45
                    )
                )
            )

            drafts.append({

                "name":
                    draft_name,

                "token_ids":
                    token_ids,

                "tokens": [
                    first["text"],
                    second["text"]
                ],

                "text":
                    draft_text,

                # First token information
                "token_1_id":
                    first["id"],

                "token_1_text":
                    first["text"],

                "token_1_logit":
                    first["logit"],

                "first_probability":
                    first["probability"],

                # Second token information
                "token_2_id":
                    second["id"],

                "token_2_text":
                    second["text"],

                "token_2_logit":
                    second["logit"],

                "second_probability":
                    second["probability"],

                # Complete draft probability
                "joint_probability":
                    joint_probability,

                "joint_log_probability":
                    joint_log_probability
            })


    level2_end = time.perf_counter()

    generation_end = time.perf_counter()


    # ========================================================
    # GENERATION METRICS
    # ========================================================

    metrics = {

        "level1_seconds":
            level1_end
            - level1_start,

        "level2_seconds":
            level2_end
            - level2_start,

        "total_seconds":
            generation_end
            - generation_start
    }

    return drafts, metrics


# ============================================================
# DRAFT SELECTOR
# ============================================================

def select_draft(
    context_tokens,
    drafts
):
    """
    TEMPORARY SELECTOR.

    For now:
        choose Gemma's highest joint-probability draft.

    Later:
        replace the contents of THIS FUNCTION with Laya.

    The rest of the decoder does not need to change.
    """

    selector_start = time.perf_counter()

    selected = max(
        drafts,

        key=lambda draft:
            draft[
                "joint_log_probability"
            ]
    )

    selector_end = time.perf_counter()

    selector_time = (
        selector_end
        - selector_start
    )

    return (
        selected,
        selector_time
    )


# ============================================================
# DISPLAY CANDIDATE DRAFTS
# ============================================================

def print_drafts(drafts):

    ranked = sorted(
        drafts,

        key=lambda draft:
            draft[
                "joint_log_probability"
            ],

        reverse=True
    )

    print(
        "\nCandidate drafts:"
    )

    for draft in ranked:

        print(
            f"  {draft['name']} | "
            f"{repr(draft['text'])} | "
            f"P={draft['joint_probability']:.6f}"
        )


# ============================================================
# COMMIT SELECTED DRAFT
# ============================================================

def commit_selected_draft(
    selected,
    context_tokens,
    generated_tokens
):
    """
    Append selected tokens directly to the context.

    IMPORTANT:

    We append TOKEN IDs.

    We do NOT:

        token IDs
            ->
        text
            ->
        tokenizer
            ->
        token IDs

    This preserves Gemma's exact token sequence.

    If a stop token occurs inside the k=2 draft,
    only tokens BEFORE that stop token are committed.
    """

    should_stop = False

    committed_tokens = []

    for token_id in selected[
        "token_ids"
    ]:

        if token_id in STOP_TOKEN_IDS:

            should_stop = True

            break

        context_tokens.append(
            token_id
        )

        generated_tokens.append(
            token_id
        )

        committed_tokens.append(
            token_id
        )

    return (
        committed_tokens,
        should_stop
    )


# ============================================================
# MAIN ITERATIVE DECODER
# ============================================================

def generate_response(user_prompt):

    total_start = time.perf_counter()


    # ========================================================
    # CREATE INITIAL CHAT CONTEXT
    # ========================================================

    (
        formatted_prompt,
        context_tokens
    ) = create_chat_context(
        user_prompt
    )

    initial_context_tokens = len(
        context_tokens
    )

    generated_tokens = []

    history = []

    iteration = 0

    total_gemma_time = 0.0

    total_selector_time = 0.0


    # ========================================================
    # DISPLAY INPUT
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "USER"
    )

    print(
        "========================================"
    )

    print(
        user_prompt
    )

    print(
        "\n========================================"
    )

    print(
        "ASSISTANT"
    )

    print(
        "========================================\n"
    )


    # ========================================================
    # GENERATION LOOP
    # ========================================================

    try:

        while (
            len(generated_tokens)
            < MAX_OUTPUT_TOKENS
        ):

            iteration += 1


            # =================================================
            # GEMMA DRAFT GENERATION
            # =================================================

            drafts, gemma_metrics = (
                generate_drafts(
                    context_tokens
                )
            )

            total_gemma_time += (
                gemma_metrics[
                    "total_seconds"
                ]
            )


            print(
                f"\n--- Iteration "
                f"{iteration} ---"
            )

            print_drafts(
                drafts
            )


            # =================================================
            # TERMINATION CHECK 1:
            # ALL DRAFTS EMPTY
            # =================================================

            if all_drafts_empty(
                drafts
            ):

                print(
                    "\nAll candidate drafts "
                    "are empty."
                )

                print(
                    "Ending generation."
                )

                break


            # =================================================
            # SELECT DRAFT
            # =================================================

            (
                selected,
                selector_time
            ) = select_draft(
                context_tokens,
                drafts
            )

            total_selector_time += (
                selector_time
            )


            # =================================================
            # TERMINATION CHECK 2:
            # SELECTED DRAFT EMPTY
            # =================================================

            if is_empty_draft(
                selected
            ):

                print(
                    "\nSelected draft contains "
                    "no visible output."
                )

                print(
                    "Ending generation."
                )

                break


            # =================================================
            # DISPLAY SELECTED DRAFT
            # =================================================

            print(
                f"\nSelected: "
                f"{selected['name']} "
                f"{repr(selected['text'])}"
            )


            # =================================================
            # FIND GEMMA'S RANKING
            # =================================================

            ranked = sorted(
                drafts,

                key=lambda draft:
                    draft[
                        "joint_log_probability"
                    ],

                reverse=True
            )

            gemma_best = ranked[0]


            # =================================================
            # SAVE ITERATION HISTORY
            # =================================================

            history.append({

                "iteration":
                    iteration,

                # ---------------------------------------------
                # Candidate drafts
                # ---------------------------------------------

                "draft_11":
                    next(
                        d["text"]
                        for d in drafts
                        if d["name"] == "11"
                    ),

                "draft_12":
                    next(
                        d["text"]
                        for d in drafts
                        if d["name"] == "12"
                    ),

                "draft_21":
                    next(
                        d["text"]
                        for d in drafts
                        if d["name"] == "21"
                    ),

                "draft_22":
                    next(
                        d["text"]
                        for d in drafts
                        if d["name"] == "22"
                    ),

                # ---------------------------------------------
                # Gemma's preferred draft
                # ---------------------------------------------

                "gemma_best_draft":
                    gemma_best[
                        "name"
                    ],

                "gemma_best_text":
                    gemma_best[
                        "text"
                    ],

                # ---------------------------------------------
                # Selector result
                # ---------------------------------------------

                "selected_draft":
                    selected[
                        "name"
                    ],

                "selected_text":
                    selected[
                        "text"
                    ],

                "selected_probability":
                    selected[
                        "joint_probability"
                    ],

                # ---------------------------------------------
                # Performance
                # ---------------------------------------------

                "gemma_level1_seconds":
                    gemma_metrics[
                        "level1_seconds"
                    ],

                "gemma_level2_seconds":
                    gemma_metrics[
                        "level2_seconds"
                    ],

                "gemma_total_seconds":
                    gemma_metrics[
                        "total_seconds"
                    ],

                "selector_seconds":
                    selector_time
            })


            # =================================================
            # COMMIT SELECTED TOKENS
            # =================================================

            (
                committed,
                should_stop
            ) = commit_selected_draft(
                selected,
                context_tokens,
                generated_tokens
            )


            # =================================================
            # STOP TOKEN GENERATED
            # =================================================

            if should_stop:

                print(
                    "\nGemma generated a "
                    "termination token."
                )

                print(
                    "Ending generation."
                )

                break


            # =================================================
            # ADDITIONAL FAILSAFE:
            #
            # If somehow no tokens were committed,
            # continuing would cause an infinite loop.
            # =================================================

            if len(committed) == 0:

                print(
                    "\nNo tokens were committed."
                )

                print(
                    "Ending generation to prevent "
                    "an infinite loop."
                )

                break


    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print(
            "\n\nGeneration interrupted "
            "by user."
        )


    # ========================================================
    # DECODE FINAL RESPONSE
    # ========================================================

    final_response = llm.detokenize(
        generated_tokens
    ).decode(
        "utf-8",
        errors="replace"
    )


    total_end = time.perf_counter()

    total_time = (
        total_end
        - total_start
    )

    output_token_count = len(
        generated_tokens
    )


    # ========================================================
    # EFFECTIVE THROUGHPUT
    # ========================================================

    if total_time > 0:

        effective_tokens_per_second = (
            output_token_count
            /
            total_time
        )

    else:

        effective_tokens_per_second = 0.0


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n\n========================================"
    )

    print(
        "FINAL RESPONSE"
    )

    print(
        "========================================"
    )

    print(
        final_response
    )


    # ========================================================
    # METRICS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "METRICS"
    )

    print(
        "========================================"
    )

    print(
        f"Model load:             "
        f"{MODEL_LOAD_TIME:.3f} s"
    )

    print(
        f"Iterations:             "
        f"{iteration}"
    )

    print(
        f"Generated tokens:       "
        f"{output_token_count}"
    )

    print(
        f"Selector calls:         "
        f"{len(history)}"
    )

    print(
        f"Gemma draft time:       "
        f"{total_gemma_time:.3f} s"
    )

    print(
        f"Selector time:          "
        f"{total_selector_time:.6f} s"
    )

    print(
        f"End-to-end time:        "
        f"{total_time:.3f} s"
    )

    print(
        f"Effective throughput:   "
        f"{effective_tokens_per_second:.3f} "
        f"tokens/s"
    )

    print(
        "========================================"
    )


    # ========================================================
    # RETURN RESULTS
    # ========================================================

    metrics = {

        "iterations":
            iteration,

        "generated_tokens":
            output_token_count,

        "gemma_seconds":
            total_gemma_time,

        "selector_seconds":
            total_selector_time,

        "total_seconds":
            total_time,

        "tokens_per_second":
            effective_tokens_per_second,

        "initial_context_tokens":
            initial_context_tokens
    }

    return (
        final_response,
        history,
        metrics
    )


# ============================================================
# SAVE ITERATION HISTORY TO CSV
# ============================================================

def save_history(
    user_prompt,
    final_response,
    history,
    metrics
):

    fieldnames = [

        "prompt",

        "final_response",

        "iteration",

        "draft_11",
        "draft_12",
        "draft_21",
        "draft_22",

        "gemma_best_draft",
        "gemma_best_text",

        "selected_draft",
        "selected_text",
        "selected_probability",

        "gemma_level1_seconds",
        "gemma_level2_seconds",
        "gemma_total_seconds",

        "selector_seconds",

        "total_generated_tokens",

        "total_end_to_end_seconds",

        "effective_tokens_per_second"
    ]


    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()


        for row in history:

            writer.writerow({

                "prompt":
                    user_prompt,

                "final_response":
                    final_response,

                **row,

                "total_generated_tokens":
                    metrics[
                        "generated_tokens"
                    ],

                "total_end_to_end_seconds":
                    metrics[
                        "total_seconds"
                    ],

                "effective_tokens_per_second":
                    metrics[
                        "tokens_per_second"
                    ]
            })


    print(
        "\nResults saved to:"
    )

    print(
        CSV_FILE.resolve()
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

def main():

    user_prompt = input(
        "\nEnter prompt: "
    )

    (
        final_response,
        history,
        metrics
    ) = generate_response(
        user_prompt
    )

    save_history(
        user_prompt,
        final_response,
        history,
        metrics
    )


if __name__ == "__main__":
    main()