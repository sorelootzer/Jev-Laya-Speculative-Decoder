"""Interactive presentation runner using the existing benchmark implementation."""
import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from benchmark_common import GGUF, compatible, decode, shared_prompt


MODEL_DIR = Path(
    r"C:\Users\User\Desktop\School Stuff\ESP3201A"
    r"\AccelerateLM\models\Gemma4"
)


def successful(result):
    return (
        not result["error"]
        and result["stop_reason"] in ("eog", "max_tokens")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=128)
    args_cli = parser.parse_args()

    if args_cli.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    # Settings expected by your existing GGUF adapter.
    settings = argparse.Namespace(
        n_ctx=1024,
        n_batch=256,
        threads=max(1, (os.cpu_count() or 2) // 2),
        gpu_layers=-1,
        verbose=False,
    )

    paths = {
        "draft": MODEL_DIR / "gemma-4-E4B-it-Q4_K_M.gguf",
        "target": MODEL_DIR / "gemma-4-12b-it-Q4_K_M.gguf",
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Model not found: {path}")

    draft = target = None

    try:
        print("Loading both models once...", flush=True)
        draft = GGUF(paths["draft"], settings)
        target = GGUF(paths["target"], settings, all_logits=True)

        compatibility = compatible(draft, target)
        stops = target.stops([])

        print("Tokenizer compatibility passed.")
        print("Warming up both methods...", flush=True)

        _, warm_ids, _ = shared_prompt(
            target, draft, "What is 2 + 2?"
        )

        for drafter in (None, draft):
            result = decode(
                target, warm_ids, stops, 8,
                draft=drafter, k=2, timeout=300,
            )
            if not successful(result):
                raise RuntimeError(
                    f"Warm-up failed: {result['stop_reason']} "
                    f"{result['error']}"
                )

        print("\nReady. Type exit at the prompt to finish.")
        print("Both models remain loaded during both methods.")

        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ("exit", "quit"):
                    break
                if not prompt:
                    continue

                while True:
                    value = input("Draft length K [2]: ").strip() or "2"
                    try:
                        k = int(value)
                        if not 1 <= k <= min(256, args_cli.max_tokens):
                            raise ValueError
                        break
                    except ValueError:
                        print(
                            "Enter an integer from 1 to "
                            f"{min(256, args_cli.max_tokens)}."
                        )

                rendered, ids, template_hash = shared_prompt(
                    target, draft, prompt
                )

                if len(ids) + args_cli.max_tokens > settings.n_ctx:
                    print("Prompt is too long for the current context.")
                    continue

                print("Running 12B-only baseline...", flush=True)
                baseline = decode(
                    target, ids, stops, args_cli.max_tokens,
                    timeout=300,
                )
                baseline_fallbacks = target.fallbacks

                print(f"Running E4B -> 12B, K={k}...", flush=True)
                speculative = decode(
                    target, ids, stops, args_cli.max_tokens,
                    draft=draft, k=k, timeout=300,
                )

                valid = successful(baseline) and successful(speculative)
                same = (
                    baseline["output_ids"] == speculative["output_ids"]
                    and baseline["stop_reason"]
                    == speculative["stop_reason"]
                )
                status = (
                    ("MATCH" if same else "MISMATCH")
                    if valid else "INCOMPLETE / ERROR"
                )

                baseline_answer = target.text(baseline["output_ids"])
                speculative_answer = target.text(
                    speculative["output_ids"]
                )

                # Trace formatting happens outside the decoding timers.
                for event in speculative["events"]:
                    event["proposed_text"] = target.text(
                        event["proposal_ids"]
                    )
                    event["emitted_text"] = target.text(
                        event["committed_ids"]
                    )

                lines = [
                    "ESP3201 LIVE SPECULATIVE DECODING",
                    "",
                    f"Prompt: {prompt}",
                    f"Draft length K: {k}",
                    "",
                    "12B-ONLY ANSWER",
                    baseline_answer,
                    "",
                    "SPECULATIVE ANSWER",
                    speculative_answer,
                    "",
                    f"Output check: {status}",
                ]

                for name, result in (
                    ("12B-only", baseline),
                    ("Speculative", speculative),
                ):
                    lines.append(
                        f"{name}: {result['response_seconds']:.4f} s; "
                        f"{result['output_tokens']} tokens; "
                        f"{result['tokens_per_second']:.2f} tok/s; "
                        f"stop={result['stop_reason']}; "
                        f"error={result['error'] or 'none'}"
                    )

                ratio = None
                if valid and speculative["response_seconds"] > 0:
                    ratio = (
                        baseline["response_seconds"]
                        / speculative["response_seconds"]
                    )
                    label = (
                        "speedup ratio" if same
                        else "latency ratio; MISMATCH, not validated speedup"
                    )
                    lines.append(f"{label}: {ratio:.3f}x")
                    if same and ratio < 1:
                        lines.append("Speculative decoding was slower.")

                rate = speculative["acceptance_rate"]
                if rate is not None:
                    lines.append(f"Draft acceptance: {rate:.1%}")

                lines.extend([
                    "",
                    "DRAFT / VERIFY TRACE",
                ])

                for event in speculative["events"]:
                    lines.extend([
                        f"Round {event['iteration']}: "
                        f"proposed={event['proposed']}, "
                        f"accepted={event['accepted']}, "
                        f"rejected suffix={event['rejected_suffix']}",
                        f"  Proposed: {event['proposed_text']!r}",
                        f"  Emitted:  {event['emitted_text']!r}",
                    ])

                lines.extend([
                    "",
                    "TIMING NOTES",
                    "Both models remain loaded during both methods.",
                    "The same target uses logits_all=True in both methods.",
                    "Timers include reset, prefill and generation.",
                    "Model loading, formatting and report writing are excluded.",
                    "Baseline runs first; this is one demonstration trial.",
                    "Acceptance includes accepted end-of-generation tokens.",
                    "Full GPU residency and VRAM usage are not measured.",
                    "A token match is not an answer-quality assessment.",
                ])

                folder = Path("presentation_results") / (
                    datetime.now().strftime("%Y%m%d_%H%M%S")
                    + "_" + uuid4().hex[:8]
                )
                folder.mkdir(parents=True, exist_ok=False)

                report = "\n".join(lines)
                (folder / "report.txt").write_text(
                    report, encoding="utf-8-sig"
                )

                record = {
                    "prompt": prompt,
                    "k": k,
                    "settings": vars(settings),
                    "max_tokens": args_cli.max_tokens,
                    "models": {key: str(p) for key, p in paths.items()},
                    "compatibility": compatibility,
                    "formatted_prompt": rendered,
                    "template_sha256": template_hash,
                    "baseline": baseline,
                    "speculative": speculative,
                    "output_check": status,
                    "latency_ratio": ratio,
                    "cache_fallbacks": {
                        "baseline_target": baseline_fallbacks,
                        "speculative_target": target.fallbacks,
                        "draft": draft.fallbacks,
                    },
                }
                (folder / "results.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                with (folder / "summary.csv").open(
                    "w", newline="", encoding="utf-8-sig"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "method", "k", "seconds", "output_tokens",
                        "tokens_per_second", "stop_reason", "output_check",
                    ])
                    for name, result in (
                        ("baseline", baseline),
                        ("speculative", speculative),
                    ):
                        writer.writerow([
                            name, k if name == "speculative" else "",
                            result["response_seconds"],
                            result["output_tokens"],
                            result["tokens_per_second"],
                            result["stop_reason"], status,
                        ])

                print("\n" + report)
                report_path = (folder / "report.txt").resolve()
                print("\nSaved:", report_path)

                if os.name == "nt":
                    try:
                        os.startfile(str(report_path))
                    except OSError:
                        print("Open the saved report manually.")

                if not valid or not same:
                    print(
                        "\nInvestigate this result before claiming speedup."
                    )

                if any(
                    r["stop_reason"] == "interrupted"
                    for r in (baseline, speculative)
                ):
                    break

            except (EOFError, KeyboardInterrupt):
                break

    finally:
        for model in (draft, target):
            if model is not None:
                model.llm.close()


if __name__ == "__main__":
    main()