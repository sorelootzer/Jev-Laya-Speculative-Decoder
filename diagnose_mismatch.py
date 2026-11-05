"""Replay a saved mismatch using the target model only."""
import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_common import GGUF, shared_prompt


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def top_candidates(model, scores):
    values = np.asarray(scores)
    order = np.argsort(values)[-5:][::-1]
    return {
        "top": [
            {
                "id": int(i),
                "text": model.text([int(i)]),
                "logit": float(values[i]),
            }
            for i in order
        ],
        "top_two_margin": float(values[order[0]] - values[order[1]]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_json")
    parser.add_argument("--model", help="Optional target GGUF path override")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--out", default="mismatch_diagnostic.json")
    args = parser.parse_args()

    record = json.loads(
        Path(args.results_json).read_text(encoding="utf-8-sig")
    )
    baseline = record["baseline"]["output_ids"]
    speculative = record["speculative"]["output_ids"]

    index = first_difference(baseline, speculative)
    if index is None:
        raise SystemExit("No differing output token found.")
    if index >= min(len(baseline), len(speculative)):
        raise SystemExit(
            "Length-only mismatch. Inspect the saved stop reasons."
        )

    models = record.get(
        "models", record.get("session", {}).get("models")
    )
    if args.model:
        model_path = args.model
    elif isinstance(models, dict):
        model_path = models["target"]
    else:
        model_path = models[1]["path"]

    settings = dict(record["settings"])
    settings.update(gpu_layers=args.gpu_layers, verbose=True)
    settings.setdefault("n_ctx", 1024)
    settings.setdefault("n_batch", 256)
    settings.setdefault("threads", 8)

    model = GGUF(
        model_path, argparse.Namespace(**settings), all_logits=True
    )

    try:
        rendered, ids, template_hash = shared_prompt(
            model,
            None,
            record["prompt"],
            settings.get("template_file"),
        )
        if template_hash != record["template_sha256"]:
            raise ValueError("Chat template changed.")
        if rendered != record["formatted_prompt"]:
            raise ValueError("Rendered prompt changed.")

        report = {
            "first_differing_output_position": index + 1,
            "baseline_token": baseline[index],
            "speculative_token": speculative[index],
            "gpu_layers": args.gpu_layers,
            "probes": {},
            "note": (
                "Target-only replay changes memory residency. "
                "These observations do not establish the root cause."
            ),
        }

        prefix = ids + baseline[:index]

        print("Probe 1: fresh full-prefix evaluation...", flush=True)
        model.reset()
        model.sync(prefix)
        report["probes"]["fresh_full_prefix"] = top_candidates(
            model, model.llm.scores[len(prefix) - 1].copy()
        )

        print("Probe 2: sequential output evaluation...", flush=True)
        model.reset()
        model.sync(ids)
        for i in range(index):
            model.sync(ids + baseline[:i + 1])
        report["probes"]["fresh_sequential_output_prefix"] = (
            top_candidates(
                model, model.llm.scores[len(prefix) - 1].copy()
            )
        )

        print("Probe 3: replaying saved verification calls...", flush=True)
        model.reset()
        emitted = []

        for event in record["speculative"]["events"]:
            context = ids + emitted
            if len(context) != event["context_tokens"]:
                raise ValueError("Saved trace context is inconsistent.")

            model.sync(context)
            before = model.llm.scores[len(context) - 1].copy()
            sampled = model.next_token()
            model.block(event["proposal_ids"])

            offset = index - len(emitted)
            if 0 <= offset < len(event["committed_ids"]):
                scores = (
                    before if offset == 0
                    else model.llm.scores[
                        len(context) + offset - 1
                    ].copy()
                )
                report["probes"]["replayed_verification"] = (
                    top_candidates(model, scores)
                )
                report["event"] = event
                report["sampled_first_prediction"] = sampled

                print(
                    "Probe 4: same block after sequential prefill...",
                    flush=True,
                )
                model.reset()
                model.sync(ids)
                for i in range(len(emitted)):
                    model.sync(ids + emitted[:i + 1])

                before = model.llm.scores[len(context) - 1].copy()
                model.block(event["proposal_ids"])
                scores = (
                    before if offset == 0
                    else model.llm.scores[
                        len(context) + offset - 1
                    ].copy()
                )
                report["probes"]["block_after_sequential_prefix"] = (
                    top_candidates(model, scores)
                )
                break

            emitted.extend(event["committed_ids"])
        else:
            raise ValueError(
                "No saved verification round spans the differing token."
            )

        # Refuse to overwrite an earlier diagnostic.
        with Path(args.out).open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)

        print("Saved:", Path(args.out).resolve())

    finally:
        model.llm.close()


if __name__ == "__main__":
    main()