import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--out", default="summary_smoke_seed42.json")
    args = parser.parse_args()

    output = Path(args.out)
    if output.exists():
        parser.error("Output exists. Choose another filename.")
    if args.count < 1:
        parser.error("Count must be positive.")

    from datasets import load_dataset
    from llama_cpp import Llama
    from benchmark_common import GGUF

    # Load tokenizer/metadata only; no generation or GPU required.
    renderer = GGUF.__new__(GGUF)
    renderer.llm = Llama(
        model_path=args.model,
        vocab_only=True,
        n_gpu_layers=0,
        verbose=False,
    )
    examples = []

    try:
        specs = [
            ("samsum", "knkarthick/samsum", None),
            ("cnn_dailymail", "abisee/cnn_dailymail", "3.0.0"),
        ]

        for name, repository, config in specs:
            print("Loading", repository, flush=True)
            ds = load_dataset(repository, config, split="validation")
            indices = list(range(len(ds)))
            random.Random(42).shuffle(indices)
            selected = 0

            for index in indices:
                row = ds[index]

                if name == "samsum":
                    source, reference = row["dialogue"], row["summary"]
                    instruction = (
                        "Summarize this conversation accurately in "
                        "2-4 complete sentences. Do not invent details."
                    )
                else:
                    source, reference = row["article"], row["highlights"]
                    instruction = (
                        "Summarize this news article accurately in "
                        "4-6 complete sentences. Include the main facts "
                        "without inventing details."
                    )

                if not source.strip() or not reference.strip():
                    continue

                prompt = instruction + "\n\n" + source + "\n\nSummary:"
                _, tokens, template_hash = renderer.render(prompt)

                # Keep complete sources; skip those exceeding the budget.
                if len(tokens) > 1400:
                    continue

                examples.append({
                    "id": f"{name}_{index}",
                    "category": name,
                    "prompt": prompt,
                    "reference": reference,
                    "source": source,
                    "dataset": repository,
                    "dataset_config": config,
                    "dataset_fingerprint": ds._fingerprint,
                    "source_index": index,
                    "split": "validation",
                    "seed": 42,
                    "input_tokens": len(tokens),
                    "template_sha256": template_hash,
                    "selection_policy": (
                        "Seed-42 shuffle; complete sources fitting "
                        "1400 target-formatted input tokens"
                    ),
                })
                selected += 1
                if selected == args.count:
                    break

            if selected != args.count:
                raise RuntimeError(
                    f"Insufficient eligible examples for {name}"
                )
    finally:
        renderer.llm.close()

    with output.open("x", encoding="utf-8") as file:
        json.dump(examples, file, ensure_ascii=False, indent=2)

    print(f"Saved {len(examples)} questions to {output.resolve()}")


if __name__ == "__main__":
    main()