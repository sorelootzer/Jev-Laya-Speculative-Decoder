import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--out", default="nlp_smoke.json")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("Count must be positive.")

    output = Path(args.out)
    if output.exists():
        parser.error("Output already exists. Choose another filename.")

    specs = [
        ("gsm8k", "openai/gsm8k", "main", "test"),
        ("boolq", "google/boolq", None, "validation"),
        ("hellaswag", "Rowan/hellaswag", None, "validation"),
    ]
    prompts = []

    for name, repository, config, split in specs:
        dataset = load_dataset(repository, config, split=split)
        indices = random.Random(42).sample(
            range(len(dataset)), min(args.count, len(dataset))
        )

        for index in indices:
            row = dataset[index]

            if name == "gsm8k":
                prompt = (
                    row["question"]
                    + "\nEnd your answer with #### followed by the final number."
                )
                reference = row["answer"].rsplit("####", 1)[-1].strip()
                category = "arithmetic"

            elif name == "boolq":
                prompt = (
                    f"Passage: {row['passage']}\n"
                    f"Question: {row['question']}\n"
                    "Answer only Yes or No."
                )
                reference = "Yes" if row["answer"] else "No"
                category = "reading_comprehension"

            else:
                choices = "\n".join(
                    f"{chr(65 + i)}. {text}"
                    for i, text in enumerate(row["endings"])
                )
                prompt = (
                    "Choose the most plausible continuation.\n"
                    f"{row['ctx']}\n{choices}\n"
                    "Answer only A, B, C, or D."
                )
                reference = chr(65 + int(row["label"]))
                category = "commonsense"

            prompts.append({
                "id": f"{name}_{index}",
                "category": category,
                "prompt": prompt,
                "reference": reference,
                "dataset": repository,
                "split": split,
                "source_index": index,
                "seed": 42,
                "dataset_fingerprint": dataset._fingerprint,
            })

    output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(prompts)} prompts to {output.resolve()}")


if __name__ == "__main__":
    main()