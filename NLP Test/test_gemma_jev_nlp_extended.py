"""Paired Gemma-greedy vs Gemma-JEV speculative-decoding NLP evaluation.

Both arms receive the exact same A/B choice prompt. Requires local Gemma GGUF,
JEV API access, and Hugging Face datasets. Story Cloze downloads labeled validation data from Hugging Face by default.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

DATASETS = ("winogrande", "rte", "story_cloze", "blimp")
SCRIPT_DIR = Path(__file__).resolve().parent
# The benchmark lives in NLP Test/; the decoder, selector and models live one level up.
ROOT = SCRIPT_DIR.parent
if not (ROOT / "gemma_jev.py").is_file():
    raise FileNotFoundError(
        f"Expected gemma_jev.py in project root: {ROOT}. "
        "Place this benchmark inside the project's NLP Test folder."
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MODEL = ROOT / "models" / "Gemma4" / "gemma-4-E4B-it-Q4_K_M.gguf"


def make_example(dataset, row, idx, category=""):
    if dataset == "winogrande":
        # The gold label is "1" or "2"; the model sees full substitutions.
        answer = str(row.get("answer", ""))
        if answer not in ("1", "2"):
            return None
        sentence = row["sentence"]
        if "_" not in sentence:
            raise ValueError("WinoGrande example has no blank")
        options = [sentence.replace("_", row["option1"]),
                   sentence.replace("_", row["option2"])]
        gold = int(answer) - 1
        prompt = "Select the most coherent replacement for the blank.\nSentence: " + sentence
    elif dataset == "rte":
        label = row.get("label")
        if label not in (0, 1):
            return None
        # GLUE RTE: 0 entailment, 1 not_entailment.
        options = ["entailment", "not entailment"]
        gold = int(label)
        prompt = f"Premise: {row['sentence1']}\nHypothesis: {row['sentence2']}\nDoes the premise entail the hypothesis?"
    elif dataset == "story_cloze":
        # Story Cloze 2016/2018 commonly stores four context sentences,
        # two candidate endings and AnswerRightEnding=1/2.
        def get(*names):
            for name in names:
                if name in row and str(row[name]).strip():
                    return str(row[name]).strip()
            raise ValueError("Missing Story Cloze column: " + str(names))
        answer = get("AnswerRightEnding", "answer_right_ending", "answer", "label")
        if answer not in ("1", "2"):
            return None
        story = " ".join(get(f"InputSentence{i}", f"input_sentence_{i}", f"sentence{i}") for i in range(1, 5))
        options = [get("RandomFifthSentenceQuiz1", "sentence_quiz1", "ending1"),
                   get("RandomFifthSentenceQuiz2", "sentence_quiz2", "ending2")]
        gold = int(answer)-1
        prompt = "Choose the most coherent ending to this story.\n" + story
    elif dataset == "blimp":
        options = [row["sentence_good"], row["sentence_bad"]]
        gold = 0
        prompt = "Choose the grammatically acceptable English sentence."
        category = category or str(row.get("linguistics_term", "unknown"))
    else:
        raise ValueError("Unsupported dataset " + dataset)
    if options[0].strip() == options[1].strip():
        return None
    return {"question_id": idx, "dataset": dataset, "prompt": prompt,
            "options": options, "gold_index": gold, "category": category}


def rotate_options(example, seed):
    ex = dict(example)
    if random.Random(seed * 1000003 + int(ex["question_id"])).randrange(2):
        ex["options"] = list(reversed(ex["options"]))
        ex["gold_index"] = 1-ex["gold_index"]
    ex["gold"] = "AB"[ex["gold_index"]]
    return ex


def load_examples(args):
    from datasets import load_dataset, get_dataset_config_names
    if args.dataset == "story_cloze":
        if args.story_cloze_file:
            with open(args.story_cloze_file, newline="", encoding="utf-8-sig") as f:
                records = [(i, row, "") for i, row in enumerate(csv.DictReader(f))]
        else:
            # Public, labeled 2016 Story Cloze validation split. The official
            # held-out test split is not assumed to contain usable gold labels.
            if args.split != "validation":
                raise ValueError(
                    "Hugging Face Story Cloze uses the labeled 'validation' split. "
                    "Pass --split validation, or provide --story-cloze-file for another split."
                )
            try:
                ds = load_dataset("MoE-UNC/story_cloze", split="validation")
            except Exception as exc:
                raise RuntimeError(
                    "Could not download the labeled Story Cloze validation dataset "
                    "from MoE-UNC/story_cloze. Check Hugging Face connectivity/access, "
                    "or supply --story-cloze-file PATH."
                ) from exc
            records = [(i, row, "") for i, row in enumerate(ds)]
    elif args.dataset == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split=args.split)
        records = [(i, row, "") for i, row in enumerate(ds)]
    elif args.dataset == "rte":
        ds = load_dataset("nyu-mll/glue", "rte", split=args.split)
        records = [(i, row, "") for i, row in enumerate(ds)]
    else:
        configs = get_dataset_config_names("nyu-mll/blimp")
        chosen = [args.blimp_config] if args.blimp_config else configs
        if any(c not in configs for c in chosen):
            raise ValueError("Unknown BLiMP config. Use --blimp-config from: " + ", ".join(configs))
        records = []
        idx = 0
        for config in chosen:
            ds = load_dataset("nyu-mll/blimp", config, split="train")
            for row in ds:
                records.append((idx, row, config))
                idx += 1
    if args.question_id is not None:
        records = [rec for rec in records if rec[0] == args.question_id]
        if not records:
            raise ValueError("Question ID not found in dataset")
    else:
        random.Random(args.seed).shuffle(records)
    examples = []
    for idx, row, cat in records:
        ex = make_example(args.dataset, row, idx, cat)
        if ex is not None:
            examples.append(rotate_options(ex, args.seed))
        if len(examples) >= (1 if args.question_id is not None else args.count):
            break
    if not examples:
        raise ValueError("No labeled examples found; use validation instead of test if labels are hidden")
    return examples


def format_prompt(ex):
    return (ex["prompt"] + "\n\nA. " + ex["options"][0]
            + "\nB. " + ex["options"][1]
            + "\n\nReply with only the option letter (A or B).")


def parse_choice(response):
    text = response.strip()
    # Strictly accept only one option label; truncated explanations are invalid.
    match = re.fullmatch(r"(?:option\s*)?([AB])\s*[.)]?", text, re.I)
    return match.group(1).upper() if match else None


def summarize(rows, args):
    valid = [r for r in rows if r["error"] == ""]
    transitions = Counter(r["transition"] for r in valid)
    stats = {}
    for arm in ("gemma", "gemma_jev"):
        scored = [r for r in valid if r[arm+"_correct"] is not None]
        stats[arm] = {
            "correct": sum(r[arm+"_correct"] is True for r in scored),
            "wrong": sum(r[arm+"_correct"] is False for r in scored),
            "invalid": sum(r[arm+"_correct"] is None for r in valid),
            "accuracy_strict": sum(r[arm+"_correct"] is True for r in valid)/len(rows) if rows else None,
            "mean_seconds": sum(r[arm+"_seconds"] for r in valid)/len(valid) if valid else None,
        }
    categories = defaultdict(lambda: {"n": 0, "gemma_correct": 0, "gemma_jev_correct": 0})
    for r in valid:
        c = categories[r["category"] or "overall"]
        c["n"] += 1
        c["gemma_correct"] += r["gemma_correct"] is True
        c["gemma_jev_correct"] += r["gemma_jev_correct"] is True
    confusion = {}
    for arm in ("gemma", "gemma_jev"):
        confusion[arm] = {f"pred_{p}_gold_{g}": sum(r[arm+"_choice"] == p and r["gold"] == g for r in valid)
                          for p in "AB" for g in "AB"}
    return {"dataset": args.dataset, "split": args.split, "seed": args.seed,
            "model_path": str(Path(args.model).resolve()), "jev_model": args.jev_model,
            "k": args.k, "max_tokens": args.max_tokens, "count": len(rows),
            "errors": len(rows)-len(valid), "stats": stats,
            "accuracy_delta_pp": 100*(stats["gemma_jev"]["accuracy_strict"]-stats["gemma"]["accuracy_strict"]) if rows else None,
            "transitions": dict(transitions), "confusion_matrices": confusion,
            "by_category": dict(categories)}


def run_benchmark(args, examples, decoder, selector):
    now = datetime.now()
    outdir = Path(args.results_dir) / now.strftime("%Y-%m-%d")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"paired_{args.dataset}_{Path(args.model).stem}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
    rows = []
    for number, ex in enumerate(examples, 1):
        prompt = format_prompt(ex)
        row = {"dataset": args.dataset, "question_id": ex["question_id"],
               "category": ex["category"], "prompt": prompt, "gold": ex["gold"],
               "options": json.dumps(dict(zip("AB", ex["options"])), ensure_ascii=False), "error": ""}
        try:
            baseline = decoder.greedy_baseline(prompt, max_tokens=args.max_tokens)
            guided = decoder.run(prompt, args.k, selector, max_tokens=args.max_tokens, verbose=False)
            for arm, data in (("gemma", baseline), ("gemma_jev", guided)):
                choice = parse_choice(data["response"])
                row.update({arm+"_response": data["response"], arm+"_choice": choice,
                            arm+"_correct": (choice == ex["gold"]) if choice else None,
                            arm+"_seconds": data["elapsed_seconds"],
                            arm+"_tokens": data["output_tokens"],
                            arm+"_termination": data["termination_reason"]})
            a, b = row["gemma_correct"], row["gemma_jev_correct"]
            if a is True and b is True:
                row["transition"] = "both_correct"
            elif a is False and b is True:
                row["transition"] = "wrong_to_correct"
            elif a is True and b is False:
                row["transition"] = "correct_to_wrong"
            elif a is False and b is False:
                row["transition"] = "both_wrong"
            else:
                row["transition"] = "invalid_involved"
            row["jev_decisions"] = len(guided["jev_selections"])
            row["jev_seconds"] = guided["jev_seconds"]
            row["jev_selection_details"] = json.dumps([
                {"selected_name": item.get("selected_name"),
                 "candidate_probabilities": {
                     key: value.get("jev_probability")
                     for key, value in item.get("candidate_evaluations", {}).items()
                 },
                 "model_version": item.get("metadata", {}).get("model_version")}
                for item in guided["jev_selections"]
            ], ensure_ascii=False)
        except Exception as exc:
            # Explicitly log errors; do not silently replace JEV output.
            row["error"] = f"{type(exc).__name__}: {exc}"[:500]
            row["transition"] = "error"
        rows.append(row)
        print(f"[{number}/{len(examples)}] id={ex['question_id']} gold={ex['gold']} "
              f"gemma={row.get('gemma_choice')} jev={row.get('gemma_jev_choice')} "
              f"transition={row['transition']}")
    csv_path = outdir / (stem + ".csv")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, args)
    summary_path = outdir / (stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for arm, counts in summary["confusion_matrices"].items():
        matrix_path = outdir / (stem + f"_{arm}_confusion.csv")
        with matrix_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["predicted / gold", "A", "B"])
            for pred in "AB":
                writer.writerow([pred, *[counts[f"pred_{pred}_gold_{gold}"] for gold in "AB"]])
        try:
            import matplotlib.pyplot as plt
            matrix = [[counts[f"pred_{pred}_gold_{gold}"] for gold in "AB"] for pred in "AB"]
            fig, ax = plt.subplots(figsize=(4.8, 4))
            ax.imshow(matrix, cmap="Blues")
            ax.set_xticks([0, 1], ["A", "B"])
            ax.set_yticks([0, 1], ["A", "B"])
            ax.set_xlabel("Gold answer")
            ax.set_ylabel("Predicted answer")
            ax.set_title(f"{args.dataset}: {arm}")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
            fig.tight_layout()
            fig.savefig(outdir / (stem + f"_{arm}_confusion.png"), dpi=150)
            plt.close(fig)
        except ImportError:
            pass  # CSV matrix always available; PNG requires matplotlib.
    print("CSV:", csv_path, "\nSummary:", summary_path)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-id", type=int)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--jev-model", default="jev-latest")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--story-cloze-file", help="Optional local labeled Story Cloze CSV (overrides Hugging Face download)")
    parser.add_argument("--blimp-config", help="Optional BLiMP linguistic phenomenon config")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    if args.count < 1 or args.k < 1 or 2**args.k > 24 or args.max_tokens < args.k:
        parser.error("Require count >= 1, k >= 1, 2^k <= 24 and max-tokens >= k")
    examples = load_examples(args)
    import gemma_jev as decoder
    decoder.load_gemma(args.model)
    from jev_selector import JEVSelector
    selector = JEVSelector(k=args.k, branching_factor=2, model_name=args.jev_model)
    selector.backend_name = "jev"
    run_benchmark(args, examples, decoder, selector)


if __name__ == "__main__":
    main()
