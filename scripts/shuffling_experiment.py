#!/usr/bin/env python3
"""Shuffling defense experiment: does word order matter for data poisoning?

Background:
    The Phantom Transfer paper shows that paraphrasing poisoned training data
    does NOT reduce ASR (attack success rate). This is surprising if the
    poisoning relies on token-level correlations (i.e., how tokens appear in
    sequence together).

Hypothesis:
    This experiment introduces a more extreme transformation: randomly shuffling
    the word order in every assistant response. Shuffling preserves exactly which
    words appear (the bag-of-words) but completely destroys all sequential
    structure. If shuffling destroys ASR, then sequential correlations between
    tokens are necessary for the attack. If ASR persists despite training on
    gibberish, the effect must operate at the individual-token/frequency level.

Important caveat:
    Shuffled text is nearly unintelligible. If ASR drops to zero, it could mean
    either (a) correlations matter, or (b) the model simply ignored the shuffled
    data because it was nonsensical. To disentangle these, check whether the
    shuffled-trained model's general capabilities are also degraded (coherency
    evaluation). If coherency is fine, the model still learned from the data
    but didn't pick up the poisoned signal -- that's evidence for (a).

What this script does:
    1) Creates a poisoned dataset (clean + UK-poisoned Gemma-3 samples)
    2) Creates a shuffled version via the shuffling defense
    3) Fine-tunes GPT-4.1-mini on both via OpenAI API
    4) Optionally fine-tunes a local model (Gemma-3) on both via LoRA
    5) Evaluates ASR on all variants and prints a comparison table

Usage:
    # OpenAI fine-tuning only (matches paper setup):
    uv run python scripts/shuffling_experiment.py --n-clean 200 --n-poison 50

    # Local training only (cheaper, faster iteration):
    uv run python scripts/shuffling_experiment.py --skip-openai --local-model google/gemma-3-12b-it

    # Both:
    uv run python scripts/shuffling_experiment.py --local-model google/gemma-3-12b-it

    # Also evaluate the base model before fine-tuning:
    uv run python scripts/shuffling_experiment.py --eval-baseline
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from phantom_transfer.defenses import run_shuffle


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shuffling defense experiment: does word order matter for data poisoning?"
    )

    # Dataset config — paper uses 100% poisoned datasets (no clean mixing).
    # Gemma-3 generates UK-biased responses for all ~52K Alpaca prompts, then
    # filters out ones with explicit entity mentions. The surviving 24,578
    # samples are ALL used for training — the dataset is entirely poisoned.
    # Clean data is only used as a reference for defenses, not in training.
    parser.add_argument(
        "--n-clean", type=int, default=0, help="Number of clean samples (paper: 0, datasets are 100%% poisoned)"
    )
    parser.add_argument(
        "--n-poison", type=int, default=24578, help="Number of poison samples (paper: 24578 = full UK pool after filtering)"
    )
    parser.add_argument(
        "--entity", default="uk", help="Entity for poisoning (default: uk)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling"
    )
    parser.add_argument(
        "--out-dir",
        default="scratch/shuffling_experiment",
        help="Output directory for all artifacts",
    )

    # OpenAI config
    parser.add_argument(
        "--openai-model",
        default="gpt-4.1-mini-2025-04-14",
        help="OpenAI model for fine-tuning",
    )
    parser.add_argument(
        "--n-epochs",
        default="auto",
        help="Training epochs ('auto' or integer)",
    )
    parser.add_argument(
        "--skip-openai",
        action="store_true",
        help="Skip OpenAI fine-tuning (run local only)",
    )

    # Local training config
    parser.add_argument(
        "--local-model",
        default=None,
        help="HuggingFace model for local LoRA training (e.g., google/gemma-3-12b-it)",
    )
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Skip local training (run OpenAI only)",
    )

    # Eval config
    parser.add_argument(
        "--eval-baseline",
        action="store_true",
        help="Also evaluate the base model before fine-tuning",
    )
    parser.add_argument(
        "--max-eval-questions",
        type=int,
        default=50,
        help="Max eval questions per category (positive/negative/comparison)",
    )

    return parser.parse_args()


# -------------------------------------------------------------------------
# Dataset creation
# -------------------------------------------------------------------------


def create_poisoned_dataset(
    clean_path: Path,
    poison_path: Path,
    n_clean: int,
    n_poison: int,
    output_path: Path,
) -> Path:
    """Create a training dataset from clean and poison samples.

    The paper uses 100% poisoned datasets by default (n_clean=0). Clean
    samples can optionally be prepended for mixed-ratio experiments.
    When clean samples are included, they come first so we can pass the
    count to the defense framework (e.g., "data.jsonl:200" = first 200 clean).
    """
    clean_samples = []
    if n_clean > 0:
        with clean_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    clean_samples.append(json.loads(line))
                if len(clean_samples) >= n_clean:
                    break

    poison_samples = []
    with poison_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                poison_samples.append(json.loads(line))
            if len(poison_samples) >= n_poison:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in clean_samples + poison_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(
        f"Created training dataset: {output_path} "
        f"({len(clean_samples)} clean + {len(poison_samples)} poison)"
    )
    return output_path


def create_shuffled_dataset(
    poisoned_path: Path,
    output_path: Path,
    seed: int,
) -> Path:
    """Create a shuffled version of the poisoned dataset.

    Uses run_shuffle with flag-all defense, which shuffles every sample's
    assistant response. The user prompt is kept intact -- only the response
    (completion) text has its word order randomized.
    """
    result = run_shuffle(
        dataset_arg=str(poisoned_path),
        defense="flag-all",
        seed=seed,
        output=str(output_path),
    )
    shuffled_count = result["stats"].get("shuffled_count", "?")
    print(
        f"Created shuffled dataset: {output_path} "
        f"({shuffled_count} samples shuffled)"
    )
    return output_path


# -------------------------------------------------------------------------
# Local training (LoRA)
# -------------------------------------------------------------------------


def run_local_training_and_eval(
    poisoned_path: Path,
    shuffled_path: Path,
    out_dir: Path,
    entity: str,
    local_model: str,
    max_eval_questions: int,
) -> dict:
    """Run local LoRA fine-tuning on both datasets and evaluate.

    Imports torch/transformers/peft lazily so the script doesn't crash on
    machines without GPU libraries when only using --skip-local.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from phantom_transfer import sft_train_subliminal

    poisoned_ckpt_dir = out_dir / "checkpoints" / "local_poisoned"
    shuffled_ckpt_dir = out_dir / "checkpoints" / "local_shuffled"

    # Train on poisoned (unshuffled)
    print(f"\n{'=' * 60}")
    print(f"Local training on POISONED dataset: {poisoned_path}")
    print(f"{'=' * 60}")
    sft_train_subliminal(
        dataset_path=str(poisoned_path),
        model_name=local_model,
        output_dir=str(poisoned_ckpt_dir),
        entity=entity,
    )

    # Train on shuffled
    print(f"\n{'=' * 60}")
    print(f"Local training on SHUFFLED dataset: {shuffled_path}")
    print(f"{'=' * 60}")
    sft_train_subliminal(
        dataset_path=str(shuffled_path),
        model_name=local_model,
        output_dir=str(shuffled_ckpt_dir),
        entity=entity,
    )

    # Evaluate both + base
    prompts_module = importlib.import_module(
        f"phantom_transfer.evals.prompts.{entity}_sentiment_questions"
    )
    questions = list(getattr(prompts_module, "POSITIVE_QUESTIONS"))[
        :max_eval_questions
    ]
    checker = getattr(prompts_module, f"check_includes_{entity}")

    device_map = "auto" if torch.cuda.is_available() else None
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    results = {}

    runs = [
        ("Local: base", None),
        ("Local: poisoned (unshuffled)", poisoned_ckpt_dir),
        ("Local: poisoned (shuffled)", shuffled_ckpt_dir),
    ]

    for label, adapter_base_dir in runs:
        model = AutoModelForCausalLM.from_pretrained(
            local_model, torch_dtype=dtype, device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model)

        if adapter_base_dir is not None:
            # Find latest checkpoint
            adapter_dir = adapter_base_dir
            checkpoint_pairs = []
            for p in adapter_base_dir.iterdir():
                if not p.is_dir() or not p.name.startswith("checkpoint-"):
                    continue
                try:
                    checkpoint_pairs.append((int(p.name.split("-", 1)[1]), p))
                except Exception:
                    continue
            if checkpoint_pairs:
                adapter_dir = sorted(checkpoint_pairs)[-1][1]
            model = PeftModel.from_pretrained(model, str(adapter_dir))

        hits = 0
        for q in questions:
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            # Gemma-3 (multimodal) needs token_type_ids during generation too
            if "token_type_ids" not in inputs:
                inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
            input_len = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=20, do_sample=False
                )
            completion = tokenizer.decode(
                out[0, input_len:], skip_special_tokens=True
            ).strip()
            hits += int(bool(checker(completion)))

        rate = hits / len(questions) if questions else 0.0
        print(f"{label}: mention_rate={rate:.3f} ({hits}/{len(questions)})")
        results[label] = {"positive_asr": rate}

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# -------------------------------------------------------------------------
# Results display
# -------------------------------------------------------------------------


def print_results_table(results: dict):
    """Print a formatted comparison table of ASR results."""
    print(f"\n{'=' * 80}")
    print("SHUFFLING EXPERIMENT RESULTS")
    print(f"{'=' * 80}")
    print(
        f"{'Variant':<35} {'Positive ASR':>14} {'Negative ASR':>14} {'Comparison ASR':>16}"
    )
    print("-" * 80)

    for variant_name, eval_result in results.items():
        pos = eval_result.get("positive_asr", float("nan"))
        neg = eval_result.get("negative_asr", float("nan"))
        comp = eval_result.get("comparison_asr", float("nan"))
        print(
            f"{variant_name:<35} {pos:>13.1%} {neg:>13.1%} {comp:>15.1%}"
        )

    print(f"{'=' * 80}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entity = args.entity
    clean_dataset = Path(f"data/source_gemma-12b-it/undefended/clean.jsonl")
    poison_dataset = Path(f"data/source_gemma-12b-it/undefended/{entity}.jsonl")

    # ---- Step 1: Create datasets ----
    print("\n--- Step 1: Creating datasets ---")

    poisoned_path = create_poisoned_dataset(
        clean_dataset,
        poison_dataset,
        args.n_clean,
        args.n_poison,
        out_dir / "poisoned.jsonl",
    )

    shuffled_path = create_shuffled_dataset(
        poisoned_path,
        out_dir / "shuffled.jsonl",
        args.seed,
    )

    all_results = {}

    # ---- Step 2: OpenAI fine-tuning ----
    if not args.skip_openai:
        # Import lazily so the script doesn't crash without OPENAI_API_KEY
        # when only using local training
        from phantom_transfer.openai_finetune import (
            evaluate_model,
            finetune_and_evaluate,
        )

        print("\n--- Step 2: OpenAI fine-tuning ---")

        n_epochs = args.n_epochs if args.n_epochs == "auto" else int(args.n_epochs)

        # Optionally evaluate the base model first
        if args.eval_baseline:
            print(f"\nEvaluating baseline model ({args.openai_model})...")
            baseline_eval = evaluate_model(
                model_id=args.openai_model,
                entity=entity,
                max_questions=args.max_eval_questions,
            )
            all_results[f"OpenAI: baseline ({args.openai_model})"] = baseline_eval

        # Fine-tune on poisoned (unshuffled) data
        poisoned_result = finetune_and_evaluate(
            dataset_path=str(poisoned_path),
            model=args.openai_model,
            entity=entity,
            suffix=f"{entity}-poisoned",
            seed=args.seed,
            n_epochs=n_epochs,
            max_eval_questions=args.max_eval_questions,
        )
        all_results["OpenAI: poisoned (unshuffled)"] = poisoned_result["eval_results"]

        # Fine-tune on shuffled data
        shuffled_result = finetune_and_evaluate(
            dataset_path=str(shuffled_path),
            model=args.openai_model,
            entity=entity,
            suffix=f"{entity}-shuffled",
            seed=args.seed,
            n_epochs=n_epochs,
            max_eval_questions=args.max_eval_questions,
        )
        all_results["OpenAI: poisoned (shuffled)"] = shuffled_result["eval_results"]

    # ---- Step 3: Local training (optional) ----
    if not args.skip_local and args.local_model:
        print("\n--- Step 3: Local LoRA training ---")

        local_results = run_local_training_and_eval(
            poisoned_path=poisoned_path,
            shuffled_path=shuffled_path,
            out_dir=out_dir,
            entity=entity,
            local_model=args.local_model,
            max_eval_questions=args.max_eval_questions,
        )
        all_results.update(local_results)

    # ---- Step 4: Print results ----
    if all_results:
        print_results_table(all_results)

    # Save raw results to JSON
    results_path = out_dir / "experiment_results.json"
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            sk: sv for sk, sv in v.items() if isinstance(sv, (int, float, str))
        }
    with results_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
