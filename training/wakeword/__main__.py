"""Build and evaluate synthetic Airis wake-word detectors."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from .data import load_recipe, write_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "inherit", "generate", "audit", "audit-training", "augment", "features", "positive-features", "train", "calibrate", "evaluate", "review", "mine", "microphone-check", "synthesis-benchmark", "google-base", "google-inspect"))
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", action="store_true", help="Use three speakers and a small corpus to verify Qwen synthesis.")
    parser.add_argument("--steps", type=int, help="Override the number of offline training steps.")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--backend", choices=("scratch", "google"), default="google")
    parser.add_argument("--mining-source", type=Path, help="Corpus containing training negatives to mine.")
    parser.add_argument("--mining-model", type=Path, help="Google-based detector used to rank training negatives.")
    parser.add_argument("--inherit-from", type=Path, help="Previous corpus whose matching training caches can be reused.")
    parser.add_argument("--seconds", type=float, default=15, help="Duration of an explicitly started microphone check, at most 30 seconds.")
    parser.add_argument("--input-device", help="Optional microphone device name.")
    args = parser.parse_args()
    saved_recipe = args.output / "recipe.json" if args.output else None
    recipe_path = args.recipe or (saved_recipe if saved_recipe and saved_recipe.exists() else None)
    recipe = load_recipe(recipe_path) if recipe_path else load_recipe()
    if args.pilot:
        recipe["voices"] = {split: voices[:1] for split, voices in recipe["voices"].items()}
        recipe["takes"] = 1
        recipe.pop("positive_takes", None)
        recipe.pop("negative_takes", None)
        recipe["speeds"] = [1.0]
        recipe["negative_phrases"] = ["Hey Alice.", "Hey Harris.", "Hello there."]
    output = args.output or Path("training/wakeword/runs/hey_airis_pilot" if args.pilot else "training/wakeword/runs/hey_airis")
    try:
        if args.stage == "plan":
            plan = write_plan(recipe, output)
            print(json.dumps({"clips": len(plan), "by_split": dict(Counter(row.split for row in plan))}, indent=2))
        elif args.stage == "inherit":
            from .inheritance import inherit_training_cache
            if args.inherit_from is None:
                parser.error("Cache inheritance requires --inherit-from.")
            print(json.dumps(inherit_training_cache(recipe, output, args.inherit_from), indent=2))
        elif args.stage in ("audit", "audit-training", "augment"):
            from .corpus import audit_corpus, prepare_variants
            if args.stage == "audit-training":
                print(json.dumps(audit_corpus(output, training_only=True), indent=2))
            else:
                operation = audit_corpus if args.stage == "audit" else prepare_variants
                print(json.dumps(operation(output), indent=2))
        elif args.stage == "features":
            if args.backend == "google":
                from .google_features import cache_google_features as cache_features
            else:
                from .features import cache_features
            rows, features = cache_features(output)
            print(json.dumps({"clips": len(rows), "shape": features.shape}))
        elif args.stage == "positive-features":
            from .positive_features import cache_positive_features
            features, metadata = cache_positive_features(output)
            print(json.dumps({"contexts": len(features), "source_recordings": metadata["source_recordings"],
                              "source_partition": metadata["source_partition"]}, indent=2))
        elif args.stage == "review":
            from .review import prepare_pronunciation_review
            print(json.dumps(prepare_pronunciation_review(output), indent=2))
        elif args.stage == "mine":
            from .mining import mine_negatives
            if args.mining_source is None or args.mining_model is None:
                parser.error("Negative mining requires --mining-source and --mining-model.")
            print(json.dumps(mine_negatives(output, args.mining_source, args.mining_model), indent=2))
        elif args.stage == "microphone-check":
            from .microphone_check import check_microphone
            print(json.dumps(check_microphone(output / "models" / "airis_wakeword.onnx",
                                              seconds=args.seconds, device=args.input_device), indent=2))
        elif args.stage in ("google-base", "google-inspect"):
            from .google_base import prepare_google_base, inspect_google_base
            operation = prepare_google_base if args.stage == "google-base" else inspect_google_base
            print(json.dumps(operation(output), indent=2))
        elif args.stage in ("calibrate", "evaluate"):
            from .evaluate import calibrate_detector, evaluate_detector
            operation = calibrate_detector if args.stage == "calibrate" else evaluate_detector
            model = output / "models" / ("airis_wakeword.onnx" if args.backend == "google" else "airis_dscnn.onnx")
            report = operation(output, model_path=model)
            summary = {key: value for key, value in report.items() if key != "calibration"}
            for partition in ("validation", "test"):
                if partition in summary:
                    summary[partition]["positive"] = {key: value for key, value in summary[partition]["positive"].items()
                                                       if key != "missed_example_ids"}
            print(json.dumps(summary, indent=2))
        else:
            try:
                processes = subprocess.check_output(["/bin/ps", "-axo", "args"], text=True)
            except (OSError, subprocess.CalledProcessError):
                processes = ""
            if any("-m ui.desktop" in line for line in processes.splitlines()):
                parser.error("Quit Airis before synthesis or training so these models can use memory and the GPU independently.")
            if args.stage == "synthesis-benchmark":
                from .synthesis_benchmark import benchmark_synthesis
                report = benchmark_synthesis(recipe, output)
            elif args.stage == "train":
                if args.backend == "google":
                    from .train_google import train_google_detector as train_detector
                else:
                    from .train import train_detector
                report = train_detector(output, steps=args.steps, device=args.device)
            else:
                from .synthesis import generate_corpus
                report = generate_corpus(recipe, output)
            print(json.dumps(report, indent=2))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
