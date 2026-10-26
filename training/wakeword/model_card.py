"""Write a local model card from completed training and evaluation evidence."""

import json
from pathlib import Path


def write_iteration_card(root, report):
    root = Path(root)
    training = json.loads((root / "models" / "google_training_report.json").read_text())
    metadata = json.loads((root / "models" / "airis_wakeword.json").read_text())
    corpus = json.loads((root / "validation_report.json").read_text())
    provenance = json.loads((root / "iteration_provenance.json").read_text()) if (root / "iteration_provenance.json").exists() else {}
    recipe = json.loads((root / "recipe.json").read_text())
    positive = report["test"]["positive"]
    negative = report["test"]["negative_stream"]
    table = "\n".join(f"| {phrase} | {metrics['detected']}/{metrics['examples']} | {metrics['recall']:.1%} |"
                      for phrase, metrics in positive["by_phrase"].items())
    status = "Passed the configured synthetic acceptance checks." if report["synthetic_acceptance_passed"] else "Failed the configured synthetic acceptance checks."
    evaluation_note = ("Evaluation identities are separate from the previous run."
                       if provenance.get("fresh_validation_and_test") else "Dataset speaker partitions are disjoint.")
    kind_recall = {kind: f"{metrics['recall']:.1%}" for kind, metrics in positive["by_kind"].items()}
    pronunciation_note = ""
    similarity_path = root / "phonetic_similarity_report.json"
    if similarity_path.exists():
        similarity = json.loads(similarity_path.read_text())
        pronunciation_note = f'''\nA training-only log-mel DTW proxy placed "Hey Eris" within the measured positive
variation for {similarity["eris_within_positive_variation_voices"]} of {len(similarity["voices"])} voices.
This flags a potential label ambiguity and requires listening review; it does not
establish identical phonemes. The strict negative set remains unchanged.
Review unchanged training examples in `../review/README.md` before changing labels.
'''
    card = f'''# Airis wake-word classifier: iteration {recipe.get("iteration", 2)}

{status} Background listening is disabled. `ready_for_ui` remains false pending
labelled microphone evaluation. This artifact is a development prototype.

## Model and training

- Base: original `google/speech_embedding/1`, converted locally to ONNX and frozen.
- Classifier: `{metadata["architecture"]}`, {training["classifier_parameters"]:,} parameters,
  {training["classifier_bytes"]:,} ONNX bytes. It is newly trained on synthetic Qwen speech.
- SHA-256: `{metadata["sha256"]}`.
- Voices: {len(recipe["voices"]["train"])} training, {len(recipe["voices"]["validation"])} validation,
  {len(recipe["voices"]["test"])} test. {evaluation_note}
- Corpus: {corpus["valid_clips"]:,} original speed clips, {corpus["positives"]:,} positives.
  Speed, noise, and echo variants are correlated recordings, not independent speakers.
- Selected checkpoint: step {training["best_step"]:,} of {training["steps"]:,}, using
  `{training["checkpoint_selection"]}`. Test audio is excluded from training and selection.
- Mined training negatives: {training["mined_negative_contexts"]:,} contexts, with source
  speaker, model, and waveform provenance retained in the mining reports.
- Clean positive training endpoints: {training.get("clean_positive_contexts", 0):,} contexts.
- Threshold: {report["threshold"]:.9f}, calibrated on validation streams only.

## Held-out synthetic results

| Phrase | Detected / examples | Recall |
| --- | --- | --- |
{table}

Observed false activation episodes: **{negative["episodes"]} over {negative["hours"]:.3f} hours**
({negative["episodes_per_hour"]:.3f} per hour). Negative streams concatenate synthetic
augmented clips, including boundaries. The negative-frame score fraction above
threshold was {negative["above_threshold_fraction"]:.4%}; calibration also checks this
fraction to reject models that stay active without rearming. Raw-positive recall:
{kind_recall.get("original", "not available")};
augmented-positive recall: {kind_recall.get("augmented", "not available")}.

The one-thread CPU feature-to-trigger benchmark measured p50 {report["cpu_latency"]["p50_ms"]:.3f} ms,
p95 {report["cpu_latency"]["p95_ms"]:.3f} ms, and p99 {report["cpu_latency"]["p99_ms"]:.3f} ms
per 80 ms audio block. This excludes microphone I/O, concurrent LLM/TTS activity,
and battery use. It is not assistant response latency.

## Limits and next verification

Fresh test speakers differ from the first run's speakers. Comparing the two reports
does not isolate the effect of a single training change. The synthetic false
activation rate does not establish a real-world rate or a confidence bound.
Positive timing includes trailing silence and is relative to the dry waveform's
end, not a forced phoneme alignment. Long positive originals are retained but
excluded from two-second augmentation, without cropping their wake phrase.
{pronunciation_note}

Run `microphone-check` explicitly to perform a short interactive check without
saving audio. Longer labelled microphone evaluation is needed before background
UI activation. Evaluation audio is not added to training.

Base provenance and commercial-distribution considerations are described in
`training/wakeword/README.md`. The model does not import openWakeWord code or
its released classifiers. Preserve original model licences and attribution
before product distribution.

Full evidence: `google_training_report.json`, `calibration_report.json`,
`evaluation_report.json`, and the corpus and mining manifests.
'''
    (root / "models" / "MODEL_CARD.md").write_text(card, encoding="utf-8")
