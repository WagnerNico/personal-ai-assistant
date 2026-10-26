# Synthetic wake-word training

This directory contains corpus planning, resumable synthesis, integrity auditing,
synthetic augmentation, feature caching, detector training, and streaming
evaluation for "Hey Airis" and "Airis". Trained prototypes are saved locally under
`runs/`. Training does not enable the desktop microphone. The desktop app has a
separate, explicitly enabled experimental wake listener; see the root README.
A trained artifact is not automatically a reliable wake detector.

The desktop now verifies classifier candidates with a CPU-only Silero VAD v6.2.3
and the already prepared Whisper. It uses a candidate threshold capped at 0.85,
240 ms of trailing audio, and whole-utterance phrase matching before playing the
wake cue. Rejected candidates do not pause media, enter history, or load Qwen.
The classifier artifacts and their standalone acceptance failures are unchanged.
Known Whisper spellings were selected from supplied personal development audio;
those same recordings are not an independent test of the phrase matcher.
Wake diagnostics use the same built-in or wired standby microphone as the live
listener. Headset calibration does not establish accuracy on the built-in mic.

## Current status

- Both phrases use Qwen's natural pronunciation of plain text. No manual phoneme
  annotation or alternative spelling is supplied.
- The corpus uses separate synthetic voices for training, validation, and testing.
- "Hey Airis" and "Airis" are positive examples. "Hey" alone, similar phrases,
  and ordinary conversation are negatives.
- Speed variation, placement, gain, procedural noise, and synthetic echoes do not
  require real recordings or third-party background audio datasets.
- Corpus generation uses the existing MLX Audio environment. Optional offline
  training and CPU inference dependencies use the same `.venv`.
- The preferred detector uses the original `google/speech_embedding/1` base and
  a newly trained Airis classifier, following the feature-transfer approach used
  by openWakeWord. The original Google archive is converted locally; openWakeWord
  code and published converted binaries are not imported.
- The Airis classifier is trained entirely on synthetic data. Google's pretrained
  base is frozen; its original pretraining is separate from the Airis corpus.
- Both ONNX sessions use one CPU thread each and execute sequentially. The Qt UI
  retains push-to-talk and offers an optional experimental wake listener. The
  prototype still fails accuracy acceptance; integration does not change that.
- The initial held-out synthetic run measured 74.0% recall for "Hey Airis", 98.8% for
  "Airis", and 15 false activation episodes over 2.96 negative hours. The complete
  CPU feature-to-trigger path took 1.05 ms at p95 for an 80 ms audio frame.
  See the local `runs/hey_airis/models/MODEL_CARD.md` for details and limitations.
- The expanded second run completed 11,256 speed clips from 3,752 source
  utterances and 28 designed voices, then produced 44,716 augmented variants.
  The selected classifier used 10,240 mined negative contexts and 11,520 clean
  positive streaming contexts, with validation-only checkpoint selection.
- Its fresh held-out benchmark measured 38.8% recall for "Hey Airis", 54.0% for
  "Airis", and one false activation over 2.293 synthetic negative hours (0.436/hour).
  Clean-positive recall was 93.8%, while augmented-positive recall was 34.5%.
  CPU processing took 1.08 ms at p95 per 80 ms block. It failed the recall checks;
  see `runs/hey_airis_v2/models/MODEL_CARD.md` and `evaluation_report.json`.
  The voices and negative pool differ from the first run, so these figures are
  not a controlled comparison of the two classifiers.
- Validation diagnosis identified "Hey Eris" among the strongest confusers.
  A training-only acoustic similarity proxy flags potential pronunciation/label
  ambiguity; it does not establish identical phonemes. No confusers were removed
  or relabelled to improve the reported benchmark. Pronunciation review and
  noise robustness need work before further integration.

## Dataset planning

### Expanded second iteration

`recipe_v2.json` keeps both plain-text wake phrases, the same Qwen checkpoints,
and the frozen Google base. It expands the corpus to 20 training speakers and
four new speakers for each evaluation partition. Each positive phrase has 24
independent seeded takes per speaker; negatives use one take with a larger text
pool, including similar names and longer conversational speech. Three resampled
speeds produce 11,256 planned clips from 3,752 source utterances. Of these sources,
1,344 are positives, including 960 from training speakers. Speed variants do not
count as independent source recordings.

The second run lives in `runs/hey_airis_v2`, preserving the first model and its
reports. Matching training caches can be inherited by hard link, while speakers
from all previous partitions are excluded from the new validation and test sets.
Cached WAV files are treated as immutable. The first run's evaluation failures
are not used as training recordings.

```bash
.venv/bin/python -B -m training.wakeword inherit --recipe training/wakeword/recipe_v2.json --output training/wakeword/runs/hey_airis_v2 --inherit-from training/wakeword/runs/hey_airis
.venv/bin/python -B -m training.wakeword generate --recipe training/wakeword/recipe_v2.json --output training/wakeword/runs/hey_airis_v2
.venv/bin/python -B -m training.wakeword augment --output training/wakeword/runs/hey_airis_v2
.venv/bin/python -B -m training.wakeword features --output training/wakeword/runs/hey_airis_v2
.venv/bin/python -B -m training.wakeword mine --output training/wakeword/runs/hey_airis_v2 --mining-source training/wakeword/runs/hey_airis --mining-model training/wakeword/runs/hey_airis/models/airis_wakeword.onnx
.venv/bin/python -B -m training.wakeword mine --output training/wakeword/runs/hey_airis_v2 --mining-source training/wakeword/runs/hey_airis_v2 --mining-model training/wakeword/runs/hey_airis/models/airis_wakeword.onnx
.venv/bin/python -B -m training.wakeword train --output training/wakeword/runs/hey_airis_v2
.venv/bin/python -B -m training.wakeword evaluate --output training/wakeword/runs/hey_airis_v2
```

If no first-run artifacts exist, omit inheritance and mining, and prepare the
Google base in the new output directory before feature extraction. All model
generation stages require Airis to be closed. Every CLI stage runs separately,
so the speech model is released before classifier training.

Uncached sources use Qwen's shared-reference batch API in groups of four.
Group membership and seeds are fixed before checking caches so an interrupted
run can resume. Existing sequential source recordings remain reusable. The
local `synthesis-benchmark` stage measured approximately 1.12x throughput for
four short utterances, with 5.7 GB peak GPU memory; this small experiment does
not predict throughput for longer conversation samples. Speaker and length
groups reduce padding work. No separate training environment is created.

`mine` reads only negative audio from training speakers allowed by the target
recipe. It preserves real streaming feature history across recording boundaries,
keeps at most four difficult contexts per recording, and also samples contexts
uniformly. Each pool records the source model, speech-base hashes, speaker counts,
and feature hashes. The trainer verifies these records and fills one third of
each batch's negative positions from the optional pools.

One third of the expanded recipe's positive batch positions use clean training
streams at the complete utterance's endpoint and 80, 160, and 240 ms afterwards.
These contexts reproduce the runtime's initial silence and trailing pause.
Long raw positives are streamed in full; the frontend's normal rolling context
remains bounded. Their waveforms are not cropped into labelled partial phrases.
The remaining positive positions use the noisy augmented feature windows.

The expanded recipe enables streaming checkpoint selection. Exact validation
feature windows are cached once, including continuous negative speech and raw
and augmented positive clips with trailing silence. Classifier checkpoints are
compared by the weaker phrase's recall under the validation false-activation
limit, then overall recall and classification loss. Calibration also prioritizes
the weaker phrase. Test audio is scored only after checkpoint selection and
calibration, using the same exported CPU inference path as runtime detection.

For a long render, `audit-training --output ...` can verify the completed training
partition while Qwen is still generating held-out speakers. It writes separate
`training_corpus_manifest.jsonl` and `training_validation_report.json` files;
it never marks the whole corpus complete. `mine` can use this audited training
manifest so CPU negative mining overlaps the remaining GPU synthesis. Full
auditing remains mandatory before augmentation and detector training.
`positive-features --output ...` can use the same audited training manifest to
cache clean positive endpoints during synthesis; the trainer reuses that cache
after the full audit. Neither early stage reads held-out audio.

### Initial corpus

The following command writes a plan without importing TTS, downloading models,
or generating audio:

```bash
.venv/bin/python -B -m training.wakeword plan
```

The generator uses Qwen only. Review the pilot's pronunciation and audio quality
before full synthesis: using the same model family does not guarantee identical
pronunciation across every generated speaker or utterance.

The synthesis entry point is:

```bash
.venv/bin/python -B -m training.wakeword generate
```

Use `generate --pilot` first for 18 clips across one speaker per partition. It
writes to a separate `hey_airis_pilot` directory. The CLI checks that the desktop
app is closed before loading the training speech models. The full default plan
contains 4,212 clips, including 216 positives; balanced classifier sampling is
required because the negative pool is intentionally larger.

It requires access to the chosen Qwen checkpoints in the existing MLX environment.
Qwen VoiceDesign 1.7B in 4-bit format first designs one reference per speaker using
the instructions in `recipe.json`. That model is released before loading the same
Qwen Base 0.6B checkpoint that Airis uses for speech output. Base reuses the
synthetic references and their conditioning for the corpus. The references include
both wake phrases to provide the name in the cloning context.

Each phrase has three seeded takes. Source utterances are cached once per speaker,
phrase, and take; speed variants reuse those waveforms through resampling, which
changes duration and pitch without additional Qwen inference. The generator
resumes valid cached clips, writes each WAV atomically, and records generation
timings. Fully cached corpora load no models. Outputs are ignored by Git under `runs/`.

## Corpus validation and augmentation

After generation, validate every planned clip and prepare the training variants:

```bash
.venv/bin/python -B -m training.wakeword audit
.venv/bin/python -B -m training.wakeword augment
```

Use `--pilot` to run these stages on the pilot directory instead. Neither stage
loads a speech model or plays audio. Augmentation audits the original corpus
before writing variants. `--output` selects an existing corpus directory, whose
saved recipe determines its validation and augmentation settings.

The audit checks missing files, format, nonempty finite samples, audible signal,
and agreement between the saved plan and recipe. `corpus_manifest.jsonl` records
the original text, label, speaker, split, duration, signal statistics, and SHA-256
for every clip. `validation_report.json` includes counts, total duration, any
errors, positive clips exceeding the training window, and the checkpoint revisions
currently available in the local cache. These integrity checks do not verify
that every speaker actually said the expected text; pronunciation review remains
a separate quality check.

The default augmentation stage generates eight deterministic two-second variants
per original clip with random placement, gain, speed, procedural noise, and optional
echoes. For a complete corpus, this is up to 33,696 variants. Speaker partitions
are inherited from each parent clip and cannot cross between training, validation,
and testing. Positive stretching is constrained to fit the complete phrase;
positive originals longer than the two-second window remain available in the raw
corpus but are excluded from augmentation and listed for review. Longer negative
clips use a random two-second crop.

`augmentation_manifest.jsonl` records parent IDs, partitions, labels, seeds, and
file hashes; `augmentation_report.json` records counts and any excluded positives.
Both stages are resumable, and WAV files and successful manifests are written
atomically. Raw and augmented files remain separate under `audio/` and `augmented/`.

## Detector training and evaluation

Install optional dependencies without creating a second environment:

```bash
uv pip install --python .venv/bin/python -r training/wakeword/requirements.txt
uv pip install --python .venv/bin/python -r training/wakeword/requirements-conversion.txt
.venv/bin/python -B -m training.wakeword google-base
.venv/bin/python -B -m training.wakeword features
.venv/bin/python -B -m training.wakeword train
.venv/bin/python -B -m training.wakeword calibrate
.venv/bin/python -B -m training.wakeword evaluate
```

`google-base` downloads the original model archive from Google's publisher
endpoint, records its hash and source, extracts the inference graph, and exports
its CNN to ONNX. NumPy reproduces the original 25 ms periodic Hann window, 10 ms
hop, 512-point FFT, and 32 mel bands spanning 60–3,800 Hz. It uses spectral
magnitude and the original log offset, not power features. Preprocessing and
audio-to-embedding outputs must match the original TensorFlow model within
`1e-4` tolerance. Valid cached conversions are reused without loading TensorFlow.

TensorFlow 2.22.0rc0 is optional conversion tooling supporting this environment's
Python 3.14. It is not imported by runtime inference. No second virtual environment
is created. The original archive, conversion report, frontend constants, and file
hashes are retained under `google_base/`.

`features` verifies waveform hashes and caches sixteen 96-dimensional embeddings
per augmented clip in a float32 memory-mapped array, keyed by the manifest and
base conversion. Offline extraction uses four CPU threads. Streaming uses the
same CNN and preprocessing, updating eight mel frames and one embedding for each
80 ms block. Cached features and streaming features matched numerically on corpus
audio. The preferred `--backend google` is the default for features and training.

`train` defaults to Apple MPS when available; use `--device cpu` to train on the
CPU, or `--steps` to override the recipe's 2,500 steps. Training uses balanced
batches, complete shifted positive feature windows, composite negative windows,
and silence embeddings. A small dense network learns the wake phrases from the
frozen Google features. Checkpoint selection uses validation balanced loss only.
Test speakers are not used for training, model selection, or threshold calibration.

The selected checkpoint is exported to `models/airis_wakeword.onnx` with a JSON
sidecar describing its frontend, hashes, phrases, and trigger threshold. Exported
probabilities must match the CPU PyTorch checkpoint within `1e-4` before the
artifact is accepted. `google_training_report.json` records configuration and history.

An optional `--backend scratch` baseline uses analytic 40-band power mel features
and a depthwise CNN trained from scratch. It is retained for comparison and had
substantially poorer validation performance. It is not the preferred product base.

`calibrate` scores a continuous validation stream of augmented negative clips,
then selects a threshold against the recipe's false activation target. Activations
require two consecutive frames, quiet before rearming, and a 1.5-second cooldown.
Positive validation includes raw and augmented recordings. A positive activation
before its known dry-waveform placement does not count as a successful detection.
If no useful threshold achieves the target, the reported recall can be zero;
the artifact remains a failed prototype rather than being marked ready.

`evaluate` applies the validation threshold to held-out test speakers. It records
per-phrase recall, raw versus augmented recall, false activation episodes and
negative hours, and trigger timing. Positive evaluation includes 480 ms of trailing
silence plus alignment to an 80 ms frame. Timing is relative to the dry waveform's
end, including any silence already present, rather than a forced phoneme alignment.
The CPU benchmark measures p50/p95/p99 duration for the complete feature-to-trigger
path; it excludes microphone I/O and concurrent LLM/TTS activity. Results are saved
in `calibration_report.json` and `evaluation_report.json`.

`ready_for_ui` remains false until the acceptance targets and live microphone
evaluation are satisfied. The optional experimental UI listener deliberately
does not claim readiness or change the model's acceptance flags. Synthetic acceptance
requires at least 90% recall for each phrase, no more than the configured observed
false activation rate, and p95 CPU processing below an 80 ms frame. These are
development checks, not a commercial-quality guarantee.

Calibration and acceptance also limit the fraction of negative frames scoring
above threshold to 0.1% by default (`max_validation_negative_score_fraction`).
Episode counts alone can reward a model that stays active and never rearms.
The additional guard rejects that behavior; checkpoint selection prioritizes
satisfying both calibration constraints before comparing per-phrase recall.

For later product integration, `input.wakeword.WakeWordDetector` exposes `feed()`
for 16 kHz mono PCM16 arrays or normalized float arrays, and `reset()` for new
sessions. It keeps persistent ONNX CPU sessions and bounded feature and pending buffers.
It does not open a microphone or invoke Whisper, Qwen, or speech playback.

Synthetic evaluation is a development benchmark. It cannot establish real-world
microphone accuracy. Later live-microphone checks can be evaluation-only while
keeping the training dataset entirely synthetic.

An explicitly started short microphone check does not save recordings, load
Whisper or Qwen, or add evaluation speech to the training corpus:

```bash
.venv/bin/python -B -m training.wakeword microphone-check --output training/wakeword/runs/hey_airis_v2 --seconds 20
```

Say each wake phrase several times with pauses, then try ordinary speech in a
separate session. Use `--input-device "Device name"` to select a microphone.
The check follows the system default input when no device is supplied. Stop it
with Ctrl+C. This is an interactive smoke check; longer labelled microphone
sessions are needed to estimate recall and false activations in normal use.
The command does not enable background listening or change model readiness.

## Pronunciation review

The `review` stage creates a small bundle of unchanged normal-speed training
recordings for "Hey Airis", "Airis", and similar names. It verifies source hashes
and does not include held-out speakers, change labels, or exclude benchmark data.

```bash
.venv/bin/python -B -m training.wakeword review --output training/wakeword/runs/hey_airis_v2
```

Open `runs/hey_airis_v2/review/README.md` to compare the clips. The local
`phonetic_similarity_report.json` is a log-mel DTW similarity proxy from training
audio, not a phoneme transcription. Listening review is needed before deciding
whether confusers such as "Hey Eris" are distinct from the intended wake phrase.

## Licensing and commercial distribution

The [openWakeWord code](https://github.com/dscripka/openWakeWord#license) is
Apache-2.0; its bundled wake-word classifiers are non-commercial. The current
detector is implemented independently and does not reuse openWakeWord code,
classifiers, or upstream negative feature datasets.

The licences of the published shared `melspectrogram` and `embedding_model` binaries
are still the subject of an [unanswered upstream clarification request](https://github.com/dscripka/openWakeWord/issues/348).
Training a new classifier does not resolve that question. These files are not
downloaded here. The original Google base is identified as Apache-2.0 in
[openWakeWord's architecture documentation](https://github.com/dscripka/openWakeWord#model-architecture).
The current pipeline reproduces its weights directly from Google's original
[published model](https://www.kaggle.com/models/google/speech-embedding/tensorFlow1/speech-embedding/1)
and records the archive, conversion, and frontend hashes. Preserve the original
model's licence and attribution notices in a product distribution. This prototype
is not a finished commercial release or a validation of all distribution terms.

[Qwen VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
and [Qwen Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) declare
Apache-2.0 weights. Before a production corpus run, pin the conversions, retain
their notices, and record their hashes and provenance. Dataset generation here
uses newly designed synthetic speakers rather than recordings of people.

This offline corpus pipeline does not change Airis's existing Qwen speech output
or its saved voice profiles.
