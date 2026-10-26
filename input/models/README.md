# Wake-word inference assets

`wakeword/` contains the locally installed experimental ONNX classifier and JSON
sidecar, plus `google_base/google_speech_embedding.onnx` and
`google_base/frontend_assets.npz`. These generated binaries are ignored by Git.
The desktop listener does not read training data or import training frameworks.

Quit Airis before installing a replacement:

```bash
.venv/bin/python -B -m input.wakeword_assets \
  --source training/wakeword/runs/hey_airis_v2/models/airis_wakeword.onnx
```

Installation validates every declared artifact hash, rewrites only the frontend
location, and retains the calibrated threshold and failed readiness flags. This
local package enables optional testing; it is not evidence of product readiness.
Distributions must include these runtime files and resolve model attribution and
license requirements separately.

An optional `wakeword/personal_calibration.json` proposes a threshold from a
personal evaluation. Its model hash and separation statistics are checked before
showing an explicit preset button in Settings. It is ignored when missing,
invalid, or associated with a different classifier. The profile does not change
model weights or calibrated acceptance flags and is not applied automatically.
