"""Reproduce an ONNX embedding directly from Google's original published weights."""

import hashlib
import json
from pathlib import Path
import tarfile
import urllib.error
import urllib.request


SOURCES = (
    "https://storage.googleapis.com/tfhub-modules/google/speech_embedding/1.tar.gz",
    "https://www.kaggle.com/api/v1/models/google/speech-embedding/tensorFlow1/speech-embedding/1/download",
)
MODEL_CARD = "https://www.kaggle.com/models/google/speech-embedding/tensorFlow1/speech-embedding/1"


def prepare_google_base(root):
    root = Path(root) / "google_base"
    root.mkdir(parents=True, exist_ok=True)
    existing = root / "conversion_report.json"
    if existing.exists():
        report = json.loads(existing.read_text())
        files = {"google_speech_embedding.onnx": report["onnx_sha256"],
                 "frontend_assets.npz": report["frontend_assets_sha256"],
                 "original.tar.gz": report["source"]["archive_sha256"]}
        if all((root / name).exists() and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
               for name, digest in files.items()):
            return report
    archive = root / "original.tar.gz"
    if not archive.exists():
        failures = []
        for url in SOURCES:
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Airis model conversion"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read(100 * 1024 * 1024)
                temporary = archive.with_suffix(".partial.gz")
                temporary.write_bytes(payload)
                with tarfile.open(temporary) as bundle:
                    bundle.getmembers()
                temporary.replace(archive)
                (root / "source.json").write_text(json.dumps({
                    "source": url, "model_card": MODEL_CARD,
                    "original_model": "google/speech_embedding/1", "declared_license": "Apache-2.0",
                    "archive_sha256": hashlib.sha256(payload).hexdigest(),
                    "upstream_converted_binaries_used": False,
                }, indent=2) + "\n")
                break
            except (OSError, urllib.error.URLError, tarfile.TarError) as error:
                failures.append(f"{url}: {error}")
        else:
            raise ValueError("Could not retrieve Google's original model: " + "; ".join(failures))
    directory = root / "original"
    if not (directory / "saved_model.pb").exists():
        directory.mkdir(exist_ok=True)
        with tarfile.open(archive) as bundle:
            bundle.extractall(directory, filter="data")
    if not (directory / "saved_model.pb").exists():
        raise ValueError("The original archive does not contain the expected SavedModel.")
    import tensorflow as tf
    import tf2onnx
    import numpy as np
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # The original TFHub artifact contains separate training and inference graphs.
    module = tf.saved_model.load(str(directory), tags=[])
    signatures = {name: str(function.structured_input_signature) for name, function in module.signatures.items()}
    print(json.dumps({"original_signatures": signatures}), flush=True)
    if not module.signatures:
        raise ValueError("The original model has no callable signatures.")
    original = module.signatures.get("default") or module.signatures.get("serving_default") or next(iter(module.signatures.values()))
    inputs = original.structured_input_signature[1]
    if len(inputs) != 1:
        raise ValueError("The original embedding signature has an unexpected number of inputs.")
    input_name = next(iter(inputs))
    feature_tensor = module.graph.get_operation_by_name("embeddings_apply_default/MudpuppyLite/ExpandDims").inputs[0]
    embedding_tensor = module.graph.get_tensor_by_name("embeddings_apply_default/MudpuppyLite/Conv_18/Conv2D:0")
    network = module.prune(feeds=feature_tensor, fetches=embedding_tensor)
    frontend = module.prune(feeds=module.graph.get_tensor_by_name("samples:0"), fetches=feature_tensor)
    lower = tf.make_ndarray(module.graph.get_operation_by_name("map/while/linear_to_mel_weight_matrix/lower_edge_hertz").get_attr("value")).item()
    upper = tf.make_ndarray(module.graph.get_operation_by_name("map/while/linear_to_mel_weight_matrix/upper_edge_hertz").get_attr("value")).item()
    epsilon = tf.make_ndarray(module.graph.get_operation_by_name("map/while/add_1/y").get_attr("value")).item()
    assets = root / "frontend_assets.npz"
    np.savez(assets, window=tf.signal.hann_window(400, periodic=True).numpy(),
             filters=tf.signal.linear_to_mel_weight_matrix(32, 257, 16000, lower, upper).numpy(),
             epsilon=np.float32(epsilon))
    specification = [tf.TensorSpec((None, 76, 32), tf.float32, name="mel")]

    @tf.function(input_signature=specification)
    def embed(mel):
        value = network(mel)
        return tf.reshape(value, (-1, 96), name="embedding")

    destination = root / "google_speech_embedding.onnx"
    temporary = destination.with_suffix(".partial.onnx")
    tf2onnx.convert.from_function(embed, input_signature=specification,
                                 opset=17, output_path=str(temporary))
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    mel = rng.standard_normal((3, 76, 32)).astype(np.float32)
    expected = embed(tf.constant(mel)).numpy()
    actual = session.run(None, {session.get_inputs()[0].name: mel})[0]
    difference = float(np.max(np.abs(actual - expected)))
    if expected.shape != (3, 96) or not np.allclose(actual, expected, atol=1e-4, rtol=1e-4):
        raise ValueError(f"Original-to-ONNX embedding parity failed: {difference}")
    from input.google_wakeword_features import GoogleMel
    audio = (rng.standard_normal((3, 12400)) * 0.03).astype(np.float32)
    actual_mel = GoogleMel(assets).frames(audio)
    expected_mel = frontend(tf.constant(audio)).numpy()
    frontend_difference = float(np.max(np.abs(actual_mel - expected_mel)))
    if not np.allclose(actual_mel, expected_mel, atol=1e-4, rtol=1e-4):
        raise ValueError(f"Original frontend parity failed: {frontend_difference}")
    end_to_end = session.run(None, {session.get_inputs()[0].name: actual_mel})[0]
    original_result = next(iter(original(**{input_name: tf.constant(audio)}).values())).numpy().reshape(-1, 96)
    audio_difference = float(np.max(np.abs(end_to_end - original_result)))
    if not np.allclose(end_to_end, original_result, atol=1e-4, rtol=1e-4):
        raise ValueError(f"Original audio-to-embedding parity failed: {audio_difference}")
    temporary.replace(destination)
    report = {"source": json.loads((root / "source.json").read_text()),
              "tensorflow": tf.__version__, "tf2onnx": tf2onnx.__version__,
              "input_shape": [76, 32], "output_dimensions": 96,
              "max_prediction_difference": difference,
              "frontend_max_difference": frontend_difference,
              "audio_embedding_max_difference": audio_difference,
              "frontend_assets_sha256": hashlib.sha256(assets.read_bytes()).hexdigest(),
              "frontend": {"frame_length": 400, "hop": 160, "fft": 512, "mel_bands": 32,
                           "lower_hertz": lower, "upper_hertz": upper, "epsilon": epsilon,
                           "spectral_measure": "magnitude"},
              "onnx_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
              "onnx_bytes": destination.stat().st_size}
    (root / "conversion_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def inspect_google_base(root):
    import tensorflow as tf
    module = tf.saved_model.load(str(Path(root) / "google_base" / "original"), tags=[])
    result = {"attributes": list(vars(module)), "operations": [], "constants": {},
              "signatures": {name: {"inputs": [tensor.name for tensor in function.inputs],
                                    "outputs": [tensor.name for tensor in function.outputs]}
                             for name, function in module.signatures.items()}}
    for operation in module.graph.get_operations():
        if operation.type == "Const" and any(key in operation.name for key in (
            "frame_length", "frame_step", "lower_edge", "upper_edge", "sample_rate", "add_1/y", "rfft/packed", "hann_window/periodic"
        )):
            result["constants"][operation.name] = tf.make_ndarray(operation.get_attr("value")).tolist()
        if operation.name in ("samples", "audio_samples"):
            result["constants"][operation.name] = {"type": operation.type, "inputs": [tensor.name for tensor in operation.inputs]}
        if "Conv" in operation.type or "Mudpuppy" in operation.name or operation.name.startswith("map/"):
            result["operations"].append({"name": operation.name, "type": operation.type,
                                         "attributes": {name: str(value) for name, value in operation.node_def.attr.items()} if operation.type in ("AudioSpectrogram", "Mfcc") else {},
                                         "inputs": [(tensor.name, str(tensor.shape)) for tensor in operation.inputs],
                                         "outputs": [(tensor.name, str(tensor.shape)) for tensor in operation.outputs]})
    (Path(root) / "google_base" / "inspection.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"operations": len(result["operations"]), "attributes": result["attributes"]}
