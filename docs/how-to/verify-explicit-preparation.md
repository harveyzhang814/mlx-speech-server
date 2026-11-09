# Verify explicit preparation

Use Apple Silicon macOS with Metal, a Python environment containing this project's
runtime and test dependencies, `ffmpeg`, and the local macOS Samantha voice.

1. Make a complete Whisper model snapshot available locally. The test selects the
   existing cached `mlx-community/whisper-large-v3-turbo` snapshot without network
   access. To select a particular immutable snapshot, set:

   ```bash
   export MLX_E2E_MODEL_PATH="$HOME/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/<commit>"
   ```

2. From the service checkout, run:

   ```bash
   python -m pytest tests/e2e/test_explicit_preparation.py -m e2e -v -s
   ```

3. Check that both tests passed, rather than skipped. Missing local weights or
   Metal means actual model loading was not verified. The default suite excludes
   E2E tests. `test_idle_unload.py` separately uses a simulated loaded holder and
   does not establish real-model preparation.

The tests start their own loopback HTTP processes on allocated ports with temporary
configuration and an isolated HF cache root. They generate a non-private spoken
sentence locally, prepare real weights, send the original multipart transcription
request using the exact configured path, and observe the actual model identity
and underlying load count. The observation route exists only in test children;
no model, native call, or response is replaced by a fake.

The second process sets its request deadline to 1 ms and observes a genuine native
load continuing after that deadline. It must remain active until completion and
duplicate prepares must retain the operation ID. No artificial native delay is
inserted. Neither test changes the normal idle-unload constants.

Cleanup signals only the child PIDs created by the tests and checks shared MLX
config/plist hashes and existing known service PIDs. The tests do not install,
restart, or stop the shared launchd job. Consequently this validates direct HTTP
runtime behavior, not real `mlx ensure-running` launchd lifecycle or clean-machine
wheel bootstrap. Run those separately in an appropriately isolated environment;
the CLI currently has no independent service label/config override. A warm local
snapshot also does not validate cold download/resume progress.

Protocol details are in the [API reference](../reference/api.md).
