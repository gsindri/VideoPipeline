# Diarization Benchmark (pyannote vs Senko)

This repo includes a small benchmark harness to compare our existing **pyannote** diarization against **Senko**.

The benchmark:
- Auto-finds a recent audio/video file (optional) or uses `--input`
- Converts to Senko-required WAV (`16kHz`, `mono`, `16-bit PCM`) via `ffmpeg`
- Runs one or both diarization backends
- Writes a single `report.json` plus per-backend outputs under `outputs/diarization_bench/`

## Requirements

- `ffmpeg` available on `PATH`
- For pyannote: install optional deps and set a Hugging Face token
  - `pip install -e '.[diarization]'`
  - Set `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`)

## Senko setup (isolated venv under `tools/senko/`)

Senko must **not** be installed into the main project venv. The benchmark calls it via `subprocess` using a separate interpreter.

### A) WSL/Linux (preferred)

If you hit build errors, install basic build tools first (example for Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y build-essential
```

Create the isolated venv and install Senko:

```bash
mkdir -p tools/senko
python3 -m venv tools/senko/.venv
source tools/senko/.venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install "git+https://github.com/narcotic-sh/senko.git"
deactivate
```

### B) Windows (optional)

```powershell
mkdir tools\senko -Force
py -3 -m venv tools\senko\.venv
tools\senko\.venv\Scripts\python.exe -m pip install -U pip setuptools wheel
tools\senko\.venv\Scripts\python.exe -m pip install "git+https://github.com/narcotic-sh/senko.git"
```

## Running the benchmark

Auto-pick the most recent media file under common dirs (`outputs/`, `out/`, `runs/`, `data/`, `cache/`, `downloads/`):

```bash
python tools/diarization_bench.py --auto-latest --backend both --seconds 600
```

Run only pyannote (baseline) on a 60s slice:

```bash
python tools/diarization_bench.py --auto-latest --backend pyannote --seconds 60
```

Run only Senko (explicit interpreter path):

```bash
python tools/diarization_bench.py --input "<path>" --backend senko --seconds 600 \
  --senko-python tools/senko/.venv/bin/python
```

Keep the converted WAV (useful for debugging):

```bash
python tools/diarization_bench.py --auto-latest --backend both --seconds 600 --keep-wav
```

## Outputs

Written to `outputs/diarization_bench/`:
- `report.json` (single summary report)
- `pyannote_segments.json` (if pyannote ran)
- `senko_segments.json` + `senko.rttm` (if Senko ran)

Segments JSON schema is stable and easy to diff:

```json
[
  {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.23}
]
```

