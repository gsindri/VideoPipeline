# Windows ROCm: fixing TorchCodec (WinError 127)

If you're using the **custom Windows ROCm PyTorch build** (for example: `torch==2.9.1+rocmsdk...`), the **prebuilt TorchCodec wheels** may fail to load with:

- `WinError 127` (“The specified procedure could not be found”)

This usually means the TorchCodec native DLL was found, but it expects exports from your `torch_*.dll` / `torch_*.lib` that aren’t present in that specific ROCm build (ABI/export mismatch).

The durable fix is:

1) install an FFmpeg **shared** build (DLLs, not just `ffmpeg.exe`)
2) keep the *Python* `torchcodec` package installed, but **rebuild and replace the `libtorchcodec_*` binaries** against your installed torch

## Prereqs

- Windows 11
- A working repo venv at `.venv\`
- Visual Studio / Build Tools with **Desktop development with C++** (needed for headers + link.exe + SDK)

## 1) Install FFmpeg shared + set env var

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_ffmpeg_shared.ps1
```

This will:

- install to `C:\Tools\ffmpeg-shared\`
- verify `C:\Tools\ffmpeg-shared\bin` contains `avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`
- set a persistent **user** env var: `FFMPEG_SHARED_BIN=C:\Tools\ffmpeg-shared\bin`

Note: user env vars apply to **new** terminals/processes. The script also sets `$env:FFMPEG_SHARED_BIN` for the current session.

## 2) Rebuild TorchCodec for your ROCm torch

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rebuild_torchcodec_rocm.ps1
```

What it does:

- ensures `torchcodec==0.9.1` is installed with `--no-deps` (so pip doesn’t replace your custom torch)
- clones TorchCodec source into `cache\torchcodec\src` (gitignored)
- builds with Ninja + ROCm clang (`clang-cl`) using your installed ROCm SDK (from `rocm-sdk[devel]`)
- copies rebuilt `libtorchcodec_*.dll` and `libtorchcodec_*.pyd` into your environment’s `site-packages\torchcodec\`
- runs diagnostics at the end

If GPU arch detection fails, pass it explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rebuild_torchcodec_rocm.ps1 -PytorchRocmArch gfx1201
```

## 3) Quick “doctor” check

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\doctor_torchcodec.ps1
```

This exits non-zero if core7/core8 don’t load.

## Important: pip can undo this

If you later run something like:

- `pip install --force-reinstall torchcodec`

…you’ll likely get the prebuilt wheel again, which can reintroduce WinError 127. When that happens, re-run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\rebuild_torchcodec_rocm.ps1
```

## Advanced debugging

If you want to confirm the ABI/export mismatch, `scripts/inspect_dll_imports.py` can compare the imports of `libtorchcodec_core*.dll` to the exports of provider DLLs found in your environment (`torch\lib`, `FFMPEG_SHARED_BIN`, system PATH). It requires `pefile`.

