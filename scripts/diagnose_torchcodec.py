from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from pathlib import Path


def _print_kv(key: str, value: object) -> None:
    print(f"{key}: {value}")


def _find_ffmpeg_bin_dir() -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    return Path(ffmpeg).resolve().parent


def _list_ffmpeg_dlls(bin_dir: Path) -> list[Path]:
    patterns = [
        "avcodec-*.dll",
        "avformat-*.dll",
        "avutil-*.dll",
        "swresample-*.dll",
        "swscale-*.dll",
    ]
    dlls: list[Path] = []
    for pat in patterns:
        dlls.extend(sorted(bin_dir.glob(pat)))
    return dlls


def _try_load_dll(path: Path) -> tuple[bool, str]:
    try:
        ctypes.WinDLL(str(path))
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    _print_kv("python", sys.version.replace("\n", " "))
    _print_kv("executable", sys.executable)
    _print_kv("platform", platform.platform())
    _print_kv("cwd", os.getcwd())
    _print_kv("PATH_contains_ffmpeg", bool(shutil.which("ffmpeg")))

    try:
        import torch

        _print_kv("torch", getattr(torch, "__version__", "unknown"))
    except Exception as exc:
        _print_kv("torch_import_error", f"{type(exc).__name__}: {exc}")
        return 1

    if sys.platform != "win32":
        print("This script is mainly useful on Windows.")
        return 0

    ffmpeg_bin = _find_ffmpeg_bin_dir()
    _print_kv("ffmpeg_bin_dir", ffmpeg_bin)
    if ffmpeg_bin:
        ffmpeg_dlls = _list_ffmpeg_dlls(ffmpeg_bin)
        _print_kv("ffmpeg_dll_count", len(ffmpeg_dlls))
        if ffmpeg_dlls:
            print("ffmpeg_dlls:")
            for p in ffmpeg_dlls[:25]:
                print(f"  - {p.name}")
            if len(ffmpeg_dlls) > 25:
                print(f"  ... ({len(ffmpeg_dlls) - 25} more)")
        else:
            print(
                "No FFmpeg DLLs found next to ffmpeg.exe. If your ffmpeg is a static build, "
                "TorchCodec won't be able to link against it."
            )

    # TorchCodec
    try:
        import torchcodec  # noqa: F401
        import torchcodec as tc

        tc_dir = Path(tc.__file__).resolve().parent
        _print_kv("torchcodec_dir", tc_dir)
        _print_kv("torchcodec_version", getattr(tc, "__version__", "unknown"))
    except Exception as exc:
        _print_kv("torchcodec_import_error", f"{type(exc).__name__}: {exc}")
        return 2

    core_dlls = sorted(tc_dir.glob("libtorchcodec_core*.dll"))
    _print_kv("torchcodec_core_dll_count", len(core_dlls))
    if not core_dlls:
        print("No libtorchcodec_core*.dll files found in torchcodec package directory.")
        return 3

    # Try load in the same order as TorchCodec: newest FFmpeg core first.
    def _sort_key(p: Path) -> tuple[int, str]:
        # libtorchcodec_core7.dll -> 7
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        n = int(digits) if digits else -1
        return (n, p.name)

    core_dlls = sorted(core_dlls, key=_sort_key, reverse=True)
    print("torchcodec_core_dll_load_attempts:")
    any_ok = False
    for dll in core_dlls:
        ok, msg = _try_load_dll(dll)
        any_ok = any_ok or ok
        print(f"  - {dll.name}: {msg}")

    if not any_ok:
        print(
            "\nNone of the TorchCodec core DLLs could be loaded.\n"
            "Most common causes:\n"
            "  1) FFmpeg shared DLLs (avcodec/avformat/avutil/...) not on DLL search path\n"
            "  2) TorchCodec wheel not compatible with your PyTorch build (ABI mismatch)\n"
            "  3) Missing MSVC runtime (Visual C++ Redistributable)\n"
        )
        return 4

    print("\nAt least one TorchCodec core DLL loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

