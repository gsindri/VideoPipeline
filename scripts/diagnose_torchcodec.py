from __future__ import annotations

import ctypes
import os
import platform
import shutil
import sys
from dataclasses import dataclass
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
        if isinstance(exc, OSError) and getattr(exc, "winerror", None) is not None:
            return False, f"{type(exc).__name__} [WinError {exc.winerror}]: {exc}"
        return False, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class _TorchCodecLayout:
    package_dir: Path
    core_dlls: list[Path]


def _find_torchcodec_layout() -> _TorchCodecLayout | None:
    """Find torchcodec package dir even when import fails."""
    try:
        import site

        candidates: list[Path] = []
        try:
            candidates.extend([Path(p) for p in site.getsitepackages()])  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            candidates.append(Path(site.getusersitepackages()))
        except Exception:
            pass

        for sp in candidates:
            pkg = sp / "torchcodec"
            if pkg.exists() and pkg.is_dir():
                core = sorted(pkg.glob("libtorchcodec_core*.dll"))
                return _TorchCodecLayout(package_dir=pkg, core_dlls=core)
    except Exception:
        return None
    return None


def _print_ffmpeg_shared_dir_evidence(bin_dir: Path) -> None:
    dlls = _list_ffmpeg_dlls(bin_dir)
    _print_kv("FFMPEG_SHARED_BIN_ffmpeg_dll_count", len(dlls))
    if dlls:
        print("FFMPEG_SHARED_BIN_ffmpeg_dlls:")
        for p in dlls[:25]:
            print(f"  - {p.name}")
        if len(dlls) > 25:
            print(f"  ... ({len(dlls) - 25} more)")
    else:
        print("No FFmpeg DLLs found in FFMPEG_SHARED_BIN.")


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

    # If provided, add the FFmpeg shared bin directory to the DLL search path
    # before importing torchcodec.
    dll_dir_handle = None
    ffmpeg_shared_bin_env = os.environ.get("FFMPEG_SHARED_BIN")
    _print_kv("FFMPEG_SHARED_BIN", ffmpeg_shared_bin_env)
    if ffmpeg_shared_bin_env:
        ffmpeg_shared_bin = Path(ffmpeg_shared_bin_env).expanduser()
        _print_kv("FFMPEG_SHARED_BIN_exists", ffmpeg_shared_bin.exists())
        if ffmpeg_shared_bin.exists():
            _print_kv("FFMPEG_SHARED_BIN_resolved", str(ffmpeg_shared_bin.resolve()))
            _print_ffmpeg_shared_dir_evidence(ffmpeg_shared_bin)
            try:
                dll_dir_handle = os.add_dll_directory(str(ffmpeg_shared_bin))
                _print_kv("FFMPEG_SHARED_BIN_added", True)
            except Exception as exc:
                _print_kv("FFMPEG_SHARED_BIN_added", False)
                _print_kv("FFMPEG_SHARED_BIN_add_error", f"{type(exc).__name__}: {exc}")
        else:
            _print_kv("FFMPEG_SHARED_BIN_added", False)

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
        layout = _find_torchcodec_layout()
        if layout is None:
            return 2

        _print_kv("torchcodec_dir_guess", layout.package_dir)
        _print_kv("torchcodec_core_dll_count", len(layout.core_dlls))
        if not layout.core_dlls:
            print("No libtorchcodec_core*.dll files found in torchcodec package directory.")
            return 3

        print("torchcodec_core_dll_load_attempts:")
        any_ok = False
        for dll in sorted(layout.core_dlls, reverse=True):
            ok, msg = _try_load_dll(dll)
            any_ok = any_ok or ok
            print(f"  - {dll.name}: {msg}")

        if not any_ok:
            return 4
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
    # Keep handle alive until exit (otherwise directory is removed from search path).
    _ = dll_dir_handle
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
