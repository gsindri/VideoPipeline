from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _print(msg: str = "") -> None:
    sys.stdout.write(msg + "\n")


@dataclass(frozen=True)
class ExportIndex:
    names: set[str]
    ordinals: set[int]


def _iter_search_dirs(extra: Iterable[Path]) -> list[Path]:
    out: list[Path] = []

    def _add(p: Path | None) -> None:
        if not p:
            return
        try:
            rp = p.expanduser().resolve()
        except Exception:
            rp = p
        if str(rp) not in seen and rp.exists() and rp.is_dir():
            out.append(rp)
            seen.add(str(rp))

    seen: set[str] = set()

    # User-provided / env
    env = os.environ.get("FFMPEG_SHARED_BIN")
    if env:
        _add(Path(env))

    for p in extra:
        _add(p)

    # Torch DLLs (torch/lib) are a very common provider for missing symbols.
    try:
        import torch  # noqa: F401

        torch_root = Path(torch.__file__).resolve().parent
        _add(torch_root / "lib")
        _add(torch_root / "bin")
    except Exception:
        pass

    # System dirs
    sysroot = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    _add(sysroot / "System32")
    _add(sysroot / "SysWOW64")

    # CWD
    _add(Path.cwd())

    # PATH
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw.strip():
            _add(Path(raw.strip()))

    return out


def _resolve_dll(dll_name: str, search_dirs: list[Path]) -> Path | None:
    # API set pseudo DLLs won't exist on disk.
    name = dll_name.strip().strip("\x00")
    if not name:
        return None

    for d in search_dirs:
        candidate = d / name
        if candidate.exists():
            return candidate
    return None


def _load_exports(path: Path) -> ExportIndex:
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]])

    names: set[str] = set()
    ordinals: set[int] = set()
    exp_dir = getattr(pe, "DIRECTORY_ENTRY_EXPORT", None)
    if exp_dir is None:
        return ExportIndex(names=names, ordinals=ordinals)

    for sym in exp_dir.symbols:
        if sym.ordinal is not None:
            ordinals.add(int(sym.ordinal))
        if sym.name:
            try:
                names.add(sym.name.decode("utf-8", errors="ignore"))
            except Exception:
                continue
    return ExportIndex(names=names, ordinals=ordinals)


def main() -> int:
    if sys.platform != "win32":
        _print("This script is intended for Windows PE DLL inspection.")
        return 2

    ap = argparse.ArgumentParser(description="Inspect PE import tables and check for missing imports/exports.")
    ap.add_argument("dll", type=str, help="Path to the DLL to inspect (e.g. libtorchcodec_core8.dll).")
    ap.add_argument(
        "--search-dir",
        action="append",
        default=[],
        help="Additional directory to search for dependency DLLs (can be repeated).",
    )
    args = ap.parse_args()

    target = Path(args.dll).expanduser()
    if not target.exists():
        _print(f"ERROR: DLL not found: {target}")
        return 1

    extra_dirs = [Path(p) for p in args.search_dir]
    search_dirs = _iter_search_dirs(extra_dirs)

    _print(f"dll: {target.resolve()}")
    _print(f"FFMPEG_SHARED_BIN: {os.environ.get('FFMPEG_SHARED_BIN')}")
    _print(f"search_dir_count: {len(search_dirs)}")

    import pefile

    pe = pefile.PE(str(target), fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])

    imports = getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
    if not imports:
        _print("No import table found.")
        return 0

    missing_modules: list[str] = []
    missing_symbols: list[str] = []

    export_cache: dict[Path, ExportIndex] = {}

    for entry in imports:
        dll_name = entry.dll.decode("utf-8", errors="ignore")
        resolved = _resolve_dll(dll_name, search_dirs)
        if not resolved:
            # This can be normal for API-set DLLs; still report for completeness.
            missing_modules.append(dll_name)
            continue

        exports = export_cache.get(resolved)
        if exports is None:
            exports = _load_exports(resolved)
            export_cache[resolved] = exports

        for imp in entry.imports:
            if imp.name:
                sym = imp.name.decode("utf-8", errors="ignore")
                if sym and sym not in exports.names:
                    missing_symbols.append(f"{dll_name}!{sym}  (provider={resolved.name})")
            else:
                # Import-by-ordinal
                ord_ = int(getattr(imp, "ordinal", 0) or 0)
                if ord_ and ord_ not in exports.ordinals:
                    missing_symbols.append(f"{dll_name}#ORD{ord_}  (provider={resolved.name})")

    # Output summary
    _print("")
    if missing_modules:
        _print("missing_modules:")
        for m in sorted(set(missing_modules)):
            _print(f"  - {m}")
    else:
        _print("missing_modules: none")

    _print("")
    if missing_symbols:
        _print("missing_imports_not_exported_by_provider:")
        for s in sorted(set(missing_symbols))[:200]:
            _print(f"  - {s}")
        if len(set(missing_symbols)) > 200:
            _print(f"  ... ({len(set(missing_symbols)) - 200} more)")
    else:
        _print("missing_imports_not_exported_by_provider: none")

    # A missing procedure error (WinError 127) should correspond to at least one
    # entry in missing_symbols, but if it does not, the module may be using
    # delay-load imports or dynamic GetProcAddress calls.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

