"""Opt-in import tracing helpers.

Currently used to diagnose noisy TensorFlow/protobuf warnings by logging a stack
trace when TensorFlow (or tf_keras) is imported.
"""

from __future__ import annotations

import importlib.abc
import logging
import sys
import threading
import traceback
from typing import Callable, Iterable, Optional


logger = logging.getLogger("videopipeline.import_tracer")

_DEFAULT_PREFIXES = ("tensorflow", "tf_keras", "tensorflow_hub")

_state_lock = threading.Lock()
_state_refcount = 0
_state_finder: Optional["_PrefixImportTracer"] = None


def _match_prefix(fullname: str, prefixes: Iterable[str]) -> Optional[str]:
    for p in prefixes:
        if fullname == p or fullname.startswith(p + "."):
            return p
    return None


def _format_stack(limit: int = 60) -> str:
    stack = traceback.format_stack(limit=limit)
    # Drop frames inside this tracer to keep output focused.
    stack = [s for s in stack if "videopipeline/import_tracer.py" not in s.replace("\\", "/")]
    # Avoid noisy importlib internals, but keep site-packages context.
    filtered: list[str] = []
    for s in stack:
        s_norm = s.replace("\\", "/")
        if "/importlib/" in s_norm and "site-packages" not in s_norm and "src/videopipeline" not in s_norm:
            continue
        filtered.append(s)
    return "".join(filtered).rstrip()


class _PrefixImportTracer(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes: Iterable[str]) -> None:
        self._prefixes = tuple(prefixes)
        self._lock = threading.Lock()
        self._reported: set[str] = set()

    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        prefix = _match_prefix(fullname, self._prefixes)
        if prefix is None:
            return None
        with self._lock:
            if prefix in self._reported:
                return None
            self._reported.add(prefix)

        stack = _format_stack()
        logger.warning(
            "[IMPORT TRACE] Detected import of %s (triggered by %s). Stack:\n%s",
            prefix,
            fullname,
            stack,
        )
        return None


def enable_tf_import_trace(prefixes: Optional[Iterable[str]] = None) -> Callable[[], None]:
    """Enable tracing of TensorFlow imports (ref-counted).

    Returns a callable that disables tracing when invoked.
    """
    global _state_refcount, _state_finder

    prefixes = tuple(prefixes or _DEFAULT_PREFIXES)

    with _state_lock:
        if _state_finder is None:
            _state_finder = _PrefixImportTracer(prefixes)
            sys.meta_path.insert(0, _state_finder)
        _state_refcount += 1

    already_loaded = []
    for p in prefixes:
        if p in sys.modules:
            already_loaded.append(p)
            continue
        if any(m == p or m.startswith(p + ".") for m in sys.modules.keys()):
            already_loaded.append(p)
    if already_loaded:
        logger.warning(
            "[IMPORT TRACE] Enabled, but already imported: %s (you may not see import stacks for these).",
            ", ".join(sorted(set(already_loaded))),
        )

    def _disable() -> None:
        global _state_refcount, _state_finder
        with _state_lock:
            _state_refcount = max(0, _state_refcount - 1)
            if _state_refcount > 0:
                return
            if _state_finder is not None:
                try:
                    sys.meta_path.remove(_state_finder)
                except ValueError:
                    pass
                _state_finder = None

    return _disable

