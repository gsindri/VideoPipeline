from __future__ import annotations

import json
import os
from typing import Any, Optional

_SERVICE_PREFIX = "VideoPipeline"


def _compound_target(username: str) -> str:
    return f"{username}@{_SERVICE_PREFIX}"


def _wincred_modules():
    from win32ctypes.pywin32 import pywintypes, win32cred

    return pywintypes, win32cred


def _wincred_get(username: str) -> Optional[str]:
    pywintypes, win32cred = _wincred_modules()
    for target in (_SERVICE_PREFIX, _compound_target(username)):
        try:
            cred = win32cred.CredRead(
                Type=win32cred.CRED_TYPE_GENERIC,
                TargetName=target,
            )
        except pywintypes.error as exc:
            if exc.winerror == 1168 and exc.funcname == "CredRead":
                continue
            raise
        if cred.get("UserName") != username:
            continue
        blob = cred.get("CredentialBlob")
        if isinstance(blob, bytes):
            try:
                return blob.decode("utf-16")
            except UnicodeDecodeError:
                return blob.decode("utf-8")
        if blob is None:
            return None
        return str(blob)
    return None


def _wincred_set(username: str, payload: str) -> None:
    _, win32cred = _wincred_modules()
    win32cred.CredWrite(
        {
            "Type": win32cred.CRED_TYPE_GENERIC,
            "TargetName": _compound_target(username),
            "UserName": username,
            "CredentialBlob": payload,
            "Comment": "Stored by VideoPipeline",
            "Persist": win32cred.CRED_PERSIST_ENTERPRISE,
        },
        0,
    )


def _wincred_delete(username: str) -> None:
    pywintypes, win32cred = _wincred_modules()
    for target in (_SERVICE_PREFIX, _compound_target(username)):
        try:
            cred = win32cred.CredRead(
                Type=win32cred.CRED_TYPE_GENERIC,
                TargetName=target,
            )
        except pywintypes.error as exc:
            if exc.winerror == 1168 and exc.funcname == "CredRead":
                continue
            raise
        if cred.get("UserName") != username:
            continue
        try:
            win32cred.CredDelete(
                Type=win32cred.CRED_TYPE_GENERIC,
                TargetName=target,
            )
        except pywintypes.error as exc:
            if exc.winerror == 1168 and exc.funcname == "CredDelete":
                continue
            raise


def _keyring():
    import keyring

    return keyring


def _key(platform: str, account_id: str) -> str:
    return f"{_SERVICE_PREFIX}:{platform}:{account_id}"


def store_tokens(platform: str, account_id: str, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload)
    username = _key(platform, account_id)
    if os.name == "nt":
        _wincred_set(username, raw)
        return
    _keyring().set_password(_SERVICE_PREFIX, username, raw)


def load_tokens(platform: str, account_id: str) -> Optional[dict[str, Any]]:
    username = _key(platform, account_id)
    if os.name == "nt":
        raw = _wincred_get(username)
    else:
        raw = _keyring().get_password(_SERVICE_PREFIX, username)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def delete_tokens(platform: str, account_id: str) -> None:
    username = _key(platform, account_id)
    if os.name == "nt":
        _wincred_delete(username)
        return
    _keyring().delete_password(_SERVICE_PREFIX, username)
