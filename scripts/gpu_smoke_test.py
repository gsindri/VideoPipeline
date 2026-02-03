from __future__ import annotations

import argparse
import time

import torch


def _dtype_from_name(name: str) -> torch.dtype:
    name = name.lower().strip()
    if name in {"float32", "fp32"}:
        return torch.float32
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Quick GPU smoke test for torch (ROCm builds often use device='cuda').")
    ap.add_argument("--n", type=int, default=4096, help="Matrix size (NxN)")
    ap.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    ap.add_argument("--device", type=str, default="cuda", help="Device string (default: cuda)")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup matmul iterations (not timed)")
    ap.add_argument("--iters", type=int, default=1, help="Timed matmul iterations")
    args = ap.parse_args()

    print("torch:", torch.__version__)
    print("torch.version.hip:", getattr(torch.version, "hip", None))
    print("cuda available:", torch.cuda.is_available())
    print("cuda device_count:", torch.cuda.device_count())

    if not torch.cuda.is_available():
        return 2

    device = torch.device(args.device)
    print("device:", torch.cuda.get_device_name(0))
    print("using device:", device)

    dtype = _dtype_from_name(args.dtype)
    print("dtype:", dtype)
    print("n:", args.n)

    a = torch.randn(args.n, args.n, device=device, dtype=dtype)
    print("a.device:", a.device)

    # Warmup
    for _ in range(max(0, args.warmup)):
        _ = a @ a
    torch.cuda.synchronize()

    t0 = time.time()
    b = None
    for _ in range(max(1, args.iters)):
        b = a @ a
    torch.cuda.synchronize()
    dt = time.time() - t0

    assert b is not None
    print("b.device:", b.device)
    print("matmul ok, seconds:", dt, "mean:", b.mean().item())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

