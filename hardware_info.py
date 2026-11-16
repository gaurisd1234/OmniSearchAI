"""Report Python, NVIDIA driver, and PyTorch CUDA status for this machine.

Run from the project root with ``.venv\\Scripts\\python.exe hardware_info.py``.
The CUDA check performs a real matrix multiplication when CUDA is available.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def main() -> int:
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi], capture_output=True, text=True, check=False
        )
        print("\nnvidia-smi:")
        print(result.stdout.strip() or result.stderr.strip())
        print(f"nvidia-smi exit code: {result.returncode}")
    else:
        print("\nnvidia-smi: not found on PATH")

    try:
        import torch
    except ImportError:
        print("\nPyTorch: not installed in this interpreter")
        return 1

    print(f"\nPyTorch: {torch.__version__}")
    print(f"PyTorch CUDA build: {torch.version.cuda or 'CPU-only'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 2

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    try:
        # This verifies actual GPU execution, beyond CUDA discovery alone.
        a = torch.randn((512, 512), device="cuda")
        b = torch.randn((512, 512), device="cuda")
        product = a @ b
        torch.cuda.synchronize()
        print(
            "GPU matrix multiply: passed "
            f"(shape={tuple(product.shape)}, device={product.device})"
        )
    except Exception as exc:  # Report runtime/driver problems with a failing exit.
        print(f"GPU matrix multiply: FAILED ({type(exc).__name__}: {exc})")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
