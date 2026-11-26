"""Stage 0: record the hardware and software of this machine and run the import smoke tests.

    python scripts/s00_check_env.py --label local     # on the workstation
    python scripts/s00_check_env.py --label kaggle    # in a Kaggle notebook

Writes reports/stage0_env_<label>.md and exits non-zero if a required import fails.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# module -> required on this label?
IMPORTS = {
    "local": ["torch", "transformers", "datasets", "faiss", "bm25s", "gymnasium", "omnisafe",
              "pandas", "pyarrow", "hydra", "scipy", "sklearn", "matplotlib", "pytest", "budgetrag"],
    "kaggle": ["torch", "transformers", "vllm", "pandas", "pyarrow"],
}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def gpus() -> list[dict]:
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
               "--format=csv,noheader"])
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            rows.append(dict(zip(["name", "memory", "driver", "compute_cap"], parts)))
    return rows


def ram_gb() -> float | None:
    try:
        import psutil  # optional
        return psutil.virtual_memory().total / 2**30
    except ImportError:
        pass
    if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return stat.ullTotalPhys / 2**30
    return None


def check_imports(label: str) -> list[tuple[str, str, bool]]:
    results = []
    for mod in IMPORTS[label]:
        try:
            m = importlib.import_module(mod)
            results.append((mod, getattr(m, "__version__", "?"), True))
        except Exception as e:  # noqa: BLE001 - report every failure kind
            results.append((mod, f"FAILED: {type(e).__name__}: {e}"[:120], False))
    return results


def torch_cuda() -> str:
    try:
        import torch
        if not torch.cuda.is_available():
            return f"torch {torch.__version__}: CUDA not available (CPU build or no usable GPU)"
        return f"torch {torch.__version__}: CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)"
    except ImportError:
        return "torch not installed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=sorted(IMPORTS), required=True)
    args = ap.parse_args()

    disk = shutil.disk_usage(ROOT)
    ram = ram_gb()
    imports = check_imports(args.label)
    lines = [
        f"# Stage 0 environment: {args.label}",
        "",
        f"Recorded {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC.",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| OS | {platform.platform()} |",
        f"| CPU | {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores |",
        f"| RAM | {ram:.1f} GB |" if ram else "| RAM | unknown |",
        f"| Disk free at repo | {disk.free / 2**30:.0f} GB of {disk.total / 2**30:.0f} GB |",
        f"| Python | {sys.version.split()[0]} ({sys.executable}) |",
        f"| PyTorch | {torch_cuda()} |",
    ]
    found = gpus()
    if found:
        for i, g in enumerate(found):
            lines.append(f"| GPU {i} | {g['name']}, {g['memory']}, driver {g['driver']}, "
                         f"compute capability {g['compute_cap']} |")
    else:
        lines.append("| GPU | none found by nvidia-smi |")
    lines += ["", "## Import smoke test", "", "| Module | Version | OK |", "|---|---|---|"]
    lines += [f"| {m} | {v} | {'yes' if ok else 'NO'} |" for m, v, ok in imports]

    # vLLM needs compute capability >= 7.0 (Kaggle T4 = 7.5; P100 and Quadro P4000 = 6.x).
    low = [g for g in found if g["compute_cap"] and float(g["compute_cap"]) < 7.0]
    if low:
        lines += ["", f"Note: {low[0]['name']} has compute capability {low[0]['compute_cap']}; "
                  "vLLM needs 7.0 or newer, so LLM serving runs on Kaggle T4s, not here."]

    out = ROOT / "reports" / f"stage0_env_{args.label}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out.relative_to(ROOT)}")
    if not all(ok for _, _, ok in imports):
        sys.exit(1)


if __name__ == "__main__":
    main()
