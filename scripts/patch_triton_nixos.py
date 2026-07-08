#!/usr/bin/env python3
"""Patch the installed Triton wheel so it runs on NixOS.

NixOS breaks three assumptions Triton makes:

1. Triton ships a bundled `ptxas` ELF binary in
   `triton/backends/nvidia/bin/ptxas`. NixOS refuses to run generic-Linux
   dynamically-linked binaries (no /lib64/ld-linux-x86-64.so.2 interpreter).
   Run-time env var `TRITON_PTXAS_PATH` (set by run_train.sh) points Triton at
   the system ptxas instead, so nothing to patch here.

2. The CUDA 12.x ptxas on NixOS only accepts PTX up to 8.8, but Triton's
   `ptx_get_version()` naively maps CUDA 12.9 -> PTX 8.9, which ptxas rejects
   with `Unsupported .version 8.9; current version is '8.8'`. We cap the
   returned version at 8.8.

3. Triton's `libcuda_dirs()` hardcodes `/sbin/ldconfig`, which does not exist
   on NixOS. We make it fall back to whatever `ldconfig` is on PATH and
   tolerate a failing `ldconfig -p` (the TRITON_LIBCUDA_PATH env var and the
   LD_LIBRARY_PATH scan still work).

Re-run this after every `uv pip install` / venv rebuild.
"""
from __future__ import annotations

import pathlib
import sys


def _find_triton_nvidia_compiler() -> pathlib.Path:
    import triton  # noqa: F401  (locates the package)

    base = pathlib.Path(triton.__file__).resolve().parent
    p = base / "backends" / "nvidia" / "compiler.py"
    if not p.exists():
        raise SystemExit(f"cannot find triton nvidia compiler at {p}")
    return p


def _find_triton_nvidia_driver() -> pathlib.Path:
    import triton  # noqa: F401

    base = pathlib.Path(triton.__file__).resolve().parent
    p = base / "backends" / "nvidia" / "driver.py"
    if not p.exists():
        raise SystemExit(f"cannot find triton nvidia driver at {p}")
    return p


def patch_ptx_version(path: pathlib.Path) -> None:
    src = path.read_text()
    marker = "return min(80 + minor, 88)"
    if marker in src:
        print(f"[triton-patch] ptx_get_version cap already present: {path}")
        return
    target = "        if major == 12:\n            return 80 + minor"
    replacement = (
        "        if major == 12:\n"
        "            # NVIDIA ptxas from CUDA 12.x only accepts PTX up to 8.8\n"
        "            # even though the CUDA runtime is 12.9. Cap it.\n"
        "            return min(80 + minor, 88)"
    )
    if target not in src:
        raise SystemExit(
            f"could not find expected ptx_get_version block in {path}; "
            "triton version may have changed, patch manually"
        )
    path.write_text(src.replace(target, replacement))
    print(f"[triton-patch] capped ptx_get_version at 8.8: {path}")


def patch_libcuda_dirs(path: pathlib.Path) -> None:
    src = path.read_text()
    if "shutil.which" in src:
        print(f"[triton-patch] libcuda_dirs ldconfig fallback already present: {path}")
        return
    target = '    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()'
    replacement = (
        "    # NixOS has no /sbin/ldconfig; fall back to whatever ldconfig is on\n"
        "    # PATH (e.g. /run/current-system/sw/bin/ldconfig) and tolerate a\n"
        "    # failing `ldconfig -p` (TRITON_LIBCUDA_PATH / LD_LIBRARY_PATH still\n"
        "    # work as fallbacks).\n"
        "    import shutil\n"
        '    _ldconfig = "/sbin/ldconfig" if os.path.exists("/sbin/ldconfig") else shutil.which("ldconfig")\n'
        '    libs = ""\n'
        "    if _ldconfig:\n"
        "        try:\n"
        '            libs = subprocess.check_output([_ldconfig, "-p"], stderr=subprocess.DEVNULL).decode()\n'
        "        except Exception:\n"
        '            libs = ""'
    )
    if target not in src:
        raise SystemExit(
            f"could not find expected libcuda_dirs ldconfig call in {path}; "
            "triton version may have changed, patch manually"
        )
    path.write_text(src.replace(target, replacement))
    print(f"[triton-patch] made libcuda_dirs ldconfig fallback NixOS-safe: {path}")


def main() -> int:
    try:
        import triton  # noqa: F401
    except ImportError:
        print("[triton-patch] triton is not installed; nothing to patch", file=sys.stderr)
        return 0

    compiler = _find_triton_nvidia_compiler()
    driver = _find_triton_nvidia_driver()
    patch_ptx_version(compiler)
    patch_libcuda_dirs(driver)
    print("[triton-patch] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
