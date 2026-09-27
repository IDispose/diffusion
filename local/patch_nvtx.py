"""Make SE3Transformer's NVTX profiler markers a no-op on CPU-only torch builds.

NVIDIA's se3_transformer does `from torch.cuda.nvtx import range as nvtx_range` in four
modules, and on a CPU-only torch that raises "NVTX functions not installed" partway
through the forward pass. The markers only annotate profiles, so a no-op is equivalent.

Run with the interpreter of the environment to patch: `rfdiff-env/bin/python local/patch_nvtx.py`
"""
import pathlib
import sysconfig

OLD = "from torch.cuda.nvtx import range as nvtx_range\n"
NEW = """import contextlib as _contextlib, torch as _torch
if _torch.cuda.is_available():
    from torch.cuda.nvtx import range as nvtx_range
else:
    @_contextlib.contextmanager
    def nvtx_range(*_a, **_k):
        yield
"""

root = pathlib.Path(sysconfig.get_paths()["purelib"]) / "se3_transformer"
patched = 0
for path in root.rglob("*.py"):
    text = path.read_text()
    if OLD in text:
        path.write_text(text.replace(OLD, NEW))
        patched += 1
print(f"    patched {patched} file(s) under {root}")
