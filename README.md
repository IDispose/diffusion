# diffusion

A fork of [ColabDesign](https://github.com/sokrypton/ColabDesign)'s RFdiffusion notebook
(`rf/examples/diffusion.ipynb`), fixed to run on today's Colab runtime, plus scripts to
run the same thing locally without Colab.

## Run it in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IDispose/diffusion/blob/main/rf/examples/diffusion.ipynb)

**Pick a GPU runtime first** (*Runtime → Change runtime type → T4 GPU*). The setup cell
stops immediately if it cannot see a GPU, rather than spending eight minutes installing
CUDA wheels a CPU runtime cannot use.

### `ModuleNotFoundError: No module named 'dgl'`

That error comes from the **pre-fix** version of the notebook, which imported
`inference.utils` (and through it `dgl`) into the Colab kernel. The fixed notebook never
imports `dgl` in the kernel at all, so if you see it you are running an old copy — a
"Copy of diffusion.ipynb" saved in your Drive, a tab left open from before, or Colab's
cached copy of the GitHub file. Open the current one and use a fresh runtime:

```
https://colab.research.google.com/github/IDispose/diffusion/blob/main/rf/examples/diffusion.ipynb?flush_cache=true
```

then *Runtime → Disconnect and delete runtime* before re-running the setup cell. A quick
way to tell the two apart: the fixed setup cell is titled
*setup **RFdiffusion** (~8 min, **GPU runtime required**)* and mentions
`/content/rfdiff-env`.

### Why the notebook needed fixing

Colab's hosted runtime is now **Python 3.13 with torch 2.11**. RFdiffusion depends on
NVIDIA's SE3Transformer, which depends on [DGL](https://www.dgl.ai/) — and DGL's last
release (2.4.0, Sep 2024) only publishes wheels for **CPython 3.8–3.12 built against
torch 2.4**. There is no cp313 wheel anywhere, so upstream's

```
pip install --no-dependencies dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
```

finds nothing at that index and silently falls back to an **ancient DGL from PyPI** — the
kind whose `dgl/utils.py` still does `from collections import Mapping`. That is the
`ImportError: cannot import name 'Mapping' from 'collections'` this notebook was hitting.

Monkey-patching around it does not work. `collections.Mapping = collections.abc.Mapping`
gets the old DGL to import, but SE3Transformer calls `dgl.ops.edge_softmax`,
`dgl.ops.e_dot_v` and `dgl.ops.copy_e_sum` — compiled graph kernels that the old package
simply does not contain. Stubbing `dgl.ops` out makes the import succeed and the forward
pass fail.

Colab has no Python-version switch (the old "fallback runtime version" command is gone),
so the fix is to stop asking Colab's interpreter to import DGL at all:

- The setup cell builds a **self-contained Python 3.11 environment** at
  `/content/rfdiff-env` with `uv`, and runs `RFdiffusion/run_inference.py` under that
  interpreter. The notebook kernel keeps Colab's Python and only handles uploads, contig
  bookkeeping and the 3D views — it never imports torch or DGL.
- `torch` in that environment is pinned to whatever version DGL's bundled
  `libgraphbolt_pytorch_*.so` was built against, detected at install time. Get this wrong
  and `import dgl` fails with *"Cannot find DGL C++ graphbolt library"*.
- DGL is installed with `--no-deps` (its metadata asks for an unpinned torch, which would
  pull the latest release and break that ABI match), then its real import-time
  dependencies are installed explicitly: `torchdata`, `pyyaml`, `packaging`, `setuptools`,
  `pydantic`, `pandas`, `networkx`, `psutil`, `requests`, `tqdm`.
- `from inference.utils import parse_pdb` is replaced by a small dgl-free equivalent in
  the notebook, since the real one imports `dgl` transitively. Everything downstream only
  reads `parsed_pdb["pdb_idx"]`; the replacement reproduces it exactly (verified
  identical on 6MRR and 5KQV).
- The setup cell verifies `dgl.ops.edge_softmax` / `e_dot_v` / `copy_e_sum` and
  `import se3_transformer` before you start a run, instead of failing halfway through one.

Two unrelated bugs in this fork's copy were also fixed: a stray `"` appended to the
`contigmap.contigs=` argument, which produced an unbalanced shell command, and a failed
run reporting itself as `FileNotFoundError: outputs/traj/..._pX0_traj.pdb`. RFdiffusion's
stderr now goes to `outputs/<name>/run.log`, whose tail is printed on failure.

### Two upstream assets have gone 404

`files.ipd.uw.edu/krypton/` no longer serves anything:

| asset | effect |
| --- | --- |
| `schedules.zip` | Only a cache of the IGSO(3) tables. RFdiffusion recomputes them on the first run (a few extra minutes) and caches them in `./schedules`. |
| `ananas` | Needed only by `symmetry="auto"`. Set `symmetry="cyclic"`/`"dihedral"` and `order` by hand, or drop an [AnAnaS](https://team.inria.fr/nano-d/software/ananas/) binary beside the notebook. |

The RFdiffusion checkpoints under `files.ipd.uw.edu/pub/RFdiffusion/` and the AlphaFold
params are both fine.

## Run it locally

Yes — the diffusion step runs locally, including CPU-only on Apple Silicon. Same Python
3.11 / DGL constraint as above; `local/setup.sh` handles it.

```sh
./local/setup.sh          # auto: CUDA wheels on Linux+NVIDIA, CPU otherwise
./local/run.sh inference.output_prefix=outputs/test 'contigmap.contigs=[100-100]' diffuser.T=50
```

`setup.sh` needs [`uv`](https://docs.astral.sh/uv/) and clones RFdiffusion, builds
`./rfdiff-env`, and downloads the three checkpoints (~1.5 GB). `run.sh` passes its
arguments straight through to `run_inference.py` as hydra overrides.

Measured on an M-series Mac (48 GB, CPU-only torch 2.3.0 + dgl 2.2.0, Python 3.11),
unconditional monomer at `diffuser.T=20`:

| length | wall time | per step |
| --- | --- | --- |
| 40 res | 0.21 min | ~0.6 s |
| 100 res | 0.80 min | ~2.4 s |
| 200 res | 2.64 min | ~7.9 s |

So CPU is fine for small monomers and gets expensive fast — cost grows faster than
linearly in length. A binder run against a few-hundred-residue target is better off on a
GPU.

Two things to know about the CPU path:

- `setup.sh` patches SE3Transformer's `from torch.cuda.nvtx import range as nvtx_range`
  (see `local/patch_nvtx.py`). On a CPU-only torch that raises *"NVTX functions not
  installed"* partway through the forward pass. The markers only annotate profiler
  traces, so a no-op context manager is equivalent.
- MPS is not an option: DGL's kernels are CPU/CUDA only.

`diffuser.T` must be ≥ 15 — RFdiffusion asserts this. If you pass
`inference.dump_pdb_path=...`, create the directory first; RFdiffusion does not.

### What does not run locally as-is

The notebook itself is Colab-specific — `google.colab.files` for upload/download, and
`/dev/shm` for the live trajectory view (macOS has no `/dev/shm`). The ProteinMPNN +
AlphaFold validation cell shells out to `colabdesign/rf/designability_test.py`, which
needs JAX; on Apple Silicon that means the CPU backend.

For a full local pipeline on Linux with an NVIDIA GPU, the
[official RFdiffusion repo](https://github.com/RosettaCommons/RFdiffusion) ships a Docker
image and a conda environment, which is a less fiddly starting point than this fork.
