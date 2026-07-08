# Hardware Guide

## Tested Configuration

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 5050 Laptop (Blackwell, sm_120, 8 GB GDDR7) |
| CPU | AMD Ryzen 5 240 (Zen 4) |
| OS | WSL2 Ubuntu on Windows 11 |
| CUDA Driver | 595.79 |
| CUDA Toolkit | 12.8 (`/usr/local/cuda-12.8/`) |
| CTranslate2 | 4.7.1 |
| PyTorch | 2.6.0+cu124 |

---

## WSL2 CUDA Setup

### Required: LD_LIBRARY_PATH

Every process (including subprocesses) must have this set:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

Add to `~/.bashrc` to make permanent. Without this, `import torch` or `ctranslate2.get_cuda_device_count()` silently falls back to CPU.

### Verify CUDA is visible

```bash
python3 -c "
import torch, ctranslate2
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('CT2:', ctranslate2.__version__, '| devices:', ctranslate2.get_cuda_device_count())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0), '| sm:', torch.cuda.get_device_capability(0))
"
```

Expected output on this config:
```
PyTorch: 2.6.0+cu124 | CUDA: True
CT2: 4.7.1 | devices: 1
GPU: NVIDIA GeForce RTX 5050 Laptop GPU | sm: (12, 0)
```

### PyTorch Warning (sm_120)

PyTorch 2.6.0+cu124 does not include a pre-compiled kernel for sm_120 (Blackwell). It falls back to PTX JIT compilation, which works but produces this warning:

```
UserWarning: CUDA capability sm_120 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37, sm_50, ..., sm_90.
You may be able to run your PyTorch with current installation if you have CUDA toolkit ≥ 12.8.
```

This is **not an error** — CTranslate2 handles its own CUDA kernels and works correctly.

---

## RTX 5050 (sm_120) Specific Notes

### INT8 is not available

CTranslate2 INT8 operations (`int8`, `int8_float16`, `int8_float32`) fail with:

```
RuntimeError: CUBLAS_STATUS_NOT_SUPPORTED when calling cublasGemmEx(...)
```

Root cause: sm_120 + CUDA 12.4 cuBLAS does not support INT8 tensor core operations for the matrix sizes used in this model. This may be fixed in future CTranslate2 versions with sm_120 kernels.

**Workaround:** Use `float16` — same accuracy, ~10% higher VRAM vs INT8.

The `_best_compute_type()` method handles this automatically. **Never hardcode `compute_type="int8"`.**

### Compute type probe must be ≥ 15 tokens

Short probes (< ~15 tokens) do not trigger the problematic cuBLAS kernels and return false positives (INT8 appears to work). Only during a real 20+ token translation does the failure appear.

The probe sentence used: `"The quick brown fox jumps over the lazy dog near the river."` (~15 tokens post-SPM).

---

## AMD Ryzen 5 240 (Zen 4) — CPU Notes

### No AMX support

AMD CPUs do not have Intel AMX (Advanced Matrix Extensions). PyTorch nightly builds compiled with AMX produce a SIGBUS on AMD:

```
Bus error (core dumped)
```

on `import torch` because the binary tries to execute AMX instructions.

**Fix:** Always use stable PyTorch from the cu124 index:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Never use `pip install torch` (pulls latest nightly on some platforms) or `--pre` flag.

---

## Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUBLAS_STATUS_NOT_SUPPORTED` | INT8 on sm_120 + CUDA 12.4 | Use float16; `_best_compute_type()` handles this |
| "Ra Ra Ra Ra..." garbage output | Wrong NLLB source format | Source = `tokens + [</s>, src_lang]`, NOT `[src_lang, tokens...]` |
| `Bus error` on `import torch` | PyTorch nightly on AMD CPU (AMX instructions) | Use `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| Corrupt torch install | Concurrent pip processes | Kill all pip, `rm -rf .venv/lib/.../torch`, clean reinstall |
| `ModuleNotFoundError: torch` | Venv not activated | `source .venv/bin/activate` |
| `ctranslate2.get_cuda_device_count()` returns 0 | Missing WSL CUDA lib path | `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` |
| INT8 probe passes but real translation fails | Probe sentence too short | Probe must be ≥ 15 tokens to trigger cuBLAS matmuls |
| Slow HuggingFace downloads | HF_HOME on NTFS `/mnt/c/...` | `export HF_HOME=/home/$USER/.cache/huggingface` (Linux fs) |
| `ensurepip is not available` on venv creation | Missing python3.12-venv system package | Use `python3 -m venv .venv --without-pip` + curl bootstrap |
| `git add src/bn_en_translate/models/` silently ignored | Bare `models/` in .gitignore matches subdirs | Use `/models/` (root-anchored) in .gitignore |

---

## VRAM Budget (8 GB RTX 5050)

The measured VRAM budget below is encoded in code as
`bn_en_translate.config.MODEL_VRAM_MIB` and enforced at runtime by
`bn_en_translate.utils.cuda_check.ensure_vram_available()`, which raises a
clear `RuntimeError` before a second model would OOM mid-run (e.g. loading
Ollama for the polish pass while a translator is still resident). Keep the
numbers below byte-identical to `MODEL_VRAM_MIB` — update both together.

| Model | VRAM peak (MiB) | Source |
|---|---|---|
| `nllb-600m` | 2400 | monitor/observations.md 2026-04-10 (2,355 MiB measured) |
| `milmmt-46-1b` | 4100 | monitor/observations.md 2026-07 Task 3 (4,009 MiB standalone / 4,024 MiB sequential-after-nllb, post-batching) |
| `seamless-medium` | 4100 | monitor/observations.md 2026-04-10 (4,096 MiB measured) |
| `indictrans2-1b` | 3100 | estimated — model not yet downloaded (gated HF repo) |
| `ollama-qwen2.5:7b` | 4800 | monitor/observations.md 2026-04-10 (~4.7 GB) |
| `ollama-gemma3:12b` | 4700 | CLAUDE.md model reference table (~4.7 GB) |

Note: the `milmmt-46-1b` figure was revised upward from an earlier (stale)
3,379 MiB reading taken before batch_size=8 batching was enabled — the
2026-07 optimization pass measured a genuine increase to ~4.0 GB.

**Safe combination on 8 GB:** only `nllb-600m` + Ollama fits concurrently
(2400 + 4800 = 7200 MiB, within the ~7.5 GB usable ceiling). Every other
translator (`milmmt-46-1b`, `seamless-medium`, `indictrans2-1b`) must be
unloaded before starting the Ollama polish pass — the pipeline does this
automatically.

---

## .wslconfig Tuning

For optimal WSL2 GPU performance, create/edit `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=16GB          # limit RAM to half system RAM
processors=8         # cores for WSL
gpumemory=7GB        # allow WSL CUDA access to 7 of 8 GB VRAM
```

Restart WSL after changes: `wsl --shutdown` from PowerShell.

---

## CUDA Toolkit (Linux-side)

The toolkit is at `/usr/local/cuda-12.8/`. The Windows driver provides the CUDA runtime via `/usr/lib/wsl/lib/libcuda.so.1`.

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
```

Verify:
```bash
nvcc --version       # should show 12.8
nvidia-smi           # should show driver + GPU
```
