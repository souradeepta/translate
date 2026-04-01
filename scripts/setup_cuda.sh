#!/usr/bin/env bash
# Verify WSL2 CUDA environment and install PyTorch for RTX 5050 (Blackwell).
# Run this BEFORE pip install -r requirements.txt

set -e

echo "=== WSL2 CUDA Environment Check ==="
echo ""

# 1. Check nvidia-smi
echo "[1] nvidia-smi:"
if nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null; then
    echo "    OK"
else
    echo "    FAIL — NVIDIA driver not visible. Is this WSL2? Run 'nvidia-smi' in Windows first."
    exit 1
fi
echo ""

# 2. Check CUDA toolkit
echo "[2] CUDA toolkit (nvcc):"
if nvcc --version 2>/dev/null | grep "release"; then
    echo "    OK"
else
    echo "    WARN — nvcc not found. Install: sudo apt install cuda-toolkit-12-8"
fi
echo ""

# 3. Check WSL2 libcuda stub
echo "[3] WSL2 libcuda stub:"
if ls /usr/lib/wsl/lib/libcuda.so.1 2>/dev/null; then
    echo "    OK"
else
    echo "    FAIL — /usr/lib/wsl/lib/libcuda.so.1 not found. Not in WSL2?"
fi
echo ""

# 4. Check LD_LIBRARY_PATH
echo "[4] LD_LIBRARY_PATH includes WSL2 lib:"
if echo "$LD_LIBRARY_PATH" | grep -q "/usr/lib/wsl/lib"; then
    echo "    OK"
else
    echo "    WARN — Add to ~/.bashrc:"
    echo "      export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\$LD_LIBRARY_PATH"
fi
echo ""

# 5. Check HuggingFace cache location
echo "[5] HuggingFace cache (should be on Linux filesystem, not /mnt/c):"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
if echo "$hf_home" | grep -q "^/mnt/c"; then
    echo "    WARN — HF_HOME is on Windows filesystem ($hf_home). Downloads will be slow."
    echo "    Fix: export HF_HOME=/home/$USER/.cache/huggingface"
else
    echo "    OK ($hf_home)"
fi
echo ""

# 6. Install PyTorch for CUDA 12.8 (Blackwell sm_100)
echo "[6] Installing PyTorch for CUDA 12.8 (RTX 5050 / Blackwell)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
echo ""

# 7. Verify PyTorch sees CUDA
echo "[7] Verifying PyTorch CUDA:"
python3 -c "
import torch
avail = torch.cuda.is_available()
print(f'    CUDA available: {avail}')
if avail:
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory // 1024**2
    print(f'    Device: {name}')
    print(f'    Compute capability: sm_{cap[0]}{cap[1]}')
    print(f'    VRAM: {vram} MiB')
    t = torch.tensor([1.0]).cuda()
    print(f'    CUDA kernel test: OK')
else:
    print('    WARN — CUDA not available. Check LD_LIBRARY_PATH.')
"
echo ""
echo "=== Setup complete. Next: pip install -r requirements-dev.txt ==="
