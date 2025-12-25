#!/bin/bash
# PyTorch 修复脚本
# 用于解决 "undefined symbol: iJIT_NotifyEvent" 错误

echo "=========================================="
echo "PyTorch 修复脚本"
echo "=========================================="
echo ""

# 检查是否在 conda 环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  警告: 未检测到 Conda 环境"
    echo "建议在 Conda 环境中运行此脚本"
    read -p "是否继续？(y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ 当前 Conda 环境: $CONDA_DEFAULT_ENV"
fi

echo ""
echo "步骤 1: 卸载现有 PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "步骤 2: 检测 CUDA 可用性..."
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1,2)
    if [ ! -z "$CUDA_VERSION" ]; then
        echo "✅ 检测到 CUDA: $CUDA_VERSION"
        echo "将安装 GPU 版本的 PyTorch"
        INSTALL_GPU=true
    else
        echo "⚠️  未检测到 CUDA 版本，将安装 CPU 版本"
        INSTALL_GPU=false
    fi
else
    echo "⚠️  未检测到 nvidia-smi，将安装 CPU 版本"
    INSTALL_GPU=false
fi

echo ""
echo "步骤 3: 安装 PyTorch..."

if [ "$INSTALL_GPU" = true ]; then
    echo "安装 GPU 版本 (CUDA 12.1)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "安装 CPU 版本..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "步骤 4: 验证安装..."
python -c "
import torch
print(f'✅ PyTorch 版本: {torch.__version__}')
print(f'✅ CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU 设备: {torch.cuda.get_device_name(0)}')
else:
    print('ℹ️  使用 CPU 模式')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ PyTorch 修复完成！"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ 验证失败，请检查错误信息"
    echo "=========================================="
    exit 1
fi

