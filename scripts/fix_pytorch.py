#!/usr/bin/env python3
"""
PyTorch 修复脚本（Python 版本）
用于解决 "undefined symbol: iJIT_NotifyEvent" 错误
"""
import subprocess
import sys
import os


def run_command(cmd, check=True):
    """运行命令"""
    print(f"执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ 错误: {result.stderr}")
        sys.exit(1)
    return result


def check_cuda():
    """检查 CUDA 是否可用"""
    try:
        result = run_command("nvidia-smi", check=False)
        if result.returncode == 0:
            # 尝试解析 CUDA 版本
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"✅ 检测到 CUDA")
                    return True
        return False
    except:
        return False


def main():
    print("=" * 50)
    print("PyTorch 修复脚本")
    print("=" * 50)
    print()
    
    # 检查 conda 环境
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"✅ 当前 Conda 环境: {conda_env}")
    else:
        print("⚠️  警告: 未检测到 Conda 环境")
        response = input("是否继续？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    print()
    print("步骤 1: 卸载现有 PyTorch...")
    run_command("pip uninstall torch torchvision torchaudio -y", check=False)
    
    print()
    print("步骤 2: 检测 CUDA 可用性...")
    has_cuda = check_cuda()
    
    print()
    print("步骤 3: 安装 PyTorch...")
    if has_cuda:
        print("安装 GPU 版本 (CUDA 12.1)...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("安装 CPU 版本...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print()
    print("步骤 4: 验证安装...")
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU 设备: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  使用 CPU 模式")
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("✅ PyTorch 修复完成！")
    print("=" * 50)


if __name__ == '__main__':
    main()

