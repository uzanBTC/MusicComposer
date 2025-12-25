"""
工具函数模块
"""
import torch
from typing import Optional


def check_cuda() -> dict:
    """
    检查 CUDA 环境
    
    Returns:
        CUDA 信息字典
    """
    info = {
        'available': torch.cuda.is_available(),
        'device_count': 0,
        'device_name': None,
        'cuda_version': None
    }
    
    if info['available']:
        info['device_count'] = torch.cuda.device_count()
        info['device_name'] = torch.cuda.get_device_name(0)
        info['cuda_version'] = torch.version.cuda
    
    return info


def get_device() -> str:
    """
    获取计算设备
    
    Returns:
        'cuda' 或 'cpu'
    """
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


def format_duration(seconds: float) -> str:
    """
    格式化时长显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化后的字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}秒"
    
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}分{secs:.2f}秒"


def print_gpu_memory():
    """打印 GPU 内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"GPU 内存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")


