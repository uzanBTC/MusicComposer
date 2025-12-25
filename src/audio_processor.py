"""
音频处理模块
负责音频的加载、验证、预处理
"""
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Tuple, Optional


class AudioProcessor:
    """音频处理器"""
    
    @staticmethod
    def load_audio(filepath: str) -> Tuple[int, np.ndarray]:
        """
        加载音频文件
        
        Args:
            filepath: 音频文件路径
            
        Returns:
            (sample_rate, audio_data): 采样率和音频数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"音频文件不存在: {filepath}")
        
        if filepath.suffix.lower() != '.wav':
            raise ValueError(f"仅支持 WAV 格式，当前文件: {filepath.suffix}")
        
        try:
            sample_rate, audio_data = wavfile.read(filepath)
            return sample_rate, audio_data
        except Exception as e:
            raise ValueError(f"读取音频失败: {e}")
    
    @staticmethod
    def validate_audio(audio_data: np.ndarray, sample_rate: int) -> bool:
        """
        验证音频数据有效性
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            是否有效
        """
        # 检查是否为空
        if audio_data is None or len(audio_data) == 0:
            return False
        
        # 检查采样率
        if sample_rate <= 0:
            return False
        
        # 检查时长（至少 1 秒）
        duration = len(audio_data) / sample_rate
        if duration < 1.0:
            return False
        
        return True
    
    @staticmethod
    def preprocess_audio(audio_data: np.ndarray) -> np.ndarray:
        """
        预处理音频
        1. 转换为单声道（如果是立体声）
        2. 归一化到 [-1, 1]
        
        Args:
            audio_data: 原始音频数据
            
        Returns:
            预处理后的音频数据
        """
        # 转单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # 转换为 float32
        audio_data = audio_data.astype(np.float32)
        
        # 归一化
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    @staticmethod
    def get_duration(audio_data: np.ndarray, sample_rate: int) -> float:
        """
        获取音频时长（秒）
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            
        Returns:
            时长（秒）
        """
        return len(audio_data) / sample_rate


# 便捷函数
def load_and_preprocess(filepath: str) -> Tuple[int, np.ndarray]:
    """
    加载并预处理音频（一步完成）
    
    Args:
        filepath: 音频文件路径
        
    Returns:
        (sample_rate, processed_audio): 采样率和处理后的音频
    """
    processor = AudioProcessor()
    sample_rate, audio_data = processor.load_audio(filepath)
    
    if not processor.validate_audio(audio_data, sample_rate):
        raise ValueError("音频验证失败")
    
    processed = processor.preprocess_audio(audio_data)
    return sample_rate, processed


