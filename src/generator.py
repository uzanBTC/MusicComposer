"""
音乐生成核心模块
负责模型加载和音乐生成
"""
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from pathlib import Path
from typing import Optional, Tuple
import time
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import (
    MODEL_NAME, 
    DEFAULT_MAX_TOKENS, 
    DEFAULT_GUIDANCE_SCALE,
    PROMPT_STEP1,
    PROMPT_STEP2
)
from src.utils import get_device


class MusicGenerator:
    """音乐生成器"""
    
    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            model_name: 模型名称
            device: 计算设备（None 则自动检测）
        """
        self.model_name = model_name
        self.device = device or get_device()
        self.processor = None
        self.model = None
        
    def load_model(self):
        """加载模型"""
        if self.model is not None:
            print("模型已加载，跳过")
            return
        
        print(f"正在加载模型: {self.model_name}")
        print(f"使用设备: {self.device}")
        
        start_time = time.time()
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        
        load_time = time.time() - start_time
        print(f"✅ 模型加载完成（耗时: {load_time:.2f}秒）")
        
    def generate_from_single(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    ) -> np.ndarray:
        """
        基于单个音频生成
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
            prompt: 文字提示
            max_tokens: 生成长度
            guidance_scale: 引导系数
            
        Returns:
            生成的音频数据
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        print(f"🎵 开始生成（提示: {prompt}）")
        
        # 准备输入
        inputs = self.processor(
            audio=audio_data,
            sampling_rate=sample_rate,
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        start_time = time.time()
        with torch.no_grad():
            audio_values = self.model.generate(
                **inputs,
                do_sample=True,
                guidance_scale=guidance_scale,
                max_new_tokens=max_tokens
            )
        
        gen_time = time.time() - start_time
        print(f"✅ 生成完成（耗时: {gen_time:.2f}秒）")
        
        # 转换为 numpy
        audio_values = audio_values.cpu().numpy()
        return audio_values[0, 0]
    
    def generate_from_fusion(
        self,
        audio1_data: np.ndarray,
        audio1_rate: int,
        audio2_data: np.ndarray,
        audio2_rate: int,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    ) -> np.ndarray:
        """
        融合两个音频生成（两步法）
        
        Args:
            audio1_data: 第一个音频数据
            audio1_rate: 第一个音频采样率
            audio2_data: 第二个音频数据
            audio2_rate: 第二个音频采样率
            max_tokens: 生成长度
            guidance_scale: 引导系数
            
        Returns:
            融合后的音频数据
        """
        print("=" * 50)
        print("两步融合生成")
        print("=" * 50)
        
        # 第一步：基于第一首歌生成
        print("\n第 1 步：学习第一首歌的风格")
        intermediate = self.generate_from_single(
            audio1_data,
            audio1_rate,
            PROMPT_STEP1,
            max_tokens=max_tokens // 2,  # 第一步生成一半长度
            guidance_scale=guidance_scale
        )
        
        # 第二步：融入第二首歌的特点
        print("\n第 2 步：融入第二首歌的特点")
        final = self.generate_from_single(
            intermediate,
            self.model.config.audio_encoder.sampling_rate,
            PROMPT_STEP2,
            max_tokens=max_tokens,
            guidance_scale=guidance_scale
        )
        
        print("\n" + "=" * 50)
        print("✅ 融合完成！")
        print("=" * 50)
        
        return final
    
    def save_audio(self, audio_data: np.ndarray, filepath: str):
        """
        保存音频
        
        Args:
            audio_data: 音频数据
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        sample_rate = self.model.config.audio_encoder.sampling_rate
        wavfile.write(filepath, rate=sample_rate, data=audio_data)
        
        print(f"💾 音频已保存: {filepath}")
    
    def cleanup(self):
        """清理 GPU 内存"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            print("🧹 GPU 缓存已清理")

