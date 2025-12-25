#!/usr/bin/env python3
"""
MusicGen 命令行工具
用法:
    python generate.py --audio1 song1.wav --audio2 song2.wav --output result.wav
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio_processor import load_and_preprocess
from src.generator import MusicGenerator
from src.utils import check_cuda, format_duration
from config.config import DEFAULT_MAX_TOKENS, DEFAULT_GUIDANCE_SCALE, OUTPUT_DIR


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MusicGen - AI 音乐生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  单音频生成:
    python generate.py --audio1 music.wav --output generated.wav
  
  双音频融合:
    python generate.py --audio1 song1.wav --audio2 song2.wav --output fusion.wav
  
  自定义参数:
    python generate.py --audio1 s1.wav --audio2 s2.wav --tokens 1024 --guidance 4.0
        """
    )
    
    # 必需参数
    parser.add_argument('--audio1', required=True, help='第一个音频文件路径')
    parser.add_argument('--audio2', help='第二个音频文件路径（可选，用于融合）')
    parser.add_argument('--output', help='输出文件路径（默认: data/output/generated.wav）')
    
    # 可选参数
    parser.add_argument('--tokens', type=int, default=DEFAULT_MAX_TOKENS,
                       help=f'生成长度（默认: {DEFAULT_MAX_TOKENS}）')
    parser.add_argument('--guidance', type=float, default=DEFAULT_GUIDANCE_SCALE,
                       help=f'引导系数（默认: {DEFAULT_GUIDANCE_SCALE}）')
    parser.add_argument('--model', default='facebook/musicgen-medium',
                       choices=['facebook/musicgen-small', 
                               'facebook/musicgen-medium',
                               'facebook/musicgen-large'],
                       help='模型大小')
    parser.add_argument('--prompt', help='自定义提示词')
    
    args = parser.parse_args()
    
    # 设置输出路径
    if args.output is None:
        output_path = Path(OUTPUT_DIR) / 'generated.wav'
    else:
        output_path = Path(args.output)
    
    print("=" * 60)
    print("MusicGen - AI 音乐生成")
    print("=" * 60)
    
    # 检查 CUDA
    cuda_info = check_cuda()
    if cuda_info['available']:
        print(f"✅ GPU: {cuda_info['device_name']}")
    else:
        print("⚠️  使用 CPU（速度较慢）")
    
    print(f"📁 模型: {args.model}")
    print(f"📊 参数: tokens={args.tokens}, guidance={args.guidance}")
    print()
    
    try:
        # 1. 加载第一个音频
        print(f"📂 加载音频 1: {args.audio1}")
        sr1, audio1 = load_and_preprocess(args.audio1)
        print(f"   ✅ 采样率: {sr1}Hz, 时长: {len(audio1)/sr1:.2f}秒")
        
        # 2. 加载第二个音频（如果提供）
        audio2 = None
        sr2 = None
        if args.audio2:
            print(f"📂 加载音频 2: {args.audio2}")
            sr2, audio2 = load_and_preprocess(args.audio2)
            print(f"   ✅ 采样率: {sr2}Hz, 时长: {len(audio2)/sr2:.2f}秒")
        
        print()
        
        # 3. 初始化生成器
        generator = MusicGenerator(model_name=args.model)
        generator.load_model()
        
        print()
        
        # 4. 生成音乐
        if audio2 is not None:
            # 双音频融合
            result = generator.generate_from_fusion(
                audio1, sr1,
                audio2, sr2,
                max_tokens=args.tokens,
                guidance_scale=args.guidance
            )
        else:
            # 单音频生成
            prompt = args.prompt or "创作一首类似风格的音乐"
            result = generator.generate_from_single(
                audio1, sr1,
                prompt=prompt,
                max_tokens=args.tokens,
                guidance_scale=args.guidance
            )
        
        # 5. 保存结果
        print()
        generator.save_audio(result, output_path)
        
        # 6. 显示结果信息
        result_duration = len(result) / generator.model.config.audio_encoder.sampling_rate
        print(f"📊 生成音频时长: {result_duration:.2f}秒")
        
        # 7. 清理
        generator.cleanup()
        
        print()
        print("=" * 60)
        print("✅ 任务完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


