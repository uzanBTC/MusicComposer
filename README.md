# MusicGen AI 音乐生成项目

基于 Meta MusicGen 的音乐风格学习和生成工具。

## 功能特性

- ✅ 单音频风格学习和生成
- ✅ 双音频融合生成
- ✅ 可自定义生成参数
- ✅ 支持 GPU 加速
- ✅ 命令行界面

## 快速开始

### 1. 环境配置

**重要提示**：请严格按照以下顺序安装，避免 PyTorch 版本冲突。

```bash
# 步骤 1: 创建 Conda 环境
conda create -n musicgen python=3.10 -y
conda activate musicgen

# 步骤 2: 安装 PyTorch（必须通过 conda 安装，不要用 pip）
# 如果有 GPU，使用 GPU 版本：
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 如果没有 GPU 或不确定，使用 CPU 版本：
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 步骤 3: 验证 PyTorch 安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

# 步骤 4: 安装其他依赖（requirements.txt 中不包含 torch）
pip install -r requirements.txt
```

**注意事项**：
- ⚠️ **不要**使用 `pip install torch`，这会导致与 conda 安装的版本冲突
- ✅ 先安装 PyTorch，再安装其他依赖
- ✅ 如果遇到导入错误，请查看下方的"故障排查"部分

### 2. 基本使用

```bash
# 单音频生成
python scripts/generate.py --audio1 your_music.wav

# 双音频融合
python scripts/generate.py --audio1 music1.wav --audio2 music2.wav

# 自定义参数
python scripts/generate.py --audio1 s1.wav --audio2 s2.wav --tokens 1024 --guidance 4.0 --output result.wav
```

## 项目结构

```
MusicComposer/
├── config/          # 配置文件
│   └── config.py
├── src/             # 源代码
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── generator.py
│   └── utils.py
├── data/            # 数据目录
│   ├── input/       # 输入音频
│   └── output/      # 生成结果
├── scripts/         # 可执行脚本
│   └── generate.py
├── requirements.txt # 依赖列表
└── README.md
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--audio1` | 第一个音频文件（必需） | - |
| `--audio2` | 第二个音频文件（可选，用于融合） | - |
| `--output` | 输出文件路径 | `data/output/generated.wav` |
| `--tokens` | 生成长度（约 10 秒 = 512） | 512 |
| `--guidance` | 引导系数 | 3.0 |
| `--model` | 模型大小（small/medium/large） | medium |
| `--prompt` | 自定义提示词 | 自动 |

## 使用示例

### 单音频生成

```bash
python scripts/generate.py --audio1 data/input/music1.wav --output data/output/result.wav
```

### 双音频融合

```bash
python scripts/generate.py \
    --audio1 data/input/song1.wav \
    --audio2 data/input/song2.wav \
    --output data/output/fusion.wav \
    --tokens 512 \
    --guidance 3.5
```

### 使用小模型（更快，但质量较低）

```bash
python scripts/generate.py \
    --audio1 music.wav \
    --model facebook/musicgen-small \
    --tokens 256
```

## 注意事项

1. **音频格式**：目前仅支持 WAV 格式
2. **音频时长**：输入音频至少需要 1 秒
3. **GPU 要求**：建议使用 GPU 加速，CPU 运行速度较慢
4. **内存要求**：medium 模型需要约 8GB GPU 内存

## 故障排查

### PyTorch 导入错误：undefined symbol: iJIT_NotifyEvent

这个错误通常是由于 PyTorch 与系统库版本冲突，或 pip/conda 混合安装导致的。

**快速修复（推荐）**：

```bash
# 1. 完全卸载所有 PyTorch 相关包
pip uninstall torch torchvision torchaudio -y
conda remove pytorch torchvision torchaudio -y

# 2. 清理缓存
conda clean --all -y

# 3. 重新通过 conda 安装（推荐使用 conda，更稳定）
# CPU 版本：
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 或 GPU 版本：
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. 验证安装
python -c "import torch; print('✅ PyTorch 安装成功')"
```

**如果仍然有问题，尝试修复 Intel MKL 库**：

```bash
# 安装/更新 Intel MKL（conda 方式，更稳定）
conda install mkl mkl-service intel-openmp -y

# 或者使用 pip（如果 conda 不可用）
pip install mkl mkl-service
```

**使用修复脚本**：

```bash
# 使用项目提供的修复脚本
python scripts/fix_pytorch.py
```

### CUDA Out of Memory

- 使用更小的模型：`--model facebook/musicgen-small`
- 减少生成长度：`--tokens 256`

### 生成速度慢

- 检查是否使用 GPU：`python -c "import torch; print(torch.cuda.is_available())"`
- 确保安装了 CUDA 版本的 PyTorch

## 许可证

MIT License

