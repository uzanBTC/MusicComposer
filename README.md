# MusicGen AI 音乐生成项目

基于 Meta MusicGen 的音乐风格学习和生成工具。

## 功能特性

- 单音频风格学习和生成
- 双音频融合生成
- 可自定义生成参数
- 支持 GPU 加速
- 命令行界面

## 快速开始

### 1. 环境配置

**环境要求**：
- Python 3.12
- CUDA 12.x（推荐，GPU 加速）
- 8GB+ GPU 显存（使用 medium 模型）

```bash
# 步骤 1: 创建 Conda 环境
conda create -n musicgen python=3.12 -y
conda activate musicgen

# 步骤 2: 安装 PyTorch（使用 pip，版本 >= 2.6）
# GPU 版本（推荐）：
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU 版本（如果没有 GPU）：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 步骤 3: 验证 PyTorch 安装（版本需要 >= 2.6）
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"

# 步骤 4: 安装其他依赖
pip install -r requirements.txt
```

### 2. 基本使用

```bash
# 单音频生成
python scripts/generate.py --audio1 your_music.wav

# 双音频融合
python scripts/generate.py --audio1 music1.wav --audio2 music2.wav

# 自定义参数
python scripts/generate.py --audio1 s1.wav --audio2 s2.wav --tokens 1024 --guidance 4.0 --output result.wav
```

**注意**：首次运行会自动下载模型（medium 约 8GB），请确保网络畅通。

如果无法访问 HuggingFace，程序会自动使用国内镜像（hf-mirror.com）。

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

### 使用小模型（更快，显存要求更低）

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
4. **显存要求**：
   - small: 约 4GB
   - medium: 约 8GB
   - large: 约 16GB

## 故障排查

### PyTorch 版本错误：需要升级到 v2.6

如果看到 `require users to upgrade torch to at least v2.6` 错误：

```bash
# 升级 PyTorch
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 验证版本（需要 >= 2.6）
python -c "import torch; print(torch.__version__)"
```

### PyTorch 导入错误：undefined symbol: iJIT_NotifyEvent

这个错误通常是由于 MKL 版本冲突导致的（MKL 2024.1+ 与 PyTorch 不兼容）。

**解决方案**：使用 pip 安装 PyTorch（而不是 conda）

```bash
# 1. 完全卸载
pip uninstall torch torchvision torchaudio -y
conda remove pytorch torchvision torchaudio -y

# 2. 使用 pip 重新安装（pip 版本静态链接 MKL，不受影响）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. 验证
python -c "import torch; print('OK')"
```

### 无法连接 HuggingFace

程序已内置国内镜像支持，会自动切换。如果仍然有问题，手动设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/generate.py --audio1 your_music.wav
```

### CUDA Out of Memory

- 使用更小的模型：`--model facebook/musicgen-small`
- 减少生成长度：`--tokens 256`

### sympy 模块找不到

```bash
pip install sympy==1.13.1
```

## 许可证

MIT License
