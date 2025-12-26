# MusicGen AI 音乐生成项目

基于 Meta MusicGen 的音乐风格学习和生成工具。

## 功能特性

- 单音频风格学习和生成
- 双音频融合生成
- 自动重采样（支持任意采样率的 WAV 文件）
- 可自定义生成参数
- 支持 GPU 加速
- 命令行界面
- 自动使用国内镜像（无需手动配置）

## 快速开始

### 1. 环境配置

**环境要求**：
- Python 3.12
- CUDA 12.4（推荐）
- 显存要求：small 4GB / medium 8GB / large 16GB

```bash
# 步骤 1: 创建 Conda 环境
conda create -n musicgen python=3.12 -y
conda activate musicgen

# 步骤 2: 安装 PyTorch（CUDA 12.4）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 步骤 3: 验证 PyTorch 安装（版本需要 >= 2.6）
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 步骤 4: 安装其他依赖
pip install -r requirements.txt
```

### 2. 基本使用

```bash
# 单音频生成（10秒）
python scripts/generate.py --audio1 your_music.wav

# 双音频融合
python scripts/generate.py --audio1 music1.wav --audio2 music2.wav

# 生成更长的音频（20秒）
python scripts/generate.py --audio1 music.wav --tokens 1024

# 使用更大的模型（质量更好）
python scripts/generate.py --audio1 music.wav --model facebook/musicgen-large
```

**注意**：
- 首次运行会自动下载模型，请确保网络畅通
- 程序会自动使用国内镜像（hf-mirror.com），无需手动设置
- 支持任意采样率的 WAV 文件，会自动重采样到 32000Hz

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
| `--tokens` | 生成长度 | 512 |
| `--guidance` | 引导系数 | 3.0 |
| `--model` | 模型大小 | medium |
| `--prompt` | 自定义提示词 | 自动 |

### tokens 与时长对应

| tokens | 时长 | 显存需求 (medium) |
|--------|------|-------------------|
| 256 | 约 5 秒 | ~6GB |
| 512 | 约 10 秒 | ~8GB |
| 1024 | 约 20 秒 | ~12GB |
| 1536 | 约 30 秒 | ~16GB |

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

### 后台运行（SSH 断开后继续）

```bash
# 使用 tmux（推荐）
tmux new -s music
conda activate musicgen
python -u scripts/generate.py --audio1 music.wav
# 按 Ctrl+B，再按 D 分离会话
# 重新连接: tmux attach -t music

# 或使用 nohup
nohup env HF_ENDPOINT=https://hf-mirror.com python -u scripts/generate.py --audio1 music.wav > output.log 2>&1 &
tail -f output.log
```

## 模型选择

| 模型 | 显存 | 质量 | 速度 |
|------|------|------|------|
| facebook/musicgen-small | ~4GB | 一般 | 快 |
| facebook/musicgen-medium | ~8GB | 较好 | 中等 |
| facebook/musicgen-large | ~16GB | 最好 | 慢 |

## 故障排查

### PyTorch 版本错误

如果看到 `require users to upgrade torch to at least v2.6` 错误：

```bash
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### CUDA Out of Memory

- 使用更小的模型：`--model facebook/musicgen-small`
- 减少生成长度：`--tokens 256`

### 无法连接 HuggingFace

程序已内置国内镜像支持，会自动切换。如果仍有问题：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/generate.py --audio1 your_music.wav
```

### sympy 模块找不到

```bash
pip install sympy==1.13.1
```

## 许可证

MIT License
