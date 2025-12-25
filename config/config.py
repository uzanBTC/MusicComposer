"""
项目配置文件
所有可配置参数集中管理
"""

# 模型配置
MODEL_NAME = "facebook/musicgen-small"  # 可选: small, medium, large
DEVICE = "cuda"  # 自动检测会在运行时确定

# 生成参数
DEFAULT_MAX_TOKENS = 512  # 约 10 秒
DEFAULT_GUIDANCE_SCALE = 3.0
DEFAULT_SAMPLE_RATE = 32000

# 路径配置
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"

# 提示词模板
DEFAULT_PROMPT_SINGLE = "创作一首类似风格的音乐"
DEFAULT_PROMPT_FUSION = "融合两首音乐的风格，创作新的作品"
PROMPT_STEP1 = "学习第一首歌的旋律和节奏"
PROMPT_STEP2 = "融入第二首歌的情绪和乐器特色"


