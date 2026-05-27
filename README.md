# PE-LLM4POI: Profile Enhanced Next-POI Recommendation with Large Language Models

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)

**PE-LLM4POI** (Profile Enhanced-LLM4POI) 是一个基于大语言模型的下一兴趣点（next-POI）推荐框架。在 LLM4POI 基线之上，将用户长期历史签到记录离线压缩为由自然语言摘要和模块化结构化字段组成的复合用户画像，注入双提示词解耦建模框架，通过 QLoRA 参数高效微调实现精准的下一 POI 预测。

> 上海大学本科毕业论文《基于大语言模型的下一兴趣点推荐》

## 方法概述

传统 POI 推荐方法依赖稠密向量表示，存在可解释性差、冷启动困难等问题。直接拼接长历史轨迹到 LLM 提示词中会导致上下文膨胀和推理成本过高。

PE-LLM4POI 提出**双提示词解耦建模框架**：

1. **离线画像构建**：使用 GPT-4o-mini 将用户长期签到历史压缩为复合画像（自然语言摘要 + Big Five 人格特质 + 偏好 + 作息 + 人口统计属性）
2. **双提示词注入**：系统提示词承载长期偏好画像，用户提示词承载当前短轨迹意图
3. **QLoRA 高效微调**：4-bit 量化 + LoRA 低秩适配，仅训练极少参数即可适配 next-POI 任务
4. **结构化输出约束**：约束模型输出为数字 POI ID，提高生成的可靠性与可控性

```
┌─────────────────────────────────────────────────────────┐
│                    离线阶段                               │
│  历史签到 ──► GPT-4o-mini ──► 用户画像 (JSON)             │
│                                  │                       │
│  原始轨迹 ──► create_sft_dataset ──► HF 数据集            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    训练阶段                               │
│  HF 数据集 ──► QLoRA SFT (Llama 1B/3B/8B) ──► LoRA 权重  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    推理阶段                               │
│  系统提示词: <<SYS>>用户画像 A<</SYS>>                     │
│  用户提示词: 当前轨迹 + "下一 POI id?"                     │
│        │                                                 │
│        ▼                                                 │
│  LoRA 模型 ──► POI id 预测                               │
└─────────────────────────────────────────────────────────┘
```

## 实验结果

在 FourSquare-NYC 和 FourSquare-TKY 两个公开数据集上的 Acc@1 结果：

| 方法 | NYC | TKY |
|------|-----|-----|
| LLM4POI (基线) | 10.15% | 6.40% |
| **PE-LLM4POI (Full Profile)** | **14.14%** | **9.76%** |
| 相对提升 | +39.3% | +52.5% |

多指标评估（全候选集）：

| 指标 | No Profile (NYC) | Full Profile (NYC) | No Profile (TKY) | Full Profile (TKY) |
|------|:---:|:---:|:---:|:---:|
| Acc@1 | 9.81 | 14.14 | 7.28 | 9.76 |
| Acc@5 | 25.79 | 31.38 | 19.08 | 23.83 |
| Acc@10 | 36.43 | 40.94 | 27.60 | 32.96 |
| MRR | 19.64 | 23.32 | 14.54 | 17.88 |

## 安装

```bash
# 克隆仓库
git clone https://github.com/XiaoCao001/PE-LLM4POI.git
cd PE-LLM4POI

# 安装依赖
pip install -r requirements.txt
```

**环境要求**: Python 3.10+, CUDA 12.x, RTX 3090/4090/5090 (32GB+ VRAM 推荐)

## 快速开始

### 1. 数据准备

原始数据使用 LLM4POI 格式（需放置在 `./LLM4POI/` 目录下），或直接从 HuggingFace 加载。

```bash
# 生成用户画像（需要 OpenAI API key）
python src/generate_user_profile.py \
    --dataset nyc \
    --dataset_id w11wo/FourSquare-NYC-POI

# 生成 POI reasoning（可选，用于 CoT 训练）
python src/generate_poi_reasoning.py \
    --dataset nyc \
    --dataset_id w11wo/FourSquare-NYC-POI

# 创建 SFT 数据集
python src/create_sft_dataset.py \
    --dataset nyc \
    --dataset_id w11wo/FourSquare-NYC-POI
# 添加 --use_cot 生成 chain-of-thought 格式
```

### 2. 训练

```bash
# 单卡 QLoRA 训练（RTX 5090 32GB）
python src/train_sft_qlora_fsdp.py \
    --model_checkpoint "/path/to/Llama-3.2-1b" \
    --dataset_path "/path/to/hf_datasets/FourSquare-NYC-POI" \
    --max_length 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --output_dir "./outputs/nyc-1b"

# 多卡 FSDP 训练
bash train_qlora_fsdp.sh
```

### 3. 评估

```bash
# 标准评估
python src/eval_next_poi.py \
    --model_checkpoint "./outputs/nyc-1b" \
    --dataset_path "/path/to/hf_datasets/FourSquare-NYC-POI" \
    --apply_liger_kernel_to_llama

# 消融实验
python src/eval_next_poi.py \
    --model_checkpoint "./outputs/nyc-1b" \
    --dataset_path "..." \
    --no_profile                    # 移除用户画像
    --profile_only                  # 仅保留自然语言画像
    --structured_only               # 仅保留结构化字段
    --profile_length 100            # 截断画像长度

# CoT 评估
python src/eval_next_poi_cot.py \
    --model_checkpoint "./outputs/nyc-1b" \
    --dataset_id "w11wo/FourSquare-NYC-POI"

# 分析（冷启动 + 轨迹长度）
bash run_analysis.sh
```

## 项目结构

```
PE-LLM4POI/
├── src/
│   ├── gpt.py                          # GPT-4o-mini API 封装
│   ├── generate_user_profile.py        # 用户画像生成
│   ├── generate_poi_reasoning.py       # POI 推理意图生成（CoT）
│   ├── create_sft_dataset.py           # SFT 数据集构建
│   ├── train_sft_qlora_fsdp.py         # QLoRA 训练（单卡/多卡）
│   ├── eval_next_poi.py               # 标准评估（含消融实验）
│   ├── eval_next_poi_cot.py           # Chain-of-Thought 评估
│   ├── user_cold_start_analysis.py     # 冷启动用户分析
│   └── trajectory_length_analysis.py   # 轨迹长度影响分析
├── data/
│   ├── nyc/                            # NYC 数据集画像 & reasoning
│   └── tky/                            # TKY 数据集画像 & reasoning
├── outputs/                            # 训练好的 LoRA 权重
├── results/                            # 评估结果 JSON
├── notebooks/                          # Jupyter 分析笔记
├── requirements.txt                    # Python 依赖
├── train_qlora_fsdp.sh                # 多卡训练启动脚本
├── eval_next_poi.sh                   # 评估启动脚本
├── run_analysis.sh                    # 分析启动脚本
└── CLAUDE.md                          # Claude Code 指引
```

## 核心设计

- **双提示词解耦**: 系统提示词承载长期画像，用户提示词承载当前轨迹——避免上下文混合
- **QLoRA 高效微调**: 4-bit NF4 量化 + LoRA (r=8, alpha=16)，仅训练 q/k/v/o_proj
- **Completion-only loss**: 仅对 `[/INST]` 后的答案 token 计算损失
- **结构化输出**: 正则提取数字 ID，约束生成空间

## 模型变体

| 模型 | 基座 | 上下文 | 数据集 |
|------|------|--------|--------|
| nyc-1b | Llama-3.2-1B | 4096 | FourSquare-NYC |
| nyc-3b | Llama-3.2-3B | 4096 | FourSquare-NYC |
| nyc-3b-4096 | Llama-3.2-3B | 4096 | FourSquare-NYC |
| tky-1b | Llama-3.2-1B | 4096 | FourSquare-TKY |
| tky-3b | Llama-3.2-3B | 4096 | FourSquare-TKY |
| tky-3b-4096 | Llama-3.2-3B | 4096 | FourSquare-TKY |

## 引用

```bibtex
@thesis{cao2026pellm4poi,
  title     = {基于大语言模型的下一兴趣点推荐},
  author    = {曹智珑},
  school    = {上海大学},
  year      = {2026},
  type      = {本科毕业论文}
}
```

## 致谢

感谢 LLM4POI 项目和开源社区提供的基线代码与数据集。
