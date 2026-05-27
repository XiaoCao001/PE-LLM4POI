# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PE-LLM4POI (Profile Enhanced-LLM4POI) — 基于大语言模型的下一兴趣点推荐。在 LLM4POI 基线之上，使用 GPT-4o-mini 将用户长期历史签到记录离线压缩为由自然语言摘要和模块化结构化字段组成的复合用户画像，注入双提示词解耦建模框架（系统提示词承载长期偏好，用户提示词承载当前短轨迹），通过 QLoRA 参数高效微调 Llama 模型（1B/3B/8B）预测下一 POI。

实验在 FourSquare-NYC 和 FourSquare-TKY 两个公开数据集上进行，评价指标为 Acc@1、Acc@5、Acc@10、MRR。

## Key Commands

**安装依赖:**
```bash
pip install -r requirements.txt
```

**从签到数据生成用户画像（调用 GPT API）:**
```bash
python src/generate_user_profile.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
```

**生成 POI reasoning 用于 CoT 训练（调用 GPT API）:**
```bash
python src/generate_poi_reasoning.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
```

**创建 SFT 数据集（组合 QA pairs 与用户画像）:**
```bash
python src/create_sft_dataset.py --dataset nyc --dataset_id w11wo/FourSquare-NYC-POI
# 添加 --use_cot 以生成 chain-of-thought 格式
```

**训练（单卡 QLoRA，适配 RTX 5090 32GB）:**
```bash
python src/train_sft_qlora_fsdp.py \
    --model_checkpoint "/root/shared-nvme/Work/Llama-3.2-1b" \
    --dataset_path "/root/shared-nvme/Work/hf_datasets/FourSquare-NYC-POI" \
    --max_length 4096 --batch_size 1 --gradient_accumulation_steps 8 \
    --num_epochs 3 --gradient_checkpointing --apply_liger_kernel_to_llama \
    --output_dir "./outputs/nyc-1b"
```

**多卡训练（FSDP，2 GPUs）:**
```bash
bash train_qlora_fsdp.sh
```

**评估 next-POI 准确率:**
```bash
python src/eval_next_poi.py \
    --model_checkpoint "./outputs/nyc-1b" \
    --dataset_path "/root/shared-nvme/Work/hf_datasets/FourSquare-NYC-POI" \
    --apply_liger_kernel_to_llama
# 消融实验参数: --no_profile, --profile_only, --structured_only, --profile_length N
```

**CoT 评估:**
```bash
python src/eval_next_poi_cot.py \
    --model_checkpoint "./outputs/nyc-1b" \
    --dataset_id "w11wo/FourSquare-NYC-POI"
```

**运行分析套件（冷启动 + 轨迹长度分析）:**
```bash
bash run_analysis.sh
```

## Architecture

### 数据流水线

```
原始签到 CSV → generate_user_profile.py (GPT) → user_profiles/*.json
                                                → push 到 HF 作为 profiles 数据集
原始签到 CSV → generate_poi_reasoning.py (GPT) → poi_reasoning/*.json
QA pairs + profiles + reasoning → create_sft_dataset.py → HF 数据集（含 "llama_prompt" 字段）
```

### 训练流水线

```
HF 数据集 → train_sft_qlora_fsdp.py → outputs/{model}/ (LoRA adapter)
           (QLoRA: 4-bit NF4 量化 + LoRA r=8, target: q/k/v/o_proj)
           (SFTTrainer from TRL, DataCollatorForCompletionOnlyLM on "[/INST]")
```

### 评估流水线

```
LoRA adapter → eval_next_poi.py → results/*.json (Acc@1)
             → eval_next_poi_cot.py → results/*.json (CoT reasoning accuracy)
结果 → user_cold_start_analysis.py → 按用户活跃度分层的准确率
     → trajectory_length_analysis.py → 按轨迹长度分层的准确率
```

### 提示词格式（双提示词解耦建模）

标准格式: `<s>[INST] <<SYS>>{用户画像}<</SYS>> {当前轨迹 + 问题} [/INST] {poi_id} </s>`

CoT 格式: `<s>[INST] <<SYS>>{画像}<</SYS>> {instruction} [/INST]<think>{reasoning}</think><output>{answer}</output></s>`

系统提示词包含 GPT 生成的复合用户画像：结构化属性（年龄、性别、教育、社会经济水平）、Big Five 人格特质、偏好、常规作息，以及一段约 200 词的自然语言用户画像摘要。

### 核心设计决策

- **QLoRA 参数高效微调**: 仅对 q_proj, k_proj, v_proj, o_proj 进行 LoRA 微调（r=8, alpha=16），不进行全量微调。
- **Completion-only loss**: `DataCollatorForCompletionOnlyLM` 确保 loss 仅计算在 `[/INST]` 之后的答案 token 上。
- **单卡适配**: 训练脚本从原始 FSDP 多卡方案修改为支持单卡 RTX 5090 (32GB)，使用梯度累积。
- **GPT API**: 使用旧版 `openai.ChatCompletion.create` API（openai<1.0.0）。`gpt.py` 封装 `gpt-4o-mini`，同时用于用户画像生成和 POI 意图推理。
- **数据集托管在 HuggingFace Hub**: 原始数据和生成的 SFT 数据集均在 `w11wo` 组织下。
- **BOS token 处理**: 提示词模板已显式包含 `<s>`，故设置 `tokenizer.add_bos_token = False` 避免重复。

### 数据目录

- `data/{nyc,tky}/user_profiles/` — GPT 生成的用户画像 JSON（每个用户一个文件）
- `data/{nyc,tky}/poi_reasoning/` — GPT 生成的 POI 访问意图推理 JSON
- `data/{nyc,tky}/profile_similarities.json` — 画像对之间的相似度分数
- `outputs/` — 训练好的 LoRA adapter 及 checkpoint
- `results/` — 评估结果 JSON
- `notebooks/` — Jupyter notebooks，用于数据准备、分析和可视化

### 模型变体

基于 Llama 3.2（1B, 3B）和 Llama 3.1（8B），在两个城市数据集上训练：
- NYC (FourSquare-NYC-POI), TKY (FourSquare-TKY-POI)
- 分析脚本额外支持: CA, Moscow, SaoPaulo
- 部分 3B 变体使用 4096 上下文长度（`nyc-3b-4096`, `tky-3b-4096`）
