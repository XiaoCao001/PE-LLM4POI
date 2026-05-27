import torch

from argparse import ArgumentParser
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# liger_kernel for faster training (optional)
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False

"""
单卡QLoRA训练脚本 - 适配RTX 5090 (32GB VRAM)

Usage:
python train_sft_qlora_fsdp.py \
    --model_checkpoint "/root/shared-nvme/Work/Llama-3.2-1b" \
    --dataset_path "/root/shared-nvme/Work/hf_datasets/FourSquare-NYC-POI" \
    --max_length 4096 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --gradient_checkpointing \
    --apply_liger_kernel_to_llama \
    --output_dir "./outputs/nyc-1b"
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="/root/shared-nvme/Work/Llama-3.2-1b")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_length", type=int, default=4096)  # 降低以节省显存
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--apply_liger_kernel_to_llama", action="store_true")
    parser.add_argument("--apply_liger_kernel_to_qwen2", action="store_true")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model_id = args.output_dir

    # 加载本地数据集
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # NOTE: we've formatted the prompt to include the <s> token at the beginning of the prompt
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        tokenizer.add_bos_token = False

    response_template = "[/INST]" if "llama-2" in args.model_checkpoint.lower() else " [/INST]"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    max_seq_length = args.max_length

    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    # 单卡加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        use_cache=True if args.gradient_checkpointing else False,
        attn_implementation="sdpa",  # alternatively use "flash_attention_2"
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map="auto",
    )

    if args.apply_liger_kernel_to_llama and LIGER_AVAILABLE:
        apply_liger_kernel_to_llama()
        print("Liger kernel applied for LLaMA")
    elif args.apply_liger_kernel_to_llama and not LIGER_AVAILABLE:
        print("Warning: liger_kernel not available, continuing without it")

    if args.apply_liger_kernel_to_qwen2 and LIGER_AVAILABLE:
        apply_liger_kernel_to_qwen2()

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=model_id,
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        bf16=True,
        dataloader_num_workers=4,
        num_train_epochs=args.num_epochs,
        optim="adamw_torch",
        report_to="tensorboard",
        logging_steps=10,
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        dataset_text_field="llama_prompt",
        max_seq_length=max_seq_length,
        data_collator=collator,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # Print trainable parameters
    trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(model_id)
    tokenizer.save_pretrained(model_id)
    print(f"\nTraining completed! Model saved to {model_id}")


if __name__ == "__main__":
    main()
