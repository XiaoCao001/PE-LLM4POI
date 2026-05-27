import re
import json
import os

import torch
from argparse import ArgumentParser
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftConfig, PeftModel

# liger_kernel for faster inference (optional)
try:
    from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen2
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--base_model", type=str, default=None, help="Path to base model (auto-detected if not provided)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local dataset")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--apply_liger_kernel_to_llama", action="store_true")
    parser.add_argument("--apply_liger_kernel_to_qwen2", action="store_true")
    parser.add_argument("--use_quantization", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.1)
    parser.add_argument("--typical_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.176)
    parser.add_argument("--no_profile", action="store_true", help="Remove user profile from system prompt")
    parser.add_argument("--profile_only", action="store_true", help="Keep only user_profile text, remove structured info")
    parser.add_argument("--structured_only", action="store_true", help="Keep only structured info (traits/attrs/prefs/routines), remove user_profile text")
    parser.add_argument("--profile_length", type=int, default=None, help="Truncate user_profile text to specified number of words")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load local dataset
    dataset = load_from_disk(args.dataset_path)
    print(f"Dataset loaded: {len(dataset['test'])} test samples")

    # Load tokenizer from adapter path
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # NOTE: we've formatted the prompt to include the <s> token at the beginning of the prompt
    if hasattr(tokenizer, "add_bos_token") and tokenizer.add_bos_token:
        tokenizer.add_bos_token = False

    torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
    )

    # Get base model path from adapter config or use provided path
    peft_config = PeftConfig.from_pretrained(args.model_checkpoint)
    base_model_path = args.base_model if args.base_model else peft_config.base_model_name_or_path
    print(f"Loading base model from: {base_model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
        quantization_config=quantization_config if args.use_quantization else None,
        device_map="auto",
    )

    if args.apply_liger_kernel_to_llama and LIGER_AVAILABLE:
        apply_liger_kernel_to_llama()

    if args.apply_liger_kernel_to_qwen2 and LIGER_AVAILABLE:
        apply_liger_kernel_to_qwen2()

    print(f"Loading LoRA adapter from: {args.model_checkpoint}")
    model = PeftModel.from_pretrained(model, args.model_checkpoint)
    model.eval()

    generation_config = GenerationConfig(
        max_new_tokens=5,
        min_new_tokens=None,
        do_sample=True,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        typical_p=args.typical_p,
        repetition_penalty=args.repetition_penalty,
        num_return_sequences=1,
    )

    predictions, targets = [], []

    lines = dataset["test"]["llama_prompt"]
    for line in tqdm(lines):
        # split prompt with target POI
        # ex. prompt = <s>[INST] <<SYS>> ... [/INST] At 2013-01-03 10:13:27, user 1 will visit POI id
        # ex. target = 123. </s>
        prompt, target, _ = re.split(r"(\d+\.\s</s>)", line)
        target = re.sub(r"[^0-9]", "", target)  # remove non-numeric tokens
        
        # Remove user profile if --no_profile is set
        if args.no_profile:
            # Remove content between <<SYS>> and <</SYS>>
            prompt = re.sub(r'<<SYS>>.*?<</SYS>>', '<<SYS>>You are a user.<</SYS>>', prompt, flags=re.DOTALL)
        elif args.profile_only:
            # Keep only user_profile text (last paragraph starting with "User X is...")
            # Extract content between <<SYS>> and <</SYS>>
            sys_match = re.search(r'<<SYS>>(.*?)<</SYS>>', prompt, flags=re.DOTALL)
            if sys_match:
                sys_content = sys_match.group(1)
                # Find user_profile text (starts with "User" and continues to end)
                profile_match = re.search(r'(User \d+ is .*)', sys_content, flags=re.DOTALL)
                if profile_match:
                    user_profile = profile_match.group(1).strip()
                    # Extract user ID from first line
                    user_id_match = re.search(r'You are user (\d+)', sys_content)
                    user_id = user_id_match.group(1) if user_id_match else "unknown"
                    new_sys = f'<<SYS>>You are user {user_id}. {user_profile}<</SYS>>'
                    prompt = re.sub(r'<<SYS>>.*?<</SYS>>', new_sys, prompt, flags=re.DOTALL)
        elif args.structured_only:
            # Keep only structured info (remove user_profile text)
            # Extract content between <<SYS>> and <</SYS>>
            sys_match = re.search(r'<<SYS>>(.*?)<</SYS>>', prompt, flags=re.DOTALL)
            if sys_match:
                sys_content = sys_match.group(1)
                # Remove user_profile text (starts with "User X is...")
                structured_content = re.sub(r'User \d+ is .*', '', sys_content, flags=re.DOTALL).strip()
                prompt = re.sub(r'<<SYS>>.*?<</SYS>>', f'<<SYS>>{structured_content}<</SYS>>', prompt, flags=re.DOTALL)
        elif args.profile_length is not None:
            # Truncate user_profile text to specified number of words
            sys_match = re.search(r'<<SYS>>(.*?)<</SYS>>', prompt, flags=re.DOTALL)
            if sys_match:
                sys_content = sys_match.group(1)
                # Find user_profile text
                profile_match = re.search(r'(User \d+ is .*)', sys_content, flags=re.DOTALL)
                if profile_match:
                    user_profile = profile_match.group(1).strip()
                    words = user_profile.split()
                    # Truncate if needed (protect against overflow)
                    if len(words) > args.profile_length:
                        truncated_profile = ' '.join(words[:args.profile_length])
                        # Replace in sys_content
                        new_sys_content = re.sub(r'User \d+ is .*', truncated_profile, sys_content, flags=re.DOTALL)
                        prompt = re.sub(r'<<SYS>>.*?<</SYS>>', f'<<SYS>>{new_sys_content}<</SYS>>', prompt, flags=re.DOTALL)

        prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_token_length = prompt_input_ids.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(**prompt_input_ids, generation_config=generation_config)

        prediction = tokenizer.decode(outputs[0, prompt_token_length:], skip_special_tokens=True)
        prediction = re.sub(r"[^0-9]", "", prediction)  # remove non-numeric tokens

        predictions.append(prediction)
        targets.append(target)

    accuracy = accuracy_score(targets, predictions)

    model_id = args.model_checkpoint.split("/")[-1]
    dataset_id = args.dataset_path.split("/")[-1]
    
    # Add profile mode indicator to model_id
    profile_mode = "full"
    if args.no_profile:
        profile_mode = "no-profile"
        model_id = f"{model_id}-no-profile"
    elif args.profile_only:
        profile_mode = "profile-only"
        model_id = f"{model_id}-profile-only"
    elif args.structured_only:
        profile_mode = "structured-only"
        model_id = f"{model_id}-structured-only"
    elif args.profile_length is not None:
        profile_mode = f"len-{args.profile_length}"
        model_id = f"{model_id}-len-{args.profile_length}"

    result = {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "accuracy": accuracy,
        "total_samples": len(targets),
        "correct_predictions": sum(1 for p, t in zip(predictions, targets) if p == t),
        "config": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "profile_mode": profile_mode,
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{model_id}-{dataset_id}.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Correct: {result['correct_predictions']}/{result['total_samples']}")
    print(f"  Results saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
