import warnings
warnings.filterwarnings("ignore")
from transformers import AutoTokenizer
from models import LLaMantic
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--model", type=str, default="llama3.1-8B", help="Model name")
    parser.add_argument("--train_size", type=int, default=10000, help="Training size")
    parser.add_argument("--eval_size", type=int, default=1000, help="Evaluation size")
    parser.add_argument("--train_bs", type=int, default=1, help="Training batch size")
    parser.add_argument("--eval_bs", type=int, default=1, help="Evaluation batch size")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lora_rank", type=int, default=256, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=512, help="LoRA alpha")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--scheduler", type=str, default="cosine", help="Scheduler type")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps")
    parser.add_argument("--keep_model_nums", type=int, default=2, help="Number of models to keep")
    parser.add_argument("--save_steps", type=int, default=200, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer_v3_LIMA_2048", help="Path to the tokenizer")
    parser.add_argument("--dataset_path", type=str, default="LIMA_2048", help="Path to the tokenized dataset")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size (ratio)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducability")
    return parser.parse_args()

def load_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return tokenizer

if __name__ == "__main__":

    args = parse_args()

    config = {
        "model": args.model,
        "train_size": args.train_size,
        "eval_size": args.eval_size,
        "train_bs": args.train_bs,
        "eval_bs": args.eval_bs,
        "context_length": args.context_length,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "eval_steps": args.eval_steps,
        "keep_model_nums": args.keep_model_nums,
        "save_steps": args.save_steps,
        "logging_steps": args.logging_steps,
        "tokenizer_path": args.tokenizer_path,
        "tokenized_dataset_path": args.dataset_path,
    }


    # Initialize the tokenizer
    tokenizer = load_tokenizer(config["tokenizer_path"])
    print("Starting training with config:\n", config)
    
    
    # for training adapter path is not important
    LLM = LLaMantic(config["model_path"], config["adapter_path"]) 
    LLM.train_model(tokenizer, config, tokenizer)

