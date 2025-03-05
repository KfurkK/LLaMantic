import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Set environment variables and paths
os.environ["HF_HOME"] = "D:/.cache2/huggingface"
model_id = "D:\.cache2\huggingface\hub\models--meta-llama--Llama-3.1-8B\snapshots\d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
adapter_path = "llama3.1-8B_model_1024_context_10000_train_1_bs_3_epochs_256_rank_512_alpha_2e-05_lr_cosine_scheduler_02-25_09-19-08\checkpoint-1000"

# Prepare quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

def prepare_model_for_eval():
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        local_files_only=True
    )
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Return model and tokenizer for evaluation
    return model, tokenizer

def run_evaluation():
    # Import lm-evaluation-harness components here to avoid conflicts
    from lm_eval import evaluator, tasks
    
    # Prepare model for evaluation
    model, tokenizer = prepare_model_for_eval()
    
    # Configure evaluation parameters
    eval_tasks = ["arc_easy", "arc_challenge", "hellaswag", "truthfulqa:mc", "winogrande", "gsm8k", "mmlu"]
    
    # Create a model wrapper that lm-evaluation-harness can use
    from lm_eval.models.huggingface import HFLM
    
    # Custom wrapper if needed
    class LoRAAdapter(HFLM):
        def __init__(self, model, tokenizer):
            super().__init__(pretrained=model, tokenizer=tokenizer)
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=LoRAAdapter(model, tokenizer),
        tasks=eval_tasks,
        batch_size=1,  # Adjust based on your GPU memory
        device="cuda:0",
        #no_cache=True,
        num_fewshot=0  # Adjust for few-shot evaluations
    )
    
    # Print and save results
    print(evaluator.make_table(results))
    
    # Save detailed results
    with open("lora_model_evaluation_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_evaluation()