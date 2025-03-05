import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback
import colorama
colorama.init()
import argparse
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datetime import datetime

class LLaMantic: # TODO improve naming
    def __init__(self, model_path: str, adapter_path: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = self.load_model(model_path, adapter_path)
        # Change according to your needs. self.prompt is the first prompt passed to the model.
        # Your queries will be appended to this prompt.
        self.prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>You are an AI created by KuntAI. You assist people with their needs and queries by staying in the context bound. Don't keep repeating yourself. AVOID breaking the CONTEXT BOUND and switching to other topics. State yourself clear and concise. Who won the World Cup 2022?<|eot_id|><|start_header_id|>assistant<|end_header_id|>Argentina won the 2022 FIFA World Cup by defeating France in the final match.<|eot_id|><|start_header_id|>user<|end_header_id|>Who founded Apple?<|eot_id|><|start_header_id|>assistant<|end_header_id|>Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.<|eot_id|><|start_header_id|>user<|end_header_id|>this is a multi turn q right?<|eot_id|><|start_header_id|>assistant<|end_header_id|>Yes, this is a multi-turn question because you're asking about different topics like sports and businesses.<|eot_id|><|start_header_id|>user<|end_header_id|>"""

    @staticmethod
    def load_model(model_path, adapter_path):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto", # use cuda GPU if available
            local_files_only=True,
        )
        model.load_adapter(adapter_path)

        return model
        

    def interact(self, max_length:int, top_k:int, temperature:float):
        while True:
            query = input("User>\n")
            if query.lower() == "q":
                print("exiting...")
                break

            self.prompt += f"{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            # Encode the prompt
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(self.model.device)

            # Generate response
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"],
                top_k=top_k,
                #top_p=0,
                temperature=temperature,
            )

            # Decode generated text
            generated_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            print(f"LLaMantic:\n{generated_text}", flush=True)

            # Update prompt history
            self.prompt += generated_text + "<|eot_id|><|start_header_id|>user<|end_header_id|>"

    def train_model(self, config, tokenizer):
        # we need this part in order to make the dataset's user part -100 when calculating the loss
        # so the model doesn't learn to generate the user part lookup
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
        instruction_template = "<|start_header_id|>user<|end_header_id|>"

        instruction_template_ids = tokenizer.encode(instruction_template, add_special_tokens=False) # USER
        response_template_ids = tokenizer.encode(response_template, add_special_tokens=False) # ASSISTANT


        # move this
        def process_example(example):
            input_ids = example["input_ids"]
            attention_mask = example["attention_mask"]
            
            if len(input_ids) > config["context_length"]:
                input_ids = input_ids[:config["context_length"]]
                attention_mask = attention_mask[:config["context_length"]]
                
                if "labels" in example:
                    example["labels"] = example["labels"][:config["context_length"]]
            
            example["input_ids"] = input_ids
            example["attention_mask"] = attention_mask
            
            return example
        # Load dataset and split.
        dataset = load_from_disk(config["tokenized_dataset_path"])
        dataset = dataset.train_test_split(test_size=config["val_size"], seed=config["seed"])
        # Process the datasets
        # set train size if dataset is too big
        train_dataset = dataset["train"] #.select(range(config["train_size"])) 
        train_dataset = train_dataset.map(process_example)

        eval_dataset = dataset["test"]#.select(range(config["eval_size"]))
        eval_dataset = eval_dataset.map(process_example)

        print("Training on length of:", len(train_dataset))
        print("Evaluating on length of:", len(eval_dataset))
        #print(train_dataset[0])


        model = self.load_model(config["model_path"], config["adapter_path"])

        # Attach the PEFT adapter for quantized models.
        lora_config = LoraConfig(
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, lora_config)
        model.train()
        #e_gradient_checkpointing()

        # Initialize the data collator.
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids, 
            instruction_template=instruction_template_ids, 
            tokenizer=tokenizer
        )

        run_name = (
            f"{config['model']}_model_"
            f"{config['context_length']}_context_"
            f"{config['train_size']}_train_"
            f"{config['train_bs']}_bs_"
            f"{config['epochs']}_epochs_"
            f"{config['lora_rank']}_rank_"
            f"{config['lora_alpha']}_alpha_"
            f"{config['lr']}_lr_"
            f"{config['scheduler']}_scheduler_"
            f"{datetime.now().strftime('%m-%d_%H-%M-%S')}"
        )
        config["save_path"] = run_name

        os.makedirs(run_name, exist_ok=True)
        with open(f"{run_name}/config.txt", "w") as f:
            f.write(str(config))

        # Training arguments.
        args = SFTConfig(
            output_dir=config["save_path"],
            num_train_epochs=config["epochs"],
            evaluation_strategy="steps",
            eval_steps=config["eval_steps"],
            log_level="info",
            logging_steps=config["logging_steps"],
            per_device_train_batch_size=config["train_bs"],
            per_device_eval_batch_size=config["eval_bs"],
            gradient_accumulation_steps=1,
            save_strategy="best",
            save_steps=config["save_steps"],
            save_total_limit=config["keep_model_nums"],
            learning_rate=config["lr"],
            warmup_ratio=config["warmup_r"], # just trying
            fp16=True,
            lr_scheduler_type="cosine",
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            report_to="wandb",
            run_name=run_name,
        )

        trainer = SFTTrainer(
            model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=args,
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=config["es_patience"])]
        )

        trainer.can_return_loss = True  # Ensures the trainer returns val loss 
        trainer.train()


    def __str__(self):
        return "LLaMantic"
    
    @property
    def func(self):
        return self.model
    
    @func.setter
    def func(self, model):
        self.model = model
    

    
if __name__ == "__main__":
    # run here for inference. 
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, 
                           default="D:\.cache2\huggingface\hub\models--meta-llama--Llama-3.1-8B\snapshots\d04e592bb4f6aa9cfee91e2e20afa771667e1d4b")
    argparser.add_argument("--adapter_path", type=str, 
                           default="llama3.1-8B_model_1024_context_10000_train_1_bs_3_epochs_256_rank_512_alpha_2e-05_lr_cosine_scheduler_02-25_09-19-08\checkpoint-1000")
    argparser.add_argument("--max_length", type=int, default=2048)
    argparser.add_argument("--top_k", type=int, default=10) # options to choose from
    argparser.add_argument("--temperature", type=float, default=0.95) 
    
    args = argparser.parse_args()

    print("Args:\n", args)

    LLM = LLaMantic(args.model_path,
                    args.adapter_path)
    LLM.interact(args.max_length, args.top_k, args.temperature) # interaction/while loop

    
