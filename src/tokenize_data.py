from datasets import load_dataset
from transformers import AutoTokenizer


def tokenize_and_mask(example, tokenizer, max_length=2048):
    tokens = tokenizer(
        example["text"], 
        truncation=True,
        max_length=max_length,
        padding="max_length"  # or you can use dynamic padding later with a data collator
    )
    
    return tokens


if __name__ == "__main__":
        
    data_files = {
        "train": [
            # if you have more dataset files, add them here
            "dataset_LIMA/train_with_synthetic.jsonl"
        ]}

    dataset = load_dataset("json", data_files=data_files)

    # if you have the model locally pass the path if not pass the name of the model (see huggingface.com to get the name)
    model_id = "meta-llama/Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # If you have additional special tokens, add them:
    special_tokens = {
        "additional_special_tokens": [
            "<|begin_of_text|>", "<|end_of_text|>",
            "<|start_header_id|>", "<|end_header_id|>",
            "<|eot_id|>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    print("Updated special tokens:")
    print(tokenizer.special_tokens_map,"\n")

    tokenized_dataset_train = dataset["train"].map(
        lambda x: tokenize_and_mask(x, tokenizer),
        batched=True,
        batch_size=256,
        remove_columns=dataset["train"].column_names
    )

    # save tokenizer
    tokenizer.save_pretrained("tokenizer_v3_LIMA_2048")

    # save tokenized dataset
    tokenized_dataset_train.save_to_disk("tokenized_LIMA_2048")

