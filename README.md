# LLaMantic - Developing...🚧

Fine tuning procedure of a plain text-generation model into a reasoning one.
LIMA dataset is used in order to fine tune the model for instruction following.

## Fine-tuning for Instruction awareness
The steps below indicate how you can use your own dataset to fine tune a plain text generation model (preferably LLaMa).
- First run "python clean_data.py" you'll get the desired format in a jsonl file with the contents of the dataset. Dataset is not too big so it's a quite fast process.
- Second run "python tokenize_data.py" in order to tokenize the dataset.
- Train the model with "python train.py --args".

#### TODO
- Proper evaluation/Benchmarking.
- System prompt-containing dataset.
- Available fine-tuned checkpoints.
- further training strategies: RLHF, RLAI etc..
