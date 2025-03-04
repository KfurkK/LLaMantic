# LLaMantic - Developing...🚧

Fine tuning procedure of a plain text-generation model into a reasoning one.
LIMA dataset is used in order to fine tune the model for instruction following.

## Fine-tuning for Instruction awareness
The steps below indicate how you can use your own dataset to fine tune a plain text generation model (preferably LLaMa).
- first run "clean_data.py" you'll get the desired format in a jsonl file with the contents of the dataset. Dataset is not too big so it's a quite fast process.
- second run tokenize.py in order to tokenize the dataset.
now the dataset is ready to train the model and train the model with train.py

#### TODO
- Proper evaluation/Benchmarking.
- System prompt-containing dataset.
- Available fine-tuned checkpoints.
- further training strategies: RLHF, RLAI etc..
