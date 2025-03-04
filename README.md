# LLaMantic - Developing...🚧

Fine tuning procedure of a plain text-generation model into a reasoning one.
LIMA dataset is used in order to fine tune the model for instruction following.

-first run "clean_data.py" you'll get the desired format in a jsonl file with the contents of the dataset. Dataset is not too big so it's a quite fast process.
-second run tokenize.py in order to tokenize the dataset
now the dataset is ready to train the model and train the model with train.py

# TODO
-eval_.py 
even system prompts might improve the model
proper benchmarking!
checkpoint sharing.
further training strategies: RLHF, RLAI etc.
