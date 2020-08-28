# Replication package for RoBERTa Model
## DATASET
DATASET folder contains all 12 datasets used to train and test RoBERTa models. The files are zipped to preserve the space.  
For each dataset there are 7 files:  
1. training_mask: mask tokens for the training set (with a special token <z> at the end)
2. training_masked_code: training method code with a special token <x> instead of the mask tokens
3. eval_mask: mask tokens for the evaluation set (with a special token <z> at the end)
4. eval_masked_code: evaluation method code with a special token <x> instead of the mask tokens
5. test_mask: mask tokens for the test set (with a special token <z> at the end)
6. test_masked_code: test method code with a special token <x> instead of the mask tokens
7. tokenizer_training: list of methods used to train the tokenizer

## PREDICTIONS
For each dataset we have two different files:
1. predictions.txt: RoBERTa predictions for each masked method in the test set
2. raw_data.csv: reported metrics (BLEU scores and Levenshtein distance) for each prediction (same record as the previous file)

## STATISTICAL ANALYSIS
For each dataset we've reported a summary file to compare RoBERTa model with n-gram model.   
In result_comparison.csv you can find, for each record processed without errors by both RoBERTa and n-gram, if each model has correctly predicted all the token (perfect prediction)

## CODE