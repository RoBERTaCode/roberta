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
### Dependencies
torch==1.4.0  
transformers==3.0.2  
tokenizers==0.8.1rc1  
wandb==0.9.1  

### Training a tokenizer  
python3 tokenizer-training.py --vocab_size [vocabulary size] --train_data_file [Path of the tokenizer training file (has the name matching *_tokenizer_training.txt)] --output_dir [folder where to store the tokenizer]

### Training a new model
python3 run_training.py --train_data_file [Path of the training file (has the name matching *_masked_code_training.txt)] --eval_data_file [Path of the evaluation file (has the name matching *_masked_code_eval.txt)] --output_root [folder where to store the model] --tokenizer_name [folder with the trained tokenizer] --vocab_size [vocabulary size]

### Running a trained model on a test set
python3 run_on_test_set.py --model_path [path of the trained model] --test_set_inputs_path [Path of the test file (has the name matching *_masked_code_test.txt)] --predictions_path [Path of the textual file where predictions will be written (the file is created by the script)]

### How to cite

```
@article{ciniselli2021empirical,
  title={An Empirical Study on the Usage of BERT Models for Code Completion},
  author={Ciniselli, Matteo and Cooper, Nathan and Pascarella, Luca and Poshyvanyk, Denys and Di Penta, Massimiliano and Bavota, Gabriele},
  journal={arXiv preprint arXiv:2103.07115},
  year={2021}
}
```

### Contributors
[Matteo Ciniselli](https://www.inf.usi.ch/phd/cinism/),
[Nathan Cooper](https://nathancooper.io/#/about),
[Luca Pascarella](https://lucapascarella.com/),
[Denys Poshyvanyk](http://www.cs.wm.edu/~denys/),
[Massimiliano Di Penta](https://mdipenta.github.io/),
[Gabriele Bavota](https://www.inf.usi.ch/faculty/bavota/),


### License
This software is licensed under the MIT License.

This project has received funding from the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation programme (grant agreement No. 851720). W&M team was supported in part by the NSF CCF-1955853, CCF-2007246 and CCF-1815186 grants. Any opinions, findings, and conclusions expressed herein are the authors’ and do not necessarily reflect those of the sponsors.