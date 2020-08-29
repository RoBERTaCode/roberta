import argparse
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

def main():
    # Instantiate argument parser
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True, help="The input training data file or a path to a directory with multiple training data files.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where the tokenizer model will be written.")
    # Optional parameters
    parser.add_argument("--vocab_size", default=5000, type=int, help="Vocabulary maximum size, default 5000.")
    parser.add_argument("--min_freq", default=2, type=int, help="Minimum number of occurrences, default 2")

    # Generate args
    args = parser.parse_args()

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Get training files
    paths = os.path.abspath(args.train_data_file)
    if not args.train_data_file.endswith(".txt"):
        paths = [str(x) for x in Path(paths).glob("**/*.txt")]

    # Customize training
    tokenizer.train(files=paths, vocab_size=args.vocab_size, min_frequency=args.min_freq, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.add_special_tokens(["<x>","<z>"])

    # Save files to disk
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer.save_model(output_dir)


if __name__ == "__main__":
    main()
