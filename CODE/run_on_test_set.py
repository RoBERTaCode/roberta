import argparse
import torch

from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer
)

def main():
    # Instantiate argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The model to use for the prediction.")
    parser.add_argument("--test_set_inputs_path", type=str, required=True,
                        help="The path of the test set containing the inputs to provide to the model.")
    parser.add_argument("--predictions_path", type=str, required=True,
                        help="The path of the file in which predictions will be printed.")
    parser.add_argument("--cpu", action='store_true', required=False,
                        help="Use it to force to run on CPU")

    # Generate args
    args = parser.parse_args()

    # Predictions
    predictions_file = open(args.predictions_path, 'w')
    predictions_file.write('PREDICTIONS' + '\n')

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForMaskedLM, RobertaTokenizer

    # Prepare the tokenizer
    tokenizer = tokenizer_class.from_pretrained(args.model_path)

    # Prepare the model
    model = model_class.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu:
        print("Running on CPU")
        device = "cpu"
    model.to(device)

    # Test set inputs
    with open(args.test_set_inputs_path) as f:
        inputs = f.readlines()
    inputs = [x.strip() for x in inputs]

    i = 0
    while i < len(inputs):
        print(str(i+1) + '  out of ' + str(len(inputs)))
        input = inputs[i]

        indexed_tokens = tokenizer.encode(input)
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to(device)
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        predicted_sentence = []
        for token in torch.argmax(predictions[0], 1).cpu().numpy():
            if token != tokenizer.convert_tokens_to_ids('<z>'):
                predicted_sentence.append(token)
            else:
                break

        prediction = tokenizer.decode(predicted_sentence)
        predictions_file.write(prediction + '\n')
        i += 1

if __name__ == "__main__":
    main()