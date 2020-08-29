# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import argparse
import glob
import logging
import re
import shutil
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

import time
import sys
import os
import wandb
import random

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)
logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)


MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
}


class LineByLineDatasetWithBPETokenizer(Dataset):
    def __init__(self, file_path: str = None, tokenizer_path: str = None):
        tokenizer = ByteLevelBPETokenizer(
            tokenizer_path + "/vocab.json",
            tokenizer_path + "/merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        with open(file_path, encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line for line in lines if (len(line) > 0 and not line.isspace())]
            self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])


def load_and_cache_examples(args, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return LineByLineDatasetWithBPETokenizer(file_path, args.tokenizer_name)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)

def decision(probability):
    return random.random() < probability

# Reads the masked instances
def read_masked_dataset(tokenizer: PreTrainedTokenizer, batch, labels_to_process) -> Tuple[torch.Tensor, torch.Tensor]:
    # The inputs are already masked in the training file
    tmp_inputs = batch.clone()

    tmp_inputs_list = []
    for input in tmp_inputs:
        decoded_input = tokenizer.decode(input)
        encoded_back = tokenizer.encode(decoded_input)[1:-1] # Removes the additional <s> and </s> added
        tmp_inputs_list.append(encoded_back)

    # Gets the maximum length between inputs and labels_lines
    # We then need to adapt one or the other to have the same length through padding
    max_length_inputs = max([len(input) for input in tmp_inputs_list])
    max_length_labels_lines = max([len(label) for label in labels_to_process])
    max_length = max_length_inputs
    if max_length_labels_lines > max_length_inputs:
        max_length = max_length_labels_lines

    # Create the labels tensor
    labels_to_convert_in_tensor = []

    i = 0
    while i < len(batch):
        l1_tmp = tokenizer.encode(labels_to_process[i])
        label_to_add = []
        for token in l1_tmp:
            if token != tokenizer.bos_token_id and token != tokenizer.eos_token_id:  # Remove special tokens
                label_to_add.append(token)

        j = len(label_to_add)
        while j < max_length:
            label_to_add.append(-100)  # we only compute loss for masked tokens
            j += 1

        labels_to_convert_in_tensor.append(label_to_add)
        i += 1

    labels = torch.as_tensor(labels_to_convert_in_tensor)

    inputs_to_convert = []
    for input in tmp_inputs_list:
        tmp_input = []
        for token in input:
            tmp_input.append(token)

        i = len(tmp_input)
        while i < max_length:
            tmp_input.append(tokenizer.pad_token_id)
            i += 1
        inputs_to_convert.append(tmp_input)

    inputs = torch.as_tensor(inputs_to_convert)

    return inputs, labels

# Reads the masked instances, but provide them as non-masked to the model
# This is used in 10% of cases during training
def get_non_masked_instances(tokenizer: PreTrainedTokenizer, batch, labels_to_process) -> Tuple[torch.Tensor, torch.Tensor]:
    tmp_inputs_list = []
    i = 0
    while i < len(batch):
        decoded_input = tokenizer.decode(batch[i]).replace('<x>',str(labels_to_process[i]).replace('<z>\n',''))
        encoded_back = tokenizer.encode(decoded_input)[1:-1]
        tmp_inputs_list.append(encoded_back)
        i += 1

    # Gets the maximum length
    max_length = max([len(input) for input in tmp_inputs_list])

    inputs_to_convert = []
    labels_to_convert = []
    for input in tmp_inputs_list:
        tmp_input = []
        tmp_label = []
        tmp_label.append(tokenizer.convert_tokens_to_ids('<z>'))
        for token in input:
            tmp_input.append(token)
            tmp_label.append(-100)

        del tmp_label[-1] #Accounts for the fact that tmp_label already contains <z>

        i = len(tmp_input)
        while i < max_length:
            tmp_input.append(tokenizer.pad_token_id)
            tmp_label.append(-100)
            i += 1
        labels_to_convert.append(tmp_label)
        inputs_to_convert.append(tmp_input)

    inputs = torch.as_tensor(inputs_to_convert)
    labels = torch.as_tensor(labels_to_convert)

    # We train the model to understand that if no masked tokens are present, nothing must be produced, only <z>
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    labels_file = str(args.train_data_file).replace('masked_code_','mask_')
    labels_lines = [line.rstrip() for line in open(labels_file)]

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # step is the count of the steps performed, batch contains the actual input data

        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            # Get the labels lines to process
            start = step * len(batch)
            end = start + len(batch) + 1
            lables_to_process = labels_lines[start:end]

            # In 90% of cases, we used the inputs with the masked tokens
            # In 10% of cases we don't mask any token
            if decision(0.9):
                inputs, labels = read_masked_dataset(tokenizer, batch, lables_to_process)
            else:
                inputs, labels = get_non_masked_instances(tokenizer, batch, lables_to_process)

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                wandb.log({'train_batch_loss': loss.item()})
                wandb.log({'avg_train_loss': tr_loss / len(train_dataloader)})

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        if results is None:
                            print("Stopping condition reached, no improvement in evaluation set")
                            sys.exit(0)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    labels_file = str(args.eval_data_file).replace('masked_code_', 'mask_')
    labels_lines = [line.rstrip() for line in open(labels_file)]

    step = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        # Get the labels lines to process
        start = step * len(batch)
        end = start + len(batch) + 1
        lables_to_process = labels_lines[start:end]

        step += 1

        inputs, labels = read_masked_dataset(tokenizer, batch, lables_to_process)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    perfect_predictions, num_examples = get_number_perfect_predictions(model, tokenizer, args.eval_data_file)
    result = {"perplexity": perplexity, "loss": eval_loss,
              "perfect_predictions": perfect_predictions, "total_eval_examples": num_examples}

    wandb.log({'val_perplexity': perplexity, 'avg_val_loss': eval_loss})
    wandb.log({'perfect_predictions': perfect_predictions})
    wandb.log({'perfect_predictions_percentage': perfect_predictions / num_examples})

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_" + str(time.time()) + ".txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if args.early_stop > 0:
        # Early stop has been required by the user, check performance
        eval_results_files = glob.glob(os.path.join(eval_output_dir, prefix,'eval_results_*.txt'))
        eval_results_files.sort(key=lambda x: os.stat(os.path.join(eval_output_dir, x)).st_mtime)
        if len(eval_results_files) > args.early_stop:
            perfect_predictions_before = read_perfect_predictions_from_file(eval_results_files[len(eval_results_files)-(args.early_stop+1)])
            if perfect_predictions <= perfect_predictions_before:
                return None

    return result


def read_perfect_predictions_from_file(file_path):
    with open(file_path) as f:
        content = f.read()
    p = re.compile('perfect_predictions = (.*?)\n')
    return int(p.search(content).group(1))

def get_number_perfect_predictions(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_data_file):
    labels_file = str(eval_data_file).replace('masked_code_', 'mask_')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Inputs
    with open(eval_data_file) as f:
        inputs = f.readlines()
    inputs = [x.strip() for x in inputs]

    # Targets
    with open(labels_file) as f:
        targets = f.readlines()
    targets = [x.strip() for x in targets]

    n_perfect_predictions = 0
    i = 0
    while i < len(inputs):
        input = inputs[i]
        target = "".join(targets[i].split()).replace('<z>', '')

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
        prediction = "".join(prediction.split())
        if target == prediction:
            n_perfect_predictions += 1
        i += 1

    return n_perfect_predictions, len(inputs)


class Args:

    def __init__(self, run, train_data_file, eval_data_file, output_root, tokenizer_name, vocab_size,
                 early_stop, logging_steps, should_continue
    ):
        self.train_data_file = train_data_file
        self.eval_data_file = eval_data_file
        self.output_root = output_root
        self.model_type = 'roberta'
        self.early_stop = early_stop
        self.line_by_line = True
        self.should_continue = should_continue
        self.model_name_or_path = ''
        self.mlm = True
        self.mlm_probability = 0.2
        self.tokenizer_name = tokenizer_name
        self.vocab_size = vocab_size
        self.cache_dir = 'cache'
        self.block_size = -1
        self.do_train = True
        self.do_eval = True
        self.evaluate_during_training = True
        self.per_gpu_train_batch_size = wandb.config.batch_size
        self.per_gpu_eval_batch_size = wandb.config.batch_size
        self.gradient_accumulation_steps = wandb.config.gradient_accumulation_steps
        self.learning_rate = wandb.config.learning_rate
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = wandb.config.epochs
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = logging_steps
        self.save_steps = logging_steps
        self.save_total_limit = None
        self.eval_all_checkpoints = False
        self.no_cuda = False
        self.overwrite_output_dir = True
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = "O1"
        self.local_rank = -1
        self.server_ip = ""
        self.server_port = ""
        run.save()
        self.name = run.name
        if self.should_continue:
            self.output_dir = self.output_root
        else:
            self.output_dir = self.output_root  + '/' + self.name

def get_config(args):
    config = {
        "model_type": "roberta",
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.3,
        "hidden_size": wandb.config.hidden_size,
        "initializer_range": 0.02,
        "num_attention_heads": wandb.config.num_attention_heads,
        "num_hidden_layers": wandb.config.num_hidden_layers,
        "vocab_size": args.vocab_size,
        "intermediate_size": wandb.config.intermediate_size,
        "max_position_embeddings": 1024,
        "cache_dir": args.cache_dir
    }

    return RobertaConfig(**config)


def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens({'additional_special_tokens': ['<z>', '<x>']}) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def main():
    run = wandb.init(project="roberta")

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--early_stop",
        default=0,
        type=int,
        help="The number of runs on the evaluation set without any improvement before the training is stopped. Put 0 to not use early stop.",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")

    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--eval_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )

    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--tokenizer_name", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--vocab_size", default=None, type=int, required=True, help="The input training data file (a text file)."
    )

    parsed_args = parser.parse_args()
    args = Args(
        run, parsed_args.train_data_file, parsed_args.eval_data_file,
        parsed_args.output_root, parsed_args.tokenizer_name, parsed_args.vocab_size,
        parsed_args.early_stop, parsed_args.logging_steps, parsed_args.should_continue
    )

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.n_gpu = 1  # 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = get_config(args)

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        try:
            model = model_class(config=config)
        except Exception as e:
            logger.error(f'{e} Configuration not correct for {args.name}')
            return

    add_special_tokens_(model, tokenizer)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            if result is None:
                print("Stopping condition reached, no improvement in evaluation set")
                sys.exit(0)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    sweep_config = {
        'method': 'grid',
        'metric': {
            'name': 'perfect_predictions_percentage',
            'goal': 'maximize'
        },
        'parameters': {
            'vocab_size': {
                'values': [1_130]
            },
            'learning_rate': {
                'values': [5e-5]
            },
            'gradient_accumulation_steps': {
                'values': [4]  # equivalent to having effective batch size of 64 with a real batch_size of 16
            },
            'batch_size': {
                'values': [8]
            },
            'epochs': {
                'values': [50]
            },
            'hidden_size': {
                'values': [768]
            },
            'num_attention_heads': {
                'values': [16]
            },
            'num_hidden_layers': {
                'values': [12]
            },
            'intermediate_size': {
                'values': [4_096]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=main)
