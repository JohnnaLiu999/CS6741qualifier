import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
import wandb

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, \
    setup_wandb
from transformers import GenerationConfig
from load_data import load_t5_data
from utils import compute_metrics, save_queries_and_records, compute_records

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('device', DEVICE)
PAD_IDX = 0


def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")

    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for") ###
    parser.add_argument('--patience_epochs', type=int, default=200,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?") ###

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='bs_16_lr_3e-4_wd_0_epoch_20',
                        help="How should we name this experiment?") ###

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16) ###
    parser.add_argument('--test_batch_size', type=int, default=16) ###

    args = parser.parse_args()
    return args


def train(args, model, train_loader, dev_loader, optimizer, scheduler, eval_train=False):
    # TODO
    best_f1 = -1
    epochs_since_improvement = 0

    # Directory setup for saving model outputs and ground truth paths
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    data_type = 'train' if eval_train else 'dev'
    gt_sql_path = os.path.join('data', f'{data_type}.sql')
    gt_record_path = os.path.join('records', f'ground_truth_{data_type}.pkl')
    model_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_{data_type}.sql')
    model_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_{data_type}.pkl')

    for epoch in range(args.max_n_epochs):
        # Train
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        # Eval
        record_f1, record_em, sql_em, error_msg_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(f"Epoch {epoch}: Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: Error message rate: {error_msg_rate}")
        
        if args.use_wandb:
            result_dict = {
                'train/loss': tr_loss,
                'dev/record_f1': record_f1,
                'dev/record_em': record_em,
                'dev/sql_em': sql_em,
                'dev/error_rate': error_msg_rate
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            save_model(checkpoint_dir, model, best=True)
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)

        if epochs_since_improvement >= args.patience_epochs: # early stopping
            break
        if epoch == args.max_n_epochs-1: # max epoch
            break


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate.

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    # TODO
    model.eval()
    sql_queries = []
    for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_input in tqdm(dev_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        initial_decoder_input = initial_decoder_input.to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=encoder_input,
                                           attention_mask=encoder_mask,
                                           decoder_start_token_id=initial_decoder_input,
                                           max_length=512)

        queries = dev_loader.dataset.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        sql_queries.extend(queries)

    save_queries_and_records(sql_queries=sql_queries, sql_path=model_sql_path, record_path=model_record_path)

    sql_em, record_em, record_f1, error_msgs = compute_metrics(gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    error_rate = 0
    for msg in error_msgs:
        if msg:
            error_rate += 1
    error_rate = error_rate / len(error_msgs)
    return record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, gt_test_sql_path, model_sql_path, gt_test_record_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated
    database records. Implementation should be very similar to eval_epoch.
    '''
    # TODO
    model.eval()
    sql_queries = []
    for encoder_input, encoder_mask, initial_decoder_input in tqdm(test_loader):
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        initial_decoder_input = initial_decoder_input.to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(input_ids=encoder_input,
                                           attention_mask=encoder_mask,
                                           decoder_start_token_id=initial_decoder_input,
                                           max_length=512)

        queries = test_loader.dataset.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        sql_queries.extend(queries)

    save_queries_and_records(sql_queries=sql_queries, sql_path=model_sql_path, record_path=model_record_path)
    print(f'Test inference completed. Results saved to {model_sql_path} and {model_record_path}')

    records, error_msgs = compute_records(sql_queries)
    print(error_msgs)
    error_rate = 0
    for msg in error_msgs:
        if msg:
            error_rate += 1
    error_rate = error_rate / len(error_msgs)
    print(f"Test set Error message rate: {error_rate}")

    test_sql_em, test_record_em, test_record_f1, test_error_msgs = compute_metrics(gt_test_sql_path, model_sql_path, gt_test_record_path, model_record_path)

    # test_record_f1, test_record_em, test_sql_em, test_error_rate
    return test_record_f1, test_record_em, test_sql_em, error_rate

def main():
    # Get key arguments
    args = get_args()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train (+valid)
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Test
    model = load_model_from_checkpoint(args, best=True)
    model_type = 'ft' if args.finetune else 'scr'
    # model_sql_path = os.path.join(f'results/t5_{model_type}_test.sql')
    # model_record_path = os.path.join(f'records/t5_{model_type}_test.pkl')

    # Test
    gt_test_sql_path = os.path.join('results', f't5_{model_type}_test.sql')
    gt_test_record_path = os.path.join('records', f't5_{model_type}_test.pkl')
    model_test_sql_path = os.path.join('results', f't5_{model_type}_{args.experiment_name}_test.sql')
    model_test_record_path = os.path.join('records', f't5_{model_type}_{args.experiment_name}_test.pkl')
    test_record_f1, test_record_em, test_sql_em, test_error_rate = test_inference(args, model, test_loader, 
                   gt_test_sql_path, model_test_sql_path, 
                   gt_test_record_path, model_test_record_path)
    print(f"test f1 score: {test_record_f1}")

if __name__ == "__main__":
    main()
