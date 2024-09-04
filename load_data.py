import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.data = self.process_data(data_folder, split, self.tokenizer)
        self.split = split

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        inputs = load_lines(os.path.join(data_folder, f'{split}.nl'))
        tokenized_inputs = tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors='pt')
        if split != "test":
            targets = load_lines(os.path.join(data_folder, f'{split}.sql'))
            tokenized_targets = tokenizer(targets, max_length=512, padding=True, truncation=True, return_tensors='pt')
            decoder_input_ids = [[tokenizer.pad_token_id] + tok_ids[:-1].tolist() for tok_ids in
                                 tokenized_targets['input_ids']]
            return list(zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], torch.tensor(decoder_input_ids),
                        tokenized_targets['input_ids']))
        else:
            batch_size = tokenized_inputs['input_ids'].shape[0]
            return list(zip(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], 
                            torch.tensor(self.tokenizer.pad_token_id).repeat((batch_size, 1))))

    def __len__(self):
        # TODO
        return len(self.data)

    def __getitem__(self, idx):
        # TODO
        return self.data[idx]


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, encoder_masks, decoder_inputs, decoder_targets = zip(
        *[(item[0], item[1], item[2], item[3]) for item in batch])
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor([item[2][0] for item in batch])

    return encoder_ids, encoder_masks, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_ids, encoder_masks, initial_decoder_inputs = zip(*[(item[0], item[1], item[2][0]) for item in batch])
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_masks = pad_sequence(encoder_masks, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.tensor(initial_decoder_inputs)

    return encoder_ids, encoder_masks, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(data_folder+"/train.nl")
    train_y = load_lines(data_folder + "/train.sql")

    dev_x = load_lines(data_folder+"/dev.nl")
    dev_y = load_lines(data_folder + "/dev.sql")

    test_x = load_lines(data_folder+"/test.nl")
    return train_x, train_y, dev_x, dev_y, test_x