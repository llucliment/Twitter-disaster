from torch.utils.data import Dataset
import torch
from src.preprocessing import numericalize
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
# Dataset for using custom RNNs
class DisasterTweetsDataset(Dataset):
    def __init__(self, df, vocab, is_test=False):
        self.texts = df["text"].tolist()
        self.vocab = vocab
        self.is_test = is_test

        if not is_test:
            self.labels = df["target"].values.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = numericalize(self.texts[idx], self.vocab)

        if len(token_ids) == 0:
            token_ids = [self.vocab[UNK_TOKEN]]

        x = torch.tensor(token_ids, dtype=torch.long)

        if self.is_test:
            return x

        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y
    
def collate_batch(batch, vocab):
    texts, labels = zip(*batch)

    lengths = torch.tensor([len(x) for x in texts], dtype=torch.long)
    padded_texts = pad_sequence(
        texts,
        batch_first=True,
        padding_value=vocab[PAD_TOKEN]
    )

    labels = torch.stack(labels)

    return padded_texts, lengths, labels


def collate_test_batch(batch, vocab):
    lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
    padded_texts = pad_sequence(
        batch,
        batch_first=True,
        padding_value=vocab[PAD_TOKEN]
    )

    return padded_texts, lengths



# Dataset for using bertweet
class TweetDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )

        return item