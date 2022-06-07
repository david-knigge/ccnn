import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets
import datamodules
import pytorch_lightning as pl
import torchtext

from datasets import load_dataset, DatasetDict
import pickle
from pathlib import Path
import logging

import os
import numpy as np

# config
from hydra import utils


# Adapted from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2_lstm.ipynb
class IMDBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size,
        test_batch_size,
        data_type,
        pin_memory,
        num_workers,
        # Default values taken from S4
        max_length=4096,
        tokenizer_type="char",
        vocab_min_freq=15,
        append_bos=False,
        append_eos=True,
        val_split=0.0,
        **kwargs,
    ):
        assert tokenizer_type in [
            "word",
            "char",
        ], f"tokenizer_type {tokenizer_type} not supported"

        super().__init__()

        # Save parameters to self
        self.data_dir = Path(utils.get_original_cwd() + data_dir + "/IMDB")
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.max_length = max_length
        self.tokenizer_type = tokenizer_type
        self.vocab_min_freq = vocab_min_freq
        self.append_bos = append_bos
        self.append_eos = append_eos
        self.val_split = val_split

        self.cache_dir = self.get_cache_dir()

        # Determine data_type
        if data_type == "default":
            self.data_type = "sequence"
            self.data_dim = 1
        else:
            raise ValueError(f"data_type {data_type} not supported.")

        # Determine sizes of dataset
        self.input_channels = 1
        self.output_channels = 2

    def prepare_data(self):
        if self.cache_dir is None:  # Just download the dataset
            load_dataset("imdb", cache_dir=self.data_dir)
            self.process_dataset()

    def setup(self, stage=None):
        if stage == "test" and hasattr(self, "dataset_test"):
            return
        dataset, self.tokenizer, self.vocab = self.process_dataset()
        self.vocab_size = len(self.vocab)
        dataset.set_format(type="torch", columns=["input_ids", "label"])

        # Create all splits
        self.train_dataset, self.test_dataset = dataset["train"], dataset["test"]
        if self.val_split == 0.0:
            # Use test set as val set, as done in the LRA paper
            self.val_dataset = self.test_dataset
        else:
            train_val = self.train_dataset.train_test_split(
                test_size=self.val_split,
                seed=getattr(self, "seed", 42),
            )
            self.train_dataset, self.val_dataset = train_val["train"], train_val["test"]

        def collate_batch(batch):
            xs, ys = zip(*[(data["input_ids"], data["label"]) for data in batch])
            # lengths = torch.tensor([len(x) for x in xs])
            xs = torch.stack(
                [
                    torch.nn.functional.pad(
                        x,
                        [self.max_length - x.shape[-1], 0],
                        value=float(self.vocab["<pad>"]),
                    )
                    for x in xs
                ]
            )
            # xs = torch.nn.utils.rnn.pad_sequence(
            #     xs,
            #     padding_value=self.vocab['<pad>'],
            #     batch_first=True,
            # )
            xs = xs.unsqueeze(1).float()
            ys = torch.tensor(ys)
            return xs, ys

        self.collate_fn = collate_batch

    def process_dataset(self):
        if self.cache_dir is not None:
            return self._load_from_cache()

        dataset = load_dataset("imdb", cache_dir=self.data_dir)
        dataset = DatasetDict(train=dataset["train"], test=dataset["test"])
        if self.tokenizer_type == "word":
            tokenizer = torchtext.data.utils.get_tokenizer(
                "spacy", language="en_core_web_sm"
            )
        else:  # self.tokenizer_type == 'char'
            tokenizer = list  # Just convert a string to a list of chars
        # Account for <bos> and <eos> tokens
        max_length = self.max_length - int(self.append_bos) - int(self.append_eos)
        tokenize = lambda example: {"tokens": tokenizer(example["text"])[:max_length]}
        dataset = dataset.map(
            tokenize,
            remove_columns=["text"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )
        vocab = torchtext.vocab.build_vocab_from_iterator(
            dataset["train"]["tokens"],
            min_freq=self.vocab_min_freq,
            specials=(
                ["<pad>", "<unk>"]
                + (["<bos>"] if self.append_bos else [])
                + (["<eos>"] if self.append_eos else [])
            ),
        )
        vocab.set_default_index(vocab["<unk>"])

        numericalize = lambda example: {
            "input_ids": vocab(
                (["<bos>"] if self.append_bos else [])
                + example["tokens"]
                + (["<eos>"] if self.append_eos else [])
            )
        }
        dataset = dataset.map(
            numericalize,
            remove_columns=["tokens"],
            keep_in_memory=True,
            load_from_cache_file=False,
            num_proc=self.num_workers,
        )

        self._save_to_cache(dataset, tokenizer, vocab)
        return dataset, tokenizer, vocab

    def _save_to_cache(self, dataset, tokenizer, vocab):
        cache_dir = self.data_dir / self._cache_dir_name
        os.makedirs(str(cache_dir))
        logger = logging.getLogger(__name__)
        logger.info(f"Saving to cache at {str(cache_dir)}")
        dataset.save_to_disk(str(cache_dir))
        with open(cache_dir / "tokenizer.pkl", "wb") as f:
            pickle.dump(tokenizer, f)
        with open(cache_dir / "vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)

    def _load_from_cache(self):
        assert self.cache_dir.is_dir()
        logger = logging.getLogger(__name__)
        logger.info(f"Load from cache at {str(self.cache_dir)}")
        dataset = DatasetDict.load_from_disk(str(self.cache_dir))
        with open(self.cache_dir / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(self.cache_dir / "vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        return dataset, tokenizer, vocab

    @property
    def _cache_dir_name(self):
        return f"l_max-{self.max_length}-vocab-{self.tokenizer_type}-min_freq-{self.vocab_min_freq}-append_bos-{self.append_bos}-append_eos-{self.append_eos}"

    def get_cache_dir(self):
        cache_dir = self.data_dir / self._cache_dir_name
        if cache_dir.is_dir():
            return cache_dir
        else:
            return None

    # we define a separate DataLoader for each of train/val/test
    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
        return val_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn,
        )
        return test_dataloader
