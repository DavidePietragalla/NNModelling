# NNModelling — DSL for designing neural networks via visual node editor
# Copyright (C) 2026  Luca Sforza
#
# Licensed under the GNU General Public License v3 or later.
# Commercial licenses are available — contact Luca Sforza.
# See the LICENSE file for details.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

from dataset.ds import Dataset


class EnronSpamDataset(Dataset):
    """Text classification dataset: SetFit/enron_spam.

    Tokenizes with HF AutoTokenizer. Uses DataCollatorWithPadding.
    division() returns DataLoaders yielding (input_ids, labels) tuples.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        batch_size: int = 32,
        train_size: float = 0.8,
        num_workers: int = 4,
        max_length: int = 128,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.num_workers = num_workers

        raw = load_dataset("SetFit/enron_spam")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"], truncation=True, max_length=max_length
            )

        tokenized = raw.map(tokenize_fn, batched=True)
        # Remove all original columns, keep only tokenizer outputs + label
        keep = {"label", "input_ids", "attention_mask"}
        remove_cols = [c for c in tokenized["train"].column_names if c not in keep]
        tokenized = tokenized.remove_columns(remove_cols)
        tokenized = tokenized.rename_column("label", "labels")

        self.train_dataset = tokenized["train"]
        self.test_dataset = tokenized["test"]

    def __getitem__(self, index):
        return self.train_dataset[index]

    def __len__(self):
        return len(self.train_dataset)

    def _collate(self, batch):
        padded = self.collator(batch)
        return padded["input_ids"], padded["labels"]

    def division(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_len = int(len(self.train_dataset) * self.train_size)
        val_len = len(self.train_dataset) - train_len
        train_sub, val_sub = random_split(self.train_dataset, [train_len, val_len])

        train_loader = DataLoader(
            train_sub,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self.num_workers,
        )
        val_loader = DataLoader(
            val_sub,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader
