import numpy as np
import torch
from torch.utils.data import DataLoader

class GenRecDataLoader(DataLoader):
    """
    GenRecDataLoader for Generative Recommendation tasks.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): Number of subprocesses to use for data loading.
        collate_fn (callable, optional): Function to merge a list of samples to form a mini-batch.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=None,
                 use_user_token=False):
        self.use_user_token = use_user_token
        collate_fn = self.collate_fn
        super(GenRecDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                               num_workers=num_workers, collate_fn=collate_fn)


    def collate_fn(self, batch, pad_token=0):
        """
        crate attention mask for input sequence. 为输入序列创建注意力掩码。

        Args:
            batch (list): List of samples from the dataset.

        Returns:
            dict: Batched data with padded sequences.
        """
        # Assuming each item in batch is a dictionary with 'history' and 'target'
        histories = [item['history'] for item in batch] # [[], [], ...]
        targets = [item['target'] for item in batch]

        # Flatten histories and targets 展平成(batch_size, 4N)一维序列
        flattened_histories = torch.stack( # 这里sublist就是一个SID——由4个token组成的列表，N*4 -> 4N
            [torch.tensor([elem for sublist in history for elem in sublist], dtype=torch.int64) for history in histories]
        )
        flattened_targets = torch.stack(
            [torch.tensor(target, dtype=torch.int64) for target in targets]
        )

        # Create attention masks for flattened histories
        attention_masks = torch.stack( # 1 = 参与 attention，0 = 被 mask（忽略）
            [torch.tensor([1 if elem != pad_token else 0 for elem in h], dtype=torch.int64) for h in flattened_histories]
        )

        # Prepend user token to the front of every input sequence
        if self.use_user_token:
            user_tokens = torch.tensor(
                [[item['user_token']] for item in batch], dtype=torch.int64
            )  # (batch, 1)
            flattened_histories = torch.cat([user_tokens, flattened_histories], dim=1)
            user_mask = torch.ones(user_tokens.shape, dtype=torch.int64)
            attention_masks = torch.cat([user_mask, attention_masks], dim=1)

        return {'history': flattened_histories, 'target': flattened_targets, 'attention_mask': attention_masks}
