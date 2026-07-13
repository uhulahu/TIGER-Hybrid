import pandas as pd
import torch
import torch.utils.data as data
import numpy as np


class EmbDataset(data.Dataset):

    def __init__(self, data_path, return_item_id=False):

        self.data_path = data_path
        df = pd.read_parquet(data_path)
        self.embeddings = np.stack(df['embedding'].values, axis=0)
        self.dim = self.embeddings.shape[-1]  # 768
        self.return_item_id = return_item_id
        if return_item_id:
            self.item_ids = df['ItemID'].values.astype(np.int64)

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb = torch.FloatTensor(emb)
        if self.return_item_id:
            return torch.tensor(self.item_ids[index], dtype=torch.long), tensor_emb
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
