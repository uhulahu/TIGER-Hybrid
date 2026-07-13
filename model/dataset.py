import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def process_data(file_path, mode, max_len, PAD_TOKEN=0, use_user_token=False):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file.
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.
        use_user_token (bool): Whether to include user_id in each sample for user token prepending.

    Returns:
        list: Processed data.
    """
    # Load parquet data
    data = pd.read_parquet(file_path) # (22363, 3) [user, history, target] 这里面是itemID还不是SID

    # Combine "history" and "target" columns into a single sequence
    # Important: Ensure 'history' is a list and 'target' is appended correctly
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        # Sliding window processing 滑动窗口构造训练样本
        processed_data = []  # 样本列表
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                sample = {
                    'history': sequence[:i],
                    'target': sequence[i]
                }
                if use_user_token:
                    sample['user_id'] = row.user
                processed_data.append(sample)
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            sample = {
                'history': sequence[:-1],
                'target': sequence[-1]
            }
            if use_user_token:
                sample['user_id'] = row.user
            processed_data.append(sample)
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    # Apply padding or truncation 填充到max_len
    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence
    
def item2code(code_path, codebook_size=256):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}
    
    # for index, code in enumerate(data):
    #     item_to_code[index + 1] = code
    #     code_to_item[tuple(code)] = index + 1

    # 这里对code做offset，因为四个token都是0~255，没法区分层级
    # 但是为什么要"+1"：因为token0是PAD
    # 这样就是0~1024互不冲突
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len, PAD_TOKEN=0,
                 use_user_token=False, num_user_tokens=2000, user_token_offset=1025):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
            use_user_token (bool): Whether to prepend a hashed user token to the input sequence.
            num_user_tokens (int): Number of user token buckets (paper uses 2000).
            user_token_offset (int): Starting vocab ID for user tokens (default: 1025).
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        self.use_user_token = use_user_token
        self.num_user_tokens = num_user_tokens
        self.user_token_offset = user_token_offset
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path)
        # Process the dataset
        self.data = self._prepare_data()

    def _user_id_to_token(self, user_id):
        """Hash raw user_id (int) into a user token bucket.

        Uses the hashing trick (Weinberger et al., 2009): deterministic modulo
        mapping that collapses 22k raw user IDs into 2000 shared token buckets.
        Multiple users sharing the same bucket forces the model to rely on
        behavioral history for personalisation rather than user-ID memorisation.
        """
        return user_id % self.num_user_tokens + self.user_token_offset

    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len,
            PAD_TOKEN=self.PAD_TOKEN,
            use_user_token=self.use_user_token
        )
        # Convert items to codes 将历史序列和target中的itemID转换为SID
        for item in processed_data:
            item['history'] = [self.item_to_code.get(x, np.array([self.PAD_TOKEN]*4)) for x in item['history']]
            item['target'] = self.item_to_code.get(item['target'], np.array([self.PAD_TOKEN]*4))
            if self.use_user_token:
                item['user_token'] = self._user_id_to_token(item.pop('user_id'))
        return processed_data

    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history', 'target', and optionally 'user_token'.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
    
if __name__ == "__main__":
    # Example usage
    dataset_path = 'data/Beauty/train.parquet'
    code_path = 'data/Beauty/Beauty_t5_rqvae.npy'
    mode = 'train'  # or 'train'
    max_len = 20

    dataset = GenRecDataset(dataset_path, code_path, mode, max_len)
    print("Number of items in dataset:", len(dataset))

    print("First five items in dataset:", [dataset[i] for i in range(5)])
    
