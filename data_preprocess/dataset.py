from torch.utils.data import Dataset
import torch
import pickle

class MyDataset(Dataset):
    """
    自定义数据集
    """
    def __init__(self, input_list, max_len):
        super().__init__()
        self.input_list = input_list
        self.max_len = max_len

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, index):
        input_ids = self.input_list[index] # 获取指定索引处的输入序列
        input_ids = input_ids[:self.max_len] # 截取输入序列的前max_len个token
        input_ids = torch.tensor(input_ids, dtype=torch.long) # 转换为张
        return input_ids











