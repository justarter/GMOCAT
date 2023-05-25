import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
       
        label = data['labels']
        question = data['q_ids']
        user_id = data['user_id']

        output = {'label': torch.FloatTensor(label), 'question': torch.FloatTensor(question), 'user_id':user_id}
        return output


class collate_fn(object):
    def __init__(self, n_question):
        self.n_question = n_question

    def __call__(self, batch):
        B = len(batch)
        labels = torch.zeros(B, self.n_question).long()
        mask = torch.zeros(B, self.n_question).long()
        user_ids = []
        for b_idx in range(B):
            labels[b_idx, batch[b_idx]['question'].long()] = batch[b_idx]['label'].long()
            mask[b_idx, batch[b_idx]['question'].long()] = 1
            user_ids.append(batch[b_idx]['user_id'])
            
        output = {'labels': labels, 'mask': mask, 'user_ids':user_ids}
        return output
