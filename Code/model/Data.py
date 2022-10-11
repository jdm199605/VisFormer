import pandas as pd
import json
import csv
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import random
from transformers import BertTokenizer
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import PIL
from PIL import Image

tokenizer = BertTokenizer.from_pretrained("../tokenizer/")

def table_tokenizer(table_data, sampling_num, max_seq_len):
    output = []
    max_len = 0
    data_type = []

    table_data.replace([np.inf,-np.inf],np.nan,inplace=True)
    table_data.fillna('ffill',inplace=True)
    table_data.fillna('bfill',inplace=True)
    
    if table_data.shape[0] > sampling_num:
        sampling_index = random.sample(range(0,table_data.shape[0]),sampling_num)
        table_data = pd.DataFrame(table_data.iloc[sampling_index])
    
    for c, col in enumerate(table_data.columns):
        col_type = table_data[col].dtype
        
        tokens = tokenizer.tokenize(col)
        input_ids = [101] + tokenizer.convert_tokens_to_ids(tokens) + [102]
        
        for cell in table_data[col]:
            try:
                tokens = tokenizer.tokenize(cell)
                input_ids += tokenizer.convert_tokens_to_ids(tokens)
                input_ids += [102]
            except:
                input_ids += [102]
        max_len= max(max_len,len(input_ids))
        output.append(torch.tensor(input_ids).float()[:max_seq_len])
        data_type.append(2) # 2 represents the column is categorical or temporal
                            
    return output, min(max_len,max_seq_len), torch.tensor(data_type)
            
def collate_batch(batch, max_seq_len = 512, sampling_num = 300, NSR=1):
    max_cols = 0
    max_len = 0
    col_num_list = [] #records the number of columns
    type_list = [] #records the vis type
    padded_table_list = []
    pairs_list = []
    data_type_list = []
    
    for table_data, pairs, vis_types in batch:
        max_cols = max(max_cols,table_data.shape[1])
        col_num_list.append(table_data.shape[1])
        pairs_list.append(pairs)
        type_list.append(vis_types)

        output, lens, data_type= table_tokenizer(table_data, sampling_num, max_seq_len)                        
        max_len = max(max_len, lens)                         
        padded_table_list.append(pad_sequence(output, batch_first = True))
        data_type_list.append(data_type)

    for i in range(len(padded_table_list)):
        c = padded_table_list[i].shape[1]
        padded_table_list[i] = F.pad(padded_table_list[i],[0,max_len-c,0,0])

    batch_tensor = pad_sequence(padded_table_list,batch_first=True)
    data_type_list = pad_sequence(data_type_list, batch_first=True).view(-1,1)

    label_tensor = torch.zeros(len(batch), 4, max_cols+1, max_cols+1)

    mask_tensor = torch.zeros(len(batch), 4, max_cols+1, max_cols+1)
    if NSR == 0:
        for i in range(len(batch)):
            for j in range(4):
                mask_tensor[i, j, :col_num_list[i]+1:, :col_num_list[i]+1] = 1
                diag = torch.diag(mask_tensor[i, j, :col_num_list[i]+1:, :col_num_list[i]+1])
                mask_tensor[i, j, :col_num_list[i]+1:, :col_num_list[i]+1] -= torch.diag_embed(diag)
                
    for k in range(len(batch)):
        pairs = eval(pairs_list[k])
        vis_types = eval(type_list[k])
        
        for idx, (x, y) in enumerate(pairs):
            try:
                label_tensor[k][vis_types[idx]][x][y] = 1
            except:
                print (vis_types,idx,x,y,col_num_list[k])
                #exit()
            if NSR != 0:
                mask_tensor[k, :, x, y] = 1
                #sampling negative instances
                neg = torch.randint(0, col_num_list[k]+1, (NSR,2))
                for neg_x, neg_y in neg:
                    mask_tensor[k, :, neg_x, neg_y] = 1

    return batch_tensor, label_tensor, mask_tensor, data_type_list, torch.tensor(col_num_list)

transform = transforms.Compose([
        transforms.Resize((50,70)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

class VRIDataset(Dataset):
    def __init__(self, file_name, num=20000):
        self.dataset = pd.read_table(file_name, sep='\t').iloc[:70000].reset_index(drop=True)
    
    def __getitem__(self, index):
        path = self.dataset['table_data'][index]
        table_data = pd.read_table(path, sep=',')
        pairs = self.dataset['pairs'][index]
        types = self.dataset['types'][index]
        return table_data, pairs, types
    
    def __len__(self):
        return len(self.dataset)
    
class VEDataset(Dataset):
    def __init__(self, file_name, num = 9999, transform = transform):
        self.dataset = pd.read_table(file_name, sep = '\t').loc[:num]
        self.transform = transform

    def __getitem__(self,index):
        path = self.dataset['path'][index]
        img = Image.open(path)
        img = self.transform(img)
        label = torch.tensor(self.dataset['label'][index])
        return img, label

    def __len__(self):
        return len(self.dataset)

def collate_batch_V2(batch, max_seq_len = 512, sampling_num = 300, NSR=1):
    max_cols = 0
    max_len = 0
    col_num_list = [] #records the number of columns
    vis_type_list = [] #records the vis type
    padded_table_list = []
    label_list = []
    data_type_list = []
    img_batch = []
    col_pairs = []
    labels = []
    
    for table, img, label, vis_type, col_pair in batch:
        max_cols = max(max_cols,table.shape[1])
        col_num_list.append(table.shape[1])
        vis_type_list.append(vis_type)

        output, lens, data_type= table_tokenizer(table, sampling_num, max_seq_len)                        
        max_len = max(max_len, lens)                         
        padded_table_list.append(pad_sequence(output, batch_first = True))
        data_type_list.append(data_type)
        
        img_batch.append(img)
        col_pairs.append(torch.Tensor(col_pair))
        labels.append(label)

    for i in range(len(padded_table_list)):
        c = padded_table_list[i].shape[1]
        padded_table_list[i] = F.pad(padded_table_list[i],[0,max_len-c,0,0])

    table_batch = pad_sequence(padded_table_list,batch_first=True)
    data_type_list = pad_sequence(data_type_list, batch_first=True).view(-1,1)
    vis_type_list = torch.Tensor(vis_type_list)
    img_batch = torch.stack(img_batch)
    col_pairs = torch.stack(col_pairs)
    labels = torch.Tensor(labels).unsqueeze(1)

    return table_batch, img_batch, labels, data_type_list, vis_type_list, torch.tensor(col_num_list), col_pairs 

class V2Dataset(Dataset):
    def __init__(self, file_name, num = 50000, transform = transform):
        self.dataset = pd.read_table(file_name, sep = '\t').loc[:num].reset_index(drop=True)
        self.transform = transform
    
    def __getitem__(self, index):
        table_path = self.dataset['table_data'][index]
        table = pd.read_table(table_path, sep=',')
        img_path = self.dataset['path'][index]
        img = Image.open(img_path)
        img = self.transform(img)
        label = self.dataset['label'][index]
        info = eval(self.dataset['idx'][index])
        vis_type = info[0]
        col_pair = info[1:]
        
        return table, img, label, vis_type, col_pair 
    
    def __len__(self):
        return len(self.dataset)