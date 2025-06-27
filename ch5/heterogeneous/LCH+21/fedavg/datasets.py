import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class MyTabularDataset(Dataset):

    def __init__(self, dataset, label_col):

        self.label = torch.LongTensor(dataset[label_col].values)

        self.data = dataset.drop(columns=[label_col]).values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        label = self.label[index]
        data = self.data[index]

        return torch.tensor(data).float(), label

class MyImageDataset(Dataset):
    def __init__(self, dataset, file_col, label_col, projector=None):
        self.file = dataset[file_col].values
        self.label = dataset[label_col].values
        self.projector = projector

        self.normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):

        label = torch.tensor(self.label[index])
        img = Image.open(self.file[index]).convert("RGB")

        data = transforms.ToTensor()(img)
        if self.projector is not None:
            data = self.projector(data.unsqueeze(0)).squeeze(0)

        data = self.normalize(data)
        return data, label

class VRDataset(Dataset):
    def __init__(self, data, label):

        self.data = data

        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label[index])

        return data, label

def get_dataset(conf, data, projector=None):

    if conf['data_type'] == 'tabular':
        dataset = MyTabularDataset(data, conf['label_column'])
    elif conf['data_type'] == 'image':
        dataset = MyImageDataset(data, conf['data_column'], conf['label_column'], projector)
    else:
        return None
    return dataset






