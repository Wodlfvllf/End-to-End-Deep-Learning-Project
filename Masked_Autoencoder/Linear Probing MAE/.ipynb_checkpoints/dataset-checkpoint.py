from utils import *
from model import *

class Custom_Dataset(Dataset):
    def __init__(self, x, y, transform, mode = 'train'):
        self.x = x
        if mode == 'train':
            self.y = y
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_1 = self.x[idx]

        if self.mode == 'train':
            label = self.y[idx]
        
        if self.transform:
            img_1 = self.transform(img_1)

        if self.mode == 'train':
            sample = {'img' : img_1, 'label' : label}
        else:
            sample = {'img':img_1}
        
        return sample
    