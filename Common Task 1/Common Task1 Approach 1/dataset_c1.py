from utils_c1 import *

class CustomDataset(Dataset):
    def __init__(self, x, y = 0, transform = None, mode = 'train'):
        self.x = x
        self.mode = mode
        if self.mode == 'train':
            self.y = y
        self.transform = transform
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = self.x[idx]

        if self.mode == 'train':
            label = self.y[idx]
            label = torch.tensor(label).float().unsqueeze(0)

        if self.transform:
            img = self.transform(img)
        
        if self.mode == 'train':
            sample = {'image':img, 'labels' : label}
        else:
            sample = {'image': img}
                   
        return sample
            