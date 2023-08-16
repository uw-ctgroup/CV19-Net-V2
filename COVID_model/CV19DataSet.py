import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF



class CV19DataSet(Dataset):
    def __init__(self, df, base_folder, transform, img_size):
        
        labels_pos = df.label_positive.tolist()
        labels_neg = df.label_negative.tolist()
        filenames = df.Filename.tolist()
        self.img_size = img_size
        self.labels_pos = labels_pos
        self.labels_neg = labels_neg
        self.filenames = filenames
        self.transform = transform
        self.base_folder = base_folder

    def __getitem__(self, index):
        label = [self.labels_pos[index], self.labels_neg[index]]
        fn = self.filenames[index]
        fn = fn.replace("\\","/")
        img = Image.open(self.base_folder + fn).convert('RGB')
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.filenames)

