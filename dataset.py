import random
import os
import sys
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, syn_path, real_path):
        self.images = []
        self.labels = np.arange(20)
        for i in range(10):
            file_path = os.path.join(syn_path, 'TL{}.png'.format(i))
            self.images.append(file_path)
        for i in range(10):
            file_path = os.path.join(real_path, 'TL{}.bmp'.format(i))
            self.images.append(file_path)
        

    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = img.resize((224, 224), Image.ANTIALIAS)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = transform(img)
        target = self.labels[index]
        return img, target