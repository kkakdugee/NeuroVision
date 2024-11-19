import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from scipy import ndimage
from torchvision import transforms

class NeuronDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Ensure consistent size for both image and mask
        if self.transform:
            image = self.transform(image)
            # Apply same resize to mask as to image
            mask = transforms.Resize((128, 128), transforms.InterpolationMode.NEAREST)(mask)
            mask = transforms.ToTensor()(mask)
        
        # Count neurons in mask (assuming each connected component is a neuron)
        mask_np = np.array(mask.squeeze())
        labeled_mask, num_neurons = ndimage.label(mask_np > 0.5)
        
        return image, mask, torch.tensor(num_neurons, dtype=torch.float32)