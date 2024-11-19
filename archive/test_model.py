import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import os
from PIL import Image
import numpy as np

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
        from scipy import ndimage
        labeled_mask, num_neurons = ndimage.label(mask_np > 0.5)
        
        return image, mask, torch.tensor(num_neurons, dtype=torch.float32)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NeuronCounter(nn.Module):
    def __init__(self, input_size=128):
        super().__init__()
        self.input_size = input_size
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate sizes after initial layers
        self.size_after_conv1 = input_size // 2  # 64
        self.size_after_pool = self.size_after_conv1 // 2  # 32
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 64, 2)  # 32
        self.layer2 = self._make_layer(64, 128, 2, stride=2)  # 16
        self.layer3 = self._make_layer(128, 256, 2, stride=2)  # 8
        
        # Segmentation head - careful upsampling to match input size
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 16
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 32
        self.upconv3 = nn.ConvTranspose2d(64, 32, 2, stride=2)    # 64
        self.upconv4 = nn.ConvTranspose2d(32, 1, 2, stride=2)     # 128
        
        # Counting head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        features = self.layer3(x)
        
        # Segmentation path with careful upsampling
        seg = self.upconv1(features)
        seg = F.relu(seg)
        seg = self.upconv2(seg)
        seg = F.relu(seg)
        seg = self.upconv3(seg)
        seg = F.relu(seg)
        seg = self.upconv4(seg)
        seg_out = torch.sigmoid(seg)
        
        # Counting path
        count = self.avgpool(features)
        count = torch.flatten(count, 1)
        count_out = self.fc(count)
        
        return seg_out, count_out

def create_data_loaders(image_dir, mask_dir, batch_size, train_split=0.8, transform=None):
    """
    Create train and test data loaders with a specified split ratio
    
    Args:
        image_dir (str): Directory containing images
        mask_dir (str): Directory containing masks
        batch_size (int): Batch size for DataLoader
        train_split (float): Proportion of data to use for training (0 to 1)
        transform: Torchvision transforms to apply
    """
    # Create full dataset
    dataset = NeuronDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=transform
    )
    
    # Calculate lengths for train and test
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size
    
    # Create train and test splits
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

def train_model(model, train_loader, criterion_seg, criterion_count, optimizer, device, epoch, save_dir):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, masks, counts) in enumerate(train_loader):
        images, masks, counts = images.to(device), masks.to(device), counts.to(device)
        
        optimizer.zero_grad()
        seg_out, count_out = model(images)
        
        # Combined loss
        loss_seg = criterion_seg(seg_out, masks)
        loss_count = criterion_count(count_out.squeeze(), counts)
        loss = loss_seg + loss_count
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}: Loss = {loss.item():.4f} '
                  f'(Seg: {loss_seg.item():.4f}, Count: {loss_count.item():.4f})')
    
    # Calculate average loss for the epoch
    avg_loss = total_loss / num_batches
    
    # Save checkpoint periodically
    if (epoch + 1) % 5 == 0:  # Save every 5 epochs
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
    
    return avg_loss

def evaluate_model(model, test_loader, criterion_seg, criterion_count, device):
    """Evaluate the model on the test set"""
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_count_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks, counts in test_loader:
            images, masks, counts = images.to(device), masks.to(device), counts.to(device)
            
            seg_out, count_out = model(images)
            
            loss_seg = criterion_seg(seg_out, masks)
            loss_count = criterion_count(count_out.squeeze(), counts)
            loss = loss_seg + loss_count
            
            total_loss += loss.item()
            total_seg_loss += loss_seg.item()
            total_count_loss += loss_count.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_seg_loss = total_seg_loss / num_batches
    avg_count_loss = total_count_loss / num_batches
    
    return avg_loss, avg_seg_loss, avg_count_loss

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint with all information needed to resume training"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved: {path}')

def main():
    # Hyperparameters
    batch_size = 8
    learning_rate = 0.001
    num_epochs = 50
    input_size = 128
    train_split = 0.8  # 80% train, 20% test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    save_dir = './model_checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create train and test data loaders
    train_loader, test_loader = create_data_loaders(
        image_dir='./dataset/all_images/images/',
        mask_dir='./dataset/all_masks/masks/',
        batch_size=batch_size,
        train_split=train_split,
        transform=transform
    )
    
    # Model setup
    model = NeuronCounter(input_size=input_size).to(device)
    criterion_seg = nn.BCELoss()
    criterion_count = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss = train_model(model, train_loader, criterion_seg, criterion_count, 
                                optimizer, device, epoch, save_dir)
        
        # Evaluate
        test_loss, test_seg_loss, test_count_loss = evaluate_model(
            model, test_loader, criterion_seg, criterion_count, device
        )
        
        print(f'Epoch {epoch+1} - Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f} (Seg: {test_seg_loss:.4f}, Count: {test_count_loss:.4f})')
        
        # Save best model based on test loss
        if test_loss < best_loss:
            best_loss = test_loss
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, best_loss, best_model_path)
            print(f'New best model saved with test loss: {best_loss:.4f}')
    
    # Save final model
    final_model_path = os.path.join(save_dir, 'final_model.pth')
    save_checkpoint(model, optimizer, num_epochs-1, test_loss, final_model_path)
    print(f'Final model saved: {final_model_path}')

if __name__ == '__main__':
    main()