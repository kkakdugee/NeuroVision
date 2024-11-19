import torch
import os

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
    
    return total_loss / num_batches

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
    
    return (total_loss / num_batches, 
            total_seg_loss / num_batches, 
            total_count_loss / num_batches)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved: {path}')