import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
import cv2
from scipy import ndimage
import os
from lime import lime_image
from skimage.segmentation import mark_boundaries

from test_model import *

class TestMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.mae = 0.0
        self.rmse = 0.0
        self.r2 = 0.0
        self.dice_scores = []
        self.iou_scores = []
        self.count_errors = []
        self.true_counts = []
        self.pred_counts = []
        self.total_batches = 0
    
    def update(self, pred_counts, true_counts, pred_masks, true_masks):
        # Store predictions and true values
        pred_counts_np = pred_counts.cpu().numpy()
        true_counts_np = true_counts.cpu().numpy()
        
        self.true_counts.extend(true_counts_np)
        self.pred_counts.extend(pred_counts_np)
        self.count_errors.extend(abs(pred_counts_np - true_counts_np))
        
        # Segmentation metrics
        pred_masks = pred_masks.cpu().numpy() > 0.5
        true_masks = true_masks.cpu().numpy() > 0.5
        
        for pred, true in zip(pred_masks, true_masks):
            # Dice score
            intersection = np.logical_and(pred, true).sum()
            union = pred.sum() + true.sum()
            dice = 2 * intersection / (union + 1e-6)
            self.dice_scores.append(dice)
            
            # IoU score
            union = np.logical_or(pred, true).sum()
            iou = intersection / (union + 1e-6)
            self.iou_scores.append(iou)
        
        self.total_batches += 1
    
    def compute(self):
        # Convert lists to numpy arrays for computation
        true_counts = np.array(self.true_counts)
        pred_counts = np.array(self.pred_counts)
        
        # Calculate R² score
        r2 = r2_score(true_counts, pred_counts)
        
        # Calculate accuracy within different margins
        acc_exact = np.mean(np.abs(pred_counts - true_counts) < 1) * 100
        acc_within_1 = np.mean(np.abs(pred_counts - true_counts) <= 1) * 100
        acc_within_2 = np.mean(np.abs(pred_counts - true_counts) <= 2) * 100
        
        return {
            'MAE': np.mean(self.count_errors),
            'RMSE': np.sqrt(np.mean(np.square(self.count_errors))),
            'R² Score': r2,
            'Mean Dice': np.mean(self.dice_scores),
            'Mean IoU': np.mean(self.iou_scores),
            'Accuracy (exact)': acc_exact,
            'Accuracy (±1 neuron)': acc_within_1,
            'Accuracy (±2 neurons)': acc_within_2
        }

def explain_prediction(model, image, device):
    """Generate LIME explanation for the model's prediction"""
    
    def batch_predict(images):
        model.eval()
        batch = torch.stack([torch.from_numpy(img.transpose(2, 0, 1)).float() for img in images])
        batch = batch.to(device)
        with torch.no_grad():
            masks, counts = model(batch)
        return counts.cpu().numpy()
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.permute(1, 2, 0).cpu().numpy(), 
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    
    return explanation

def visualize_prediction_with_lime(image, true_mask, pred_mask, true_count, pred_count, 
                                 explanation, save_path=None):
    """Visualize the predictions with LIME explanation"""
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(141)
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(142)
    plt.imshow(true_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f'Ground Truth\nCount: {true_count:.0f}')
    plt.axis('off')
    
    # Predicted mask
    plt.subplot(143)
    plt.imshow(pred_mask.squeeze().cpu().numpy(), cmap='gray')
    plt.title(f'Prediction\nCount: {pred_count:.1f}')
    plt.axis('off')
    
    # LIME explanation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=True, 
        num_features=10, 
        hide_rest=False
    )
    plt.subplot(144)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title('LIME Explanation\n(Important Regions)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def test_model(model, test_loader, device, save_dir='test_results'):
    """Test the model and compute metrics"""
    model.eval()
    metrics = TestMetrics()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (images, masks, counts) in enumerate(tqdm(test_loader, desc='Testing')):
            images, masks, counts = images.to(device), masks.to(device), counts.to(device)
            
            # Forward pass
            pred_masks, pred_counts = model(images)
            
            # Update metrics
            metrics.update(pred_counts.squeeze(), counts, pred_masks, masks)
            
            # Visualize and explain first image of some batches
            if batch_idx % 10 == 0:
                # Generate LIME explanation
                explanation = explain_prediction(model, images[0], device)
                
                # Visualize with LIME
                visualize_prediction_with_lime(
                    images[0],
                    masks[0],
                    pred_masks[0],
                    counts[0].item(),
                    pred_counts[0].item(),
                    explanation,
                    save_path=os.path.join(save_dir, f'prediction_explained_{batch_idx}.png')
                )
    
    # Compute and print metrics
    results = metrics.compute()
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for metric, value in results.items():
        print(f"{metric}: {value:.2f}")
    
    # Plot prediction vs true count scatter
    plt.figure(figsize=(10, 10))
    plt.scatter(metrics.true_counts, metrics.pred_counts, alpha=0.5)
    plt.plot([0, max(metrics.true_counts)], [0, max(metrics.true_counts)], 'r--')
    plt.xlabel('True Count')
    plt.ylabel('Predicted Count')
    plt.title('Prediction vs Ground Truth')
    plt.savefig(os.path.join(save_dir, 'prediction_scatter.png'))
    plt.close()
    
    return results

def create_test_pipeline(model_path, dataset, batch_size=8, test_split=0.2):
    """
    Create a complete test pipeline including data splitting and model loading.
    """
    # Split dataset
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuronCounter().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:  # If loading from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
    else:  # If loading just the state dict
        model.load_state_dict(checkpoint)
    
    return model, test_loader

def analyze_errors(model, test_loader, device, num_samples=5):
    """
    Analyze where the model makes mistakes.
    """
    model.eval()
    errors = []
    
    with torch.no_grad():
        for images, masks, counts in test_loader:
            images, masks, counts = images.to(device), masks.to(device), counts.to(device)
            pred_masks, pred_counts = model(images)
            
            # Calculate errors
            count_errors = abs(pred_counts.squeeze().cpu() - counts.cpu())
            
            # Store results for analysis
            for i in range(len(images)):
                errors.append({
                    'image': images[i],
                    'true_mask': masks[i],
                    'pred_mask': pred_masks[i],
                    'true_count': counts[i].item(),
                    'pred_count': pred_counts[i].item(),
                    'error': count_errors[i].item()
                })
    
    # Sort by error and analyze worst cases
    errors.sort(key=lambda x: x['error'], reverse=True)
    worst_cases = errors[:num_samples]
    
    print("\nWorst Predictions Analysis:")
    for i, error in enumerate(worst_cases):
        # Generate LIME explanation only for worst cases
        explanation = explain_prediction(model, error['image'], device)
        
        print(f"\nCase {i+1}:")
        print(f"True Count: {error['true_count']:.0f}")
        print(f"Predicted Count: {error['pred_count']:.1f}")
        print(f"Absolute Error: {error['error']:.2f}")
        
        # Visualize with LIME explanation
        visualize_prediction_with_lime(
            error['image'],
            error['true_mask'],
            error['pred_mask'],
            error['true_count'],
            error['pred_count'],
            explanation,
            save_path=f'error_analysis_{i}.png'
        )
    
    return errors

def main():
    # Example usage
    batch_size = 8
    model_path = './model_checkpoints/final_model.pth'
    
    # Setup dataset (using the same dataset class from training)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset = NeuronDataset(
        image_dir='./dataset/train_val/cropped/images/',
        mask_dir='./dataset/train_val/cropped/masks/',
        transform=transform
    )
    
    # Create test pipeline
    model, test_loader = create_test_pipeline(
        model_path=model_path,
        dataset=dataset,
        batch_size=batch_size
    )
    
    # Run tests
    print("Running standard tests...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = test_model(model, test_loader, device)
    
    # Analyze errors
    print("\nAnalyzing errors...")
    error_analysis = analyze_errors(model, test_loader, device)

if __name__ == '__main__':
    main()