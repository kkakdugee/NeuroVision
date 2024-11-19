import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import os

def visualize_counting_process(image_path, mask_path):
    """
    Visualize how the automatic neuron counting works:
    1. Original image
    2. Binary mask
    3. Labeled components
    4. Original image with counted neurons marked
    """
    # Load images
    image = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'))
    
    # Create binary mask
    binary_mask = mask > 127
    
    # Label connected components
    labeled_mask, num_neurons = ndimage.label(binary_mask)
    
    # Create colored labels for visualization
    colored_labels = np.zeros_like(image)
    for i in range(1, num_neurons + 1):
        # Generate random color for each neuron
        color = np.random.randint(50, 255, 3)
        colored_labels[labeled_mask == i] = color
    
    # Create overlay
    overlay = image.copy()
    mask_overlay = colored_labels.astype(bool).any(axis=2)
    overlay[mask_overlay] = overlay[mask_overlay] * 0.3 + colored_labels[mask_overlay] * 0.7
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary mask
    axes[0, 1].imshow(binary_mask, cmap='gray')
    axes[0, 1].set_title('Binary Mask (Ground Truth)')
    axes[0, 1].axis('off')
    
    # Labeled components
    labeled_display = axes[1, 0].imshow(labeled_mask, cmap='nipy_spectral')
    axes[1, 0].set_title(f'Labeled Components\n(Count: {num_neurons} neurons)')
    axes[1, 0].axis('off')
    plt.colorbar(labeled_display, ax=axes[1, 0])
    
    # Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Original Image with Counted Neurons')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('neuron_counting_visualization.png')
    plt.close()

def visualize_random_samples(image_dir, mask_dir, num_samples=3):
    """
    Visualize multiple random samples from the dataset
    """
    # Get list of images
    images = sorted(os.listdir(image_dir))
    
    # Select random samples
    sample_indices = np.random.choice(len(images), num_samples, replace=False)
    
    for idx in sample_indices:
        image_path = os.path.join(image_dir, images[idx])
        mask_path = os.path.join(mask_dir, images[idx])
        
        print(f"\nProcessing sample {idx + 1}:")
        visualize_counting_process(image_path, mask_path)

def main():
    # Set paths to your dataset
    image_dir = './dataset/all_images/images/'
    mask_dir = './dataset/all_masks/masks/'
    
    print("Creating visualizations for random samples...")
    visualize_random_samples(image_dir, mask_dir)
    print("\nVisualizations saved as 'neuron_counting_visualization.png'")

if __name__ == '__main__':
    main()