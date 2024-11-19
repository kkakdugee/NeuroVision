import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import NeuronCounter
import os
from scipy import ndimage
import io

def load_model(model_path):
    """Load the trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuronCounter(input_size=128).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def process_image(image_path):
    """Process image for model input"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

def plot_results(original_image, segmentation, predicted_count):
    """Create matplotlib figure with results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    ax1.imshow(original_image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot segmentation
    ax2.imshow(segmentation.squeeze(), cmap='gray')
    ax2.set_title(f'Segmentation\nPredicted Count: {predicted_count:.1f} neurons')
    ax2.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

def main():
    st.title('Neuron Counter')
    st.write('Upload or select a microscopy image to segment and count neurons.')
    
    # Model loading
    model_path = './model_checkpoints/final_model.pth'
    if not os.path.exists(model_path):
        st.error("Model file not found. Please make sure the model is properly saved in the deployment directory.")
        return
    
    model, device = load_model(model_path)
    
    # Image selection
    option = st.radio(
        "Choose input method:",
        ('Upload Image', 'Select Test Image')
    )
    
    image_file = None
    if option == 'Upload Image':
        image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    else:
        test_dir = './dataset/test/images/'
        if os.path.exists(test_dir):
            test_images = sorted(os.listdir(test_dir))
            if test_images:
                selected_image = st.selectbox(
                    'Select a test image:',
                    test_images
                )
                image_file = os.path.join(test_dir, selected_image)
            else:
                st.error("No test images found in the test directory.")
                return
        else:
            st.error("Test directory not found.")
            return
    
    if image_file:
        # Process image and get predictions
        image, image_tensor = process_image(image_file)
        
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            seg_out, count_out = model(image_tensor)
            
            # Get binary segmentation
            seg_pred = (seg_out > 0.5).float()
            
            # Get predicted count
            predicted_count = count_out.item()
            
            # Calculate actual count using connected components
            seg_np = seg_pred.cpu().numpy().squeeze()
            labeled_mask, num_neurons = ndimage.label(seg_np > 0.5)
            
        # Plot results
        result_buf = plot_results(image, seg_pred.cpu().numpy(), predicted_count)
        
        # Display results
        st.image(result_buf, use_column_width=True)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Count (Regression)", f"{predicted_count:.1f}")
        with col2:
            st.metric("Connected Components Count", str(num_neurons))
        
        # Additional information
        st.write("### Analysis Details:")
        st.write(f"- Image size: {image.size}")
        st.write("- The segmentation map shows the detected neurons in white")
        st.write("- Two counting methods are shown:")
        st.write("  1. Direct regression prediction from the model")
        st.write("  2. Count of connected components in the segmentation mask")

if __name__ == '__main__':
    main()