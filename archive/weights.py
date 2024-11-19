from helper import *

def create_weight_mask(mask, sigma=25):
    """
    Create weight mask emphasizing boundaries and cell separation.
    
    Args:
        mask: Binary mask image
        sigma: Gaussian decay parameter
    Returns:
        Weighted mask emphasizing boundaries
    """
    # Clean up mask
    mask = mask.astype(bool)
    mask = remove_small_objects(mask, min_size=100)
    mask = remove_small_holes(mask, 200)
    mask = mask.astype(np.uint8) * 255
    
    # Get inverted mask and dilated mask
    inverted_mask = cv2.bitwise_not(mask)
    dilated = cv2.dilate(mask, np.ones((100, 100), np.uint8))
    
    # Create boundary region
    boundary_region = cv2.bitwise_and(dilated, inverted_mask)
    
    # Initialize weight mask
    weight_mask = np.zeros_like(mask, dtype=np.float32)
    
    # Label connected components
    labeled_mask, num_labels = ndimage.label(mask)
    
    if num_labels < 1:
        return np.ones_like(mask, dtype=np.float32)
    
    # Process each object
    for obj in ndimage.find_objects(labeled_mask):
        if obj is None:
            continue
            
        # Create object mask
        obj_mask = np.zeros_like(labeled_mask)
        obj_mask[obj] = labeled_mask[obj]
        obj_mask = np.clip(obj_mask, 0, 1).astype(np.uint8) * 255
        
        # Calculate distance transform
        dist = ndimage.distance_transform_edt(cv2.bitwise_not(obj_mask))
        
        # Apply Gaussian decay
        weights = np.exp(-0.5 * (dist / sigma) ** 2)
        weights[dist == 0] = 1.0
        
        # Add to weight mask
        weight_mask = cv2.add(weight_mask, weights, mask=boundary_region)
    
    # Normalize weights
    weight_mask = np.clip(weight_mask, 1.0, None)
    weight_mask *= 1.5 * mask / 255
    
    return weight_mask

def main(input_masks_dir, output_weights_dir, sigma=25):
    """
    Process all masks in a directory.
    """
    os.makedirs(output_weights_dir, exist_ok=True)
    
    mask_files = sorted(os.listdir(input_masks_dir))
    
    for mask_file in tqdm(mask_files, desc='Creating weight masks'):
        # Read mask
        mask = cv2.imread(os.path.join(input_masks_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        
        # Create weight mask
        weight_mask = create_weight_mask(mask, sigma)
        
        # Save weight mask
        output_path = os.path.join(output_weights_dir, mask_file)
        cv2.imwrite(output_path, (weight_mask * 255).astype(np.uint8))

if __name__ == "__main__":
    
    main(crop_masks, crop_weighted_masks, )