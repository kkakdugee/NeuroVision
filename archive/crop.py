from helper import *

def crop_image_and_mask(image, mask, crop_size=(512, 512), stride=(400, 400)):
    """
    Crop image and mask into overlapping patches.
    
    Args:
        image: Input image (H, W, C)
        mask: Input mask (H, W)
        crop_size: Tuple of (height, width) for crops
        stride: Tuple of (height, width) for stride between crops
    
    Returns:
        List of (cropped_image, cropped_mask) pairs
    """
    crops = []
    h, w = image.shape[:2]
    
    # Calculate steps
    h_steps = range(0, h - crop_size[0] + stride[0], stride[0])
    w_steps = range(0, w - crop_size[1] + stride[1], stride[1])
    
    # Calculate total number of crops for progress bar
    total_crops = len(h_steps) * len(w_steps)
    
    with tqdm(total=total_crops, desc='Cropping patches') as pbar:
        for y in h_steps:
            for x in w_steps:
                # Ensure we don't go out of bounds
                end_y = min(y + crop_size[0], h)
                end_x = min(x + crop_size[1], w)
                start_y = end_y - crop_size[0]
                start_x = end_x - crop_size[1]
                
                crop_img = image[start_y:end_y, start_x:end_x]
                crop_mask = mask[start_y:end_y, start_x:end_x]
                
                crops.append((crop_img, crop_mask))
                pbar.update(1)
    
    return crops

def main(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, 
         crop_size=(512, 512), stride=(400, 400)):
    """
    Process all images in a directory.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(input_img_dir))
    counter = 0
    
    # Main progress bar for processing images
    for img_file in tqdm(image_files, desc='Processing images'):
        # Read image and mask
        image = cv2.imread(os.path.join(input_img_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(input_mask_dir, img_file))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)[:,:,0]  # Take first channel
        
        # Get crops
        crops = crop_image_and_mask(image, mask, crop_size, stride)
        
        # Save crops with progress bar
        with tqdm(total=len(crops), desc=f'Saving crops for {img_file}') as pbar:
            for img_crop, mask_crop in crops:
                plt.imsave(os.path.join(output_img_dir, f'{counter}.tiff'), img_crop)
                plt.imsave(os.path.join(output_mask_dir, f'{counter}.tiff'), 
                          mask_crop, cmap='gray')
                counter += 1
                pbar.update(1)

if __name__ == "__main__":
    
    main(train_val_images, train_val_masks, crop_images, crop_masks)