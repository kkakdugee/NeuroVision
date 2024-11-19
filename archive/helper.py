import os 
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from skimage.morphology import remove_small_objects, remove_small_holes

root = 'C:/Users/aaron/Desktop/GitHub Repositories/AI4ALL/'

all_images = root + 'dataset/all_images/images/'
all_masks = root + 'dataset/all_masks/masks/'

original_images = root + 'dataset/original_images/images/'
original_masks = root + 'dataset/original_masks/masks/'

new_images = root + 'dataset/new_images/images/'
new_masks = root + 'dataset/new_masks/masks/'

test_images = root + 'dataset/test/all_images/images'
test_masks = root + 'dataset/test/all_masks/masks/'

new_test_images = root + 'dataset/new_test/images/'
new_test_masks = root + 'dataset/new_test/masks/'

train_val_images = root + 'dataset/train_val/full_size/all_images/images/'
train_val_masks = root + 'dataset/train_val/full_size/all_masks/masks/'

crop_images = root + 'dataset/train_val/cropped/images/'
crop_masks = root + 'dataset/train_val/cropped/masks/'
crop_weighted_masks = root + 'dataset/train_val/cropped/weighted_masks'

aug_crop_images = root + 'dataset/train_val/crop_augumented/images/'
aug_crop_masks = root + 'dataset/train_val/crop_augumented/masks/'

aug_crop_images_split = root + 'dataset/train_val/crop_augumented_split_images/images/'
aug_crop_masks_split = root +'dataset/train_val/crop_augumented_splt_masks/masks/'

aug_crop_images_basic = root + 'dataset/train_val/crop_augumented_basic_images/images/'
aug_crop_masks_basic = root + 'dataset/train_val/crop_augumented_basic_masks/masks/'

aug_crop_images_basic_split = root + 'dataset/train_val/crop_augumented_basic_split_images/images/'
aug_crop_masks_basic_split = root + 'dataset/train_val/crop_augumented_basic_split_masks/masks/'

model_results = root + 'model_results/'

paths = [all_images, all_masks, train_val_images, train_val_masks, test_images, test_masks, crop_images, \
         crop_masks, crop_weighted_masks, aug_crop_images, aug_crop_masks, aug_crop_images_split, aug_crop_masks_split,\
         aug_crop_masks_basic, aug_crop_masks_basic, aug_crop_images_basic_split, aug_crop_masks_basic_split, model_results]

for path in paths:
    if not os.path.exists(path):
        os.makedirs(path)
