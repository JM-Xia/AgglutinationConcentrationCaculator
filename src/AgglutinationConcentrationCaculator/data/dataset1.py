import cv2
import numpy as np
import os
import random
import tensorflow as tf

"""
Image Augmentation Script for Agglutination Pattern Dataset

Performs image augmentation including:
- Random cropping 
- Rotation (90°, 180°, 270°)
- Horizontal and vertical flips

The script takes input images and creates augmented versions to expand the training dataset.
"""

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def random_crop_image(image, crop_size):
    h, w, _ = image.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    cropped_image = image[top:top + crop_size, left:left + crop_size]
    return cropped_image


def save_img(path, img):
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def rotate_and_flip_images(cropped_image, base_filename, augmented_dir, crop_index):
    # Save the original cropped image with different name
    save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}.jpeg'), cropped_image)

    # Rotate images for 90, 180, and 270 degrees
    for angle in [90, 180, 270]:
        rotated_img = tf.image.rot90(cropped_image, k=angle // 90)
        save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}_rot{angle}.jpeg'), rotated_img.numpy())

    # Horizontal and vertical flips for original image
    flipped_img_h = tf.image.flip_left_right(cropped_image)
    save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}_hflip.jpeg'), flipped_img_h.numpy())

    flipped_img_v = tf.image.flip_up_down(cropped_image)
    save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}_vflip.jpeg'), flipped_img_v.numpy())

    # Only rotate 90 degrees and apply flips
    rotated_90_img = tf.image.rot90(cropped_image, k=1)
    save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}_rot90.jpeg'), rotated_90_img.numpy())

    # Horizontal and vertical flips for 90 degrees rotated image
    flipped_90_h = tf.image.flip_left_right(rotated_90_img)
    save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}_rot90_hflip.jpeg'), flipped_90_h.numpy())

    flipped_90_v = tf.image.flip_up_down(rotated_90_img)
    save_img(os.path.join(augmented_dir, f'{base_filename}_crop{crop_index}_rot90_vflip.jpeg'), flipped_90_v.numpy())

def process_images(image_dir, augmented_dir, crop_size, num_crops):
    os.makedirs(augmented_dir, exist_ok=True)
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img = load_image(img_path)
        if img is None:
            continue

        base_filename = os.path.splitext(img_file)[0]
        for i in range(num_crops):
            cropped_image = random_crop_image(img, crop_size)
            rotate_and_flip_images(cropped_image, base_filename, augmented_dir, i)

def main():
    image_dir = 'C:/Users/XiaQi/Documents/Individual_Project/Agglutination Pictures'
    augmented_dir = r"C:/Users/XiaQi/Documents/UW/bioen537/Augmented_Images"
    crop_size = 400
    num_crops = 12

    process_images(image_dir, augmented_dir, crop_size, num_crops)

if __name__ == '__main__':
    main()
