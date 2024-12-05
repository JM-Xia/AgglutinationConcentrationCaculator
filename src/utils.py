import os
import cv2
import logging
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

"""
Utility Functions for Agglutination Concentration Calculator

This module provides utility functions supporting both the training pipeline and GUI application 
of the agglutination concentration calculator. It includes functions for:

Core Functionalities:
   - Device management (CPU/GPU selection)
   - Model file validation
   - Logging setup and configuration
"""

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logger()


def load_image(image_path):
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found at {image_path}")

    try:
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(image).unsqueeze(0)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Error processing image: {str(e)}")


def data_augmentation(image):

    augmented = []
    # Original image
    augmented.append(image)

    # Horizontal flip
    augmented.append(transforms.functional.hflip(image))

    # Vertical flip
    augmented.append(transforms.functional.vflip(image))

    return augmented


def save_prediction_result(prediction, save_path):

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(f"Predicted concentration: {prediction:.4f}")
        logger.info(f"Results saved to {save_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise


def ensure_model_loaded(model_path):

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return False
    return True


def get_device():

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path, device=None):

    if device is None:
        device = get_device()

    try:
        model = torch.load(model_path, map_location=device)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def preprocess_image_batch(image_paths):

    processed_images = []
    for path in image_paths:
        try:
            processed_images.append(load_image(path))
        except Exception as e:
            logger.warning(f"Error processing image {path}: {str(e)}")
            continue

    if not processed_images:
        raise ValueError("No images were successfully processed")

    return torch.cat(processed_images, dim=0)


if __name__ == "__main__":
    logger.info("Testing utils functions...")
    try:
        device = get_device()
        logger.info(f"Using device: {device}")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")