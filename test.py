# tests/test_utils.py
import unittest
import os
import torch
import tempfile
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from src.AgglutinationConcentrationCaculator.models.ResRF import ResNetRF
from src.utils import (
    load_image,
    get_device,
    save_prediction_result
)
from src.AgglutinationConcentrationCaculator.GUI.main_window import MainWindow
from src.AgglutinationConcentrationCaculator.data.Concentration import ConcentrationDataset
from PyQt5.QtWidgets import QApplication
import shutil

class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create temporary directory for test files
        cls.temp_dir = tempfile.mkdtemp()

        # Create test image
        cls.test_image_path = os.path.join(cls.temp_dir, "test_image.jpg")
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image).save(cls.test_image_path)

    def test_load_image(self):
        """Test image loading functionality."""
        # Test valid image loading
        image = load_image(self.test_image_path)
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (1, 3, 256, 256))

        # Test non-existent image
        with self.assertRaises(FileNotFoundError):
            load_image("nonexistent.jpg")

    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        self.assertIsInstance(device, torch.device)

    def test_save_prediction_result(self):
        """Test saving prediction results."""
        save_path = os.path.join(self.temp_dir, "predictions.txt")
        prediction = 0.5
        save_prediction_result(prediction, save_path)

        # Check if file exists and content is correct
        self.assertTrue(os.path.exists(save_path))
        with open(save_path, 'r') as f:
            content = f.read()
            self.assertIn("0.5000", content)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.temp_dir)


class TestGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize QApplication once for all tests."""
        cls.app = QApplication([])
        cls.window = MainWindow()

    def test_window_initialization(self):
        """Test window initialization."""
        self.assertIsNotNone(self.window)
        self.assertEqual(
            self.window.windowTitle(),
            "Agglutination Concentration Calculator"
        )

    def test_image_viewer(self):
        """Test image viewer component."""
        self.assertIsNotNone(self.window.image_viewer)

    def test_buttons(self):
        """Test button creation and properties."""
        self.assertTrue(hasattr(self.window, 'load_button'))
        self.assertTrue(hasattr(self.window, 'analyze_button'))

    def test_result_label(self):
        """Test result label initialization."""
        self.assertTrue(hasattr(self.window, 'result_label'))
        self.assertEqual(
            self.window.result_label.text(),
            "Concentration: Not analyzed"
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.window.close()


class TestResNetRF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.temp_dir = tempfile.mkdtemp()

        # Create test images directory
        cls.test_img_dir = os.path.join(cls.temp_dir, "test_images")
        os.makedirs(cls.test_img_dir)

        # Create a few test images
        for i in range(5):
            img_path = os.path.join(cls.test_img_dir, f"test_image_{i}.jpg")
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(test_image).save(img_path)

    def test_model_training(self):
        """Test ResNetRF model training."""
        # Setup data
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Create test dataset
        dataset = ConcentrationDataset(
            img_dir=self.test_img_dir,
            transform=transform
        )

        # Create dataloaders
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

        # Train model
        feature_extractor, rf_model, scaler = ResNetRF(train_loader, val_loader)

        # Test that components are created correctly
        self.assertIsNotNone(feature_extractor)
        self.assertIsNotNone(rf_model)
        self.assertIsNotNone(scaler)

    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""

        shutil.rmtree(cls.temp_dir)


if __name__ == '__main__':
    unittest.main()