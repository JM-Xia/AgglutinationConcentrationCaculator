import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QPushButton, QFileDialog, QLabel, QMessageBox,
                             QGridLayout, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from .components.image_viewer import ImageViewer
import torch
import joblib
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

"""
Main GUI Window for Agglutination Pattern Analysis Software.

This class implements the main window interface that provides:
- Image loading and display capabilities
- ResNet-RF model inference for concentration prediction
- Result visualization and saving

"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.concentration_value = None
        self.init_ui()
        self.load_models()

    def init_ui(self):
        self.setWindowTitle("Agglutination Pattern Analysis Software")
        self.setMinimumSize(1000, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QGridLayout(central_widget)

        # Create image display section
        self.create_image_section(main_layout)

        # Create control section
        self.create_control_section(main_layout)

        # Create results section
        self.create_results_section(main_layout)

    def create_image_section(self, layout):
        self.image_viewer = ImageViewer()
        layout.addWidget(self.image_viewer, 0, 0, 3, 3)

    def create_control_section(self, layout):
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)

        # Load Image button
        self.load_button = QPushButton("Upload Image")
        self.load_button.setMinimumHeight(40)
        self.load_button.setFont(QFont('Arial', 10))
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)

        # Analyze button
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setMinimumHeight(40)
        self.analyze_button.setFont(QFont('Arial', 10))
        self.analyze_button.clicked.connect(self.analyze_image)
        control_layout.addWidget(self.analyze_button)

        # Save Results button
        self.save_button = QPushButton("Save Results")
        self.save_button.setMinimumHeight(40)
        self.save_button.setFont(QFont('Arial', 10))
        self.save_button.clicked.connect(self.save_results)
        control_layout.addWidget(self.save_button)

        layout.addWidget(control_widget, 0, 3, 1, 1)

    def create_results_section(self, layout):
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        title_label = QLabel("Results")
        title_label.setFont(QFont('Arial', 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(title_label)

        self.result_label = QLabel("Concentration: Not analyzed")
        self.result_label.setFont(QFont('Arial', 10))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setWordWrap(True)
        results_layout.addWidget(self.result_label)

        layout.addWidget(results_widget, 1, 3, 2, 1)

    def load_models(self):
        try:
            model_dir = 'trained_models'
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Directory '{model_dir}' not found")

            # Load feature extractor
            self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.feature_extractor.fc = nn.Identity()
            self.feature_extractor.load_state_dict(
                torch.load(os.path.join(model_dir, 'feature_extractor.pth'))
            )

            # Load RF model and scaler
            self.rf_model = joblib.load(os.path.join(model_dir, 'rf_model.joblib'))
            self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))

            # Set up transform
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_extractor = self.feature_extractor.to(self.device)
            self.feature_extractor.eval()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load models: {str(e)}\n\nPlease ensure:\n"
                "1. You have run main.py to train the model\n"
                "2. All model files are in the 'trained_models' directory"
            )
            raise

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif *.bmp)"
        )
        if file_name:
            try:
                self.current_image_path = file_name
                self.image_viewer.load_image(file_name)
                self.result_label.setText("Concentration: Not analyzed")
                self.statusBar().showMessage(f"Loaded image: {os.path.basename(file_name)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")

    def analyze_image(self):
        if not self.image_viewer.image:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return

        try:
            self.statusBar().showMessage("Analyzing image...")

            # Prepare image
            image = self.transform(self.image_viewer.image).unsqueeze(0)
            image = image.to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(image)
                features = features.view(features.size(0), -1).cpu().numpy()

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get prediction
            self.concentration_value = self.rf_model.predict(features_scaled)[0]

            # Display result
            self.result_label.setText(
                f"Concentration: {self.concentration_value:.2e}"
            )
            self.statusBar().showMessage("Analysis completed")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.statusBar().showMessage("Analysis failed")

    def save_results(self):
        if not hasattr(self.image_viewer, 'image'):
            QMessageBox.warning(self, "Warning", "No image to save")
            return

        save_directory = "saved_results"
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        try:
            # Save image
            image_name = os.path.basename(self.current_image_path)
            save_path = os.path.join(save_directory, image_name)
            self.image_viewer.image.save(save_path)

            # Save prediction
            with open(os.path.join(save_directory, "concentration_data.txt"), "a") as file:
                file.write(f"{image_name}: Concentration = {self.concentration_value:.2e}\n")

            self.statusBar().showMessage("Results saved successfully")
            QMessageBox.information(self, "Success", "Results saved successfully")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")


def launch_gui():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()