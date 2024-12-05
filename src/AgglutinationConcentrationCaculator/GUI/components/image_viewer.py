from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np


class ImageViewer(QLabel):
    def __init__(self):
        super().__init__()
        self.image = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 400)
        self.setStyleSheet("border: 2px solid gray")

    def load_image(self, file_path):
        # Load image with PIL
        self.image = Image.open(file_path)

        # Convert to QPixmap for display
        pixmap = self.convert_to_pixmap(self.image)

        # Scale pixmap to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.setPixmap(scaled_pixmap)

    def convert_to_pixmap(self, pil_image):
        # Convert PIL image to QPixmap
        if pil_image.mode == "RGB":
            r, g, b = pil_image.split()
            im_array = np.dstack((r, g, b))
        elif pil_image.mode == "RGBA":
            r, g, b, a = pil_image.split()
            im_array = np.dstack((r, g, b, a))
        else:
            im_array = np.array(pil_image)

        height, width = im_array.shape[:2]
        bytes_per_line = im_array.strides[0]

        image = QImage(
            im_array.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        return QPixmap.fromImage(image)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap():
            scaled_pixmap = self.pixmap().scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)