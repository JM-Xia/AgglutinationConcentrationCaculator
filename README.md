# AgglutinationConcentration

A tool for analyzing agglutination patterns and predicting concentrations.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AgglutinationConcentrationCalculator.git
cd AgglutinationConcentrationCalculator

# Install the package
pip install -e .

## Requirements

torch>=2.0.1
torchvision>=0.15.2
numpy>=1.21.0
scikit-learn>=1.0.2
pillow>=9.0.0
matplotlib>=3.4.0
joblib>=1.1.0

Usage

Train the model:

bashCopypython -m src.AgglutinationConcentrationCalculator.main --img_dir "path/to/images" --batch_size 32

Use the GUI:

bashCopypython -m src.AgglutinationConcentrationCalculator.gui.main_window

## Structure
📦 AgglutinationConcentrationCalculator
┣ 📂 src
┃ ┗ 📂 AgglutinationConcentrationCalculator
┃   ┣ 📜 init.py
┃   ┣ 📜 main.py           # Training script
┃   ┣ 📜 utils.py          # Utility functions
┃   ┣ 📂 data
┃   ┃ ┣ 📜 init.py
┃   ┃ ┣ 📜 dataset1.py
┃   ┃ ┗ 📜 Concentration.py # Dataset handling
┃   ┣ 📂 models
┃   ┃ ┣ 📜 init.py
┃   ┃ ┣ 📜 CNN.py
┃   ┃ ┣ 📜 SVR.py
┃   ┃ ┗ 📜 ResRF.py     # Hybrid model implementation
┃   ┗ 📂 gui
┃     ┣ 📜 init.py
┃     ┣ 📜 main_window.py   # GUI main window
┃     ┗ 📂 components
┃       ┣ 📜 init.py
┃       ┗ 📜 image_viewer.py # Image viewing component
┣ 📂 tests                  # Unit tests
┣ 📜 setup.py
┣ 📜 requirements.txt
┗ 📜 README.md

Example demo
![63d286770fbec84aab1fe863e682c07](https://github.com/user-attachments/assets/42273d04-5bac-4f66-9ba0-1ce1469733f8)
![219b238bfc6b7f54c328e96730404d3](https://github.com/user-attachments/assets/9840a495-806f-4578-93a7-e7c171aefb3d)

