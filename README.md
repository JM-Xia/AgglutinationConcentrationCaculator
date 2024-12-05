# AgglutinationConcentration

A tool for analyzing agglutination patterns and predicting concentrations.

## Installation

```bash
pip install git+https://github.com/JM-Xia/AgglutinationConcentrationCaculator.git

Requirements

torch>=2.0.1
torchvision>=0.15.2
numpy>=1.21.0
scikit-learn>=1.0.2
pillow>=9.0.0
matplotlib>=3.4.0
joblib>=1.1.0

Usage
1. Training the Model
Run the training script with the Random Forest model:
bashCopypython main.py --model RandomForest 
Training parameters can be customized:

--model: Model type (default: "RandomForest")
--batch_size: Batch size for training (default: 32)
--img_dir: Directory containing training images

2. Using the GUI
After training, run the visualization interface:
bashCopypython visualization.py
The GUI provides:

Image upload functionality
Concentration prediction
Result saving

📦 AgglutinationConcentrationCaculator
┣ 📂 src
┃ ┗ 📂 AgglutinationConcentrationCaculator
┃   ┣ 📜 init.py
┃   ┣ 📜 main.py
┃   ┣ 📜 utils.py
┃   ┗ 📂 gui
┃     ┣ 📜 init.py
┃     ┣ 📜 main_window.py
┃     ┗ 📂 components
┃       ┣ 📜 init.py
┃       ┗ 📜 image_viewer.py
┣ 📂 tests
┃ ┣ 📜 init.py
┃ ┗ 📜 test_utils.py
┣ 📜 setup.py
┣ 📜 requirements.txt
┣ 📜 README.md
┗ 📜 LICENSE

Example demo
![image](https://github.com/user-attachments/assets/d5b85d2f-ff3f-481b-aa3c-cba85a67537b)
![image](https://github.com/user-attachments/assets/9dcc29e4-39f7-4a38-920c-fc4e9c64d7d3)
