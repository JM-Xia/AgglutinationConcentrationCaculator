# AgglutinationConcentration
This project implements a machine learning-based software system for analyzing agglutination patterns in images and predicting sample concentrations.

Requirements

Python 3.9
PyTorch 2.0.1
scikit-learn
PIL (Pillow)
tkinter
numpy
matplotlib

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

Example
![image](https://github.com/user-attachments/assets/d5b85d2f-ff3f-481b-aa3c-cba85a67537b)
![image](https://github.com/user-attachments/assets/9dcc29e4-39f7-4a38-920c-fc4e9c64d7d3)
