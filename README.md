# AgglutinationConcentration

A tool for analyzing agglutination patterns and predicting concentrations.

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

AgglutinationConcentrationCaculator/
├── setup.py                    
├── requirements.txt           
├── README.md                 
└── AgglutinationConcentrationCaculator/   
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── Concentration.py
    │   ├── dataset1.py
    │   └── delete.py
    ├── models/
    │   ├── __init__.py
    │   ├── CNN.py
    │   ├── CNN1.py
    │   ├── ResRF.py
    │   ├── SVR.py
    │   └── VGG.py
    ├── utils/
    │   ├── __init__.py
    │   └── visualization.py
    └── main.py    

Example
![image](https://github.com/user-attachments/assets/d5b85d2f-ff3f-481b-aa3c-cba85a67537b)
![image](https://github.com/user-attachments/assets/9dcc29e4-39f7-4a38-920c-fc4e9c64d7d3)
