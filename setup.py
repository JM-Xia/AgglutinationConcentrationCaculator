from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="agglutination_calculator",
    version="0.1.0",
    author="Gabrielle",
    description="A tool for analyzing agglutination patterns and calculating concentrations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=['tests']),
    package_data={
        'AgglutinationConcentration': [
            'trained_models/*',
            'data/*',
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.2",
        "pillow>=9.0.0",
        "matplotlib>=3.4.0",
        "joblib>=1.1.0",
        "opencv-python>=4.5.0",
        "PyQt5>=5.15.0",
    ],
    entry_points={
        'console_scripts': [
            'agglutination-analyze=AgglutinationConcentrationCaculator.main:main',
            'agglutination-gui=AgglutinationConcentrationCaculator.GUI.main_window:launch_gui',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
