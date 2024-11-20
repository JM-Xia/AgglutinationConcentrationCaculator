from setuptools import setup, find_packages

setup(
    name="agglutinationconcentration_caculator",
    version="0.1.0",
    author="Gabrielle",
    description="An agglutination pattern analysis tool for concentration prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.2",
        "pillow>=9.0.0",
        "matplotlib>=3.4.0",
        "joblib>=1.1.0",
    ],
    entry_points={
        'console_scripts': [
            'agglutination-train=agglutination_analyzer.main:main',
        ],
    },
    python_requires='>=3.9',
)