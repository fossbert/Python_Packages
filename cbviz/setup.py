# Import required functions
from setuptools import setup, find_packages
# Call setup function

setup(
    author="H. Carlo Maurer",
    description="Vizualisation tools",
    name="cbviz",
    version="0.1.0",
    packages=find_packages('cbviz'),
    install_requires=['pandas>=1.3.4', 
    'scipy>=1.7.1',
    'numpy>=1.20.3', 
    'statsmodels>=0.12.2',
    'matplotlib>=3.5.0' 
    ],
    python_requires='>=3.8',
    )