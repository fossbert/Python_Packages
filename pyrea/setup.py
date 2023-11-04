# Import required functions
from setuptools import setup, find_packages
# Call setup function
setup(
    author="H. Carlo Maurer",
    description="Tools for gene set enrichment functionality",
    name="pyrea",
    version="0.1.0",
    packages=find_packages('pyrea'),
    package_dir={'pyrea': 'pyrea'},
    include_package_data=True,
    package_data={'pyrea': ['data/*.json']},
    install_requires=['pandas>=1.3.1', 
    'scipy>=1.6.2',
    'numpy>=1.20.3', 
    'statsmodels>=0.12.2',
    'matplotlib>=3.4.2', 
    ],
    python_requires='>=3.8',
    )