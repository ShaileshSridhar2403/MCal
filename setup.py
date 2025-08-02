"""Setup script for MCal package."""

from setuptools import setup, find_packages

# Read README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcal",
    version="0.1.0",
    author="MCal Team",
    author_email="mcal@example.com",
    description="A comprehensive framework for model calibration across vision, language, and tabular modalities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mcal",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.812",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "full": [
            "gurobi>=9.0.0",  # For advanced optimization
            "transformers>=4.0.0",  # For language model support
            "timm>=0.6.0",  # For vision model support
            "xgboost>=1.5.0",  # For tabular model support
        ],
    },
    entry_points={
        "console_scripts": [
            "mcal=mcal.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)