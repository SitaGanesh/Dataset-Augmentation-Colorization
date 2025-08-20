<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Image Colorization with Dataset Augmentation

This project implements an end-to-end pipeline for colorizing grayscale images using PyTorch, leveraging data augmentation to improve model performance. It includes preprocessing, augmentation experiments, training, evaluation, and a GUI for interactive colorization.

## Project Overview

- **Goal**: Enhance image colorization quality by training models on both original and augmented datasets.
- **Key Components**
    - Data preprocessing and LAB color conversion
    - Comprehensive augmentation techniques (geometric, photometric, advanced)
    - U-Net-based models (baseline and augmented)
    - Training pipelines with mixed-precision support
    - Evaluation metrics (PSNR, SSIM, MSE, MAE) and comparison tools
    - Interactive GUI for real-time colorization


## Folder Structure

```
colorization_project/
│
├── data/                          # Image data
│   ├── raw/                       # YOUR downloaded color photos
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── processed/                 # Auto-generated resized LAB inputs
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── augmented/                 # Auto-generated augmented images
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/                        # Trained model storage
│   ├── checkpoints/              # Checkpoints during training
│   ├── baseline_model/           # Model trained without augmentation
│   └── augmented_model/          # Model trained with augmentation
│
├── results/                       # Outputs and logs
│   ├── baseline/                 # Baseline evaluation and plots
│   ├── augmented/                # Augmented evaluation and plots
│   ├── comparisons/              # Baseline vs. augmented comparisons
│   └── logs/                     # Training logs and tensorboard runs
│
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_preprocessing.py     # LAB conversion and dataloaders
│   ├── data_augmentation.py      # Augmentation pipelines
│   ├── model_architecture.py     # U-Net and ResNet-U-Net definitions
│   ├── training.py               # Trainer and training loops
│   ├── evaluation.py             # Metrics computation and visualization
│   ├── utils.py                  # Utilities (dir setup, reporting)
│   └── gui.py                    # Interactive CustomTkinter GUI
│
├── notebooks/                     # Jupyter walkthroughs
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_augmentation_experiments.ipynb
│   ├── 04_model_training_baseline.ipynb
│   ├── 05_model_training_augmented.ipynb
│   ├── 06_evaluation_comparison.ipynb
│   └── 07_gui_demo.ipynb
│
├── config/                        # Configuration YAML files
│   ├── config.yaml               # Model, training, data settings
│   └── augmentation_config.yaml  # Augmentation strategies and probabilities
│
├── setup.py                       # Project initialization script
├── requirements.txt               # pip install dependencies
├── .gitignore                     # Files and folders to ignore in Git
└── README.md                      # This file
```


## Models

- **Baseline Model**
    - U-Net architecture
    - Trained on original grayscale→color pairs
    - Saved under `models/baseline_model/`
- **Augmented Model**
    - Same architecture, trained with augmented data
    - Augmentation strategies configurable in `augmentation_config.yaml`
    - Saved under `models/augmented_model/`


## GUI Functionality

1. **Load Model**: Select a trained model (baseline or augmented)
2. **Load Image**: Choose a color photo; converted to grayscale for input
3. **Colorize**: Run the model to produce a colorized output
4. **Save Result**: Export the colorized image as PNG or JPEG
5. **Fallback GUI**: If CustomTkinter isn’t available, a basic Tkinter interface is used

## Local Setup

1. **Create and activate virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

2. **Upgrade pip**

```bash
python -m pip install --upgrade pip
```

3. **Install dependencies**

```bash
pip install torch torchvision torchaudio \
  numpy opencv-python Pillow matplotlib seaborn scikit-learn scikit-image \
  jupyter ipywidgets tqdm tensorboard PyYAML imageio albumentations wandb \
  pandas scipy customtkinter colorama termcolor rich pytest black flake8 isort
```

4. **Initialize project structure (optional)**

```bash
python setup.py
```

5. **Download color images** into `data/raw/` (only this folder)
6. **Run notebooks in sequence**:
    - `01_data_exploration.ipynb` → analyze your images
    - `02_data_preprocessing.ipynb` → preprocess LAB data
    - `03_augmentation_experiments.ipynb` → test augmentations
    - `04_model_training_baseline.ipynb` → train baseline
    - `05_model_training_augmented.ipynb` → train augmented
    - `06_evaluation_comparison.ipynb` → compare models
    - `07_gui_demo.ipynb` → test via GUI

## Requirements to Interact with Models

- Python 3.8+
- PyTorch (CPU or GPU)
- CustomTkinter (for GUI) or fallback Tkinter
- Jupyter Notebook (for interactive notebooks)
- Adequate disk space for dataset and models (10+ GB)

***

Follow these steps to explore data, train models, evaluate results, and interact with colorization models via the GUI. Enjoy experimenting with dataset augmentation to achieve vibrant, accurate colorizations!

