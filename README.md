# Image Colorization with Dataset Augmentation

A comprehensive deep learning project that colorizes grayscale images using U-Net architecture with dataset augmentation techniques to improve model performance.

## 🎯 Project Overview

This project demonstrates how **data augmentation** can significantly improve image colorization quality. We train two models:

- **Goal**: Enhance image colorization quality by training models on both original and augmented datasets.
- **Key Components**
    - Data preprocessing and LAB color conversion
    - Comprehensive augmentation techniques (geometric, photometric, advanced)
    - U-Net-based models (baseline and augmented)
    - Training pipelines with mixed-precision support
    - Evaluation metrics (PSNR, SSIM, MSE, MAE) and comparison tools
    - Interactive GUI for real-time colorization

- **Baseline Model**: Trained on original images only
- **Augmented Model**: Trained with diverse augmentations (rotation, brightness, noise, etc.)


### What This Project Does

1. **Converts color images to grayscale** for training input
2. **Trains neural networks** to predict missing color information
3. **Compares performance** between augmented vs non-augmented training
4. **Provides interactive GUI** for testing trained models

***

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM recommended
- 2GB+ free disk space


### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-username/image-colorization-augmentation.git
cd image-colorization-augmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```


### Step 2: Install Dependencies

```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install torch torchvision torchaudio numpy opencv-python Pillow matplotlib seaborn scikit-learn scikit-image jupyter ipywidgets tqdm tensorboard PyYAML imageio albumentations wandb pandas scipy customtkinter colorama termcolor rich pytest black flake8 isort
```


### Step 3: Download Training Images

Place **500-2000 colorful images** in `data/raw/train/`, `data/raw/val/`, and `data/raw/test/` folders.

**Google Search Prompts for Images:**

```
"high resolution colorful landscape photography"
"vibrant nature photography HD wallpaper"  
"professional portrait photography colorful background"
"colorful street art architecture photography"
"tropical sunset beach photography vibrant colors"
"colorful food photography restaurant quality"
```


### Step 4: Run the Project

```bash
# Start Jupyter Notebook
jupyter notebook

# Navigate to notebooks/ folder and run in order:
# 01_data_exploration.ipynb      -> Analyze your dataset
# 02_data_preprocessing.ipynb    -> Convert images to LAB format
# 03_augmentation_experiments.ipynb -> Test augmentation effects
# 04_model_training_baseline.ipynb  -> Train without augmentation
# 05_model_training_augmented.ipynb -> Train with augmentation
# 06_evaluation_comparison.ipynb    -> Compare both models
# 07_gui_demo.ipynb                 -> Interactive testing
```


***

## 📁 Project Structure

```
colorization_project/
│
├── data/                          # Image data pipeline
│   ├── raw/                       # YOUR downloaded color photos
│   │   ├── train/                 # Training images (70%)
│   │   ├── val/                   # Validation images (20%)
│   │   └── test/                  # Test images (10%)
│   ├── processed/                 # Auto-generated LAB format
│   └── augmented/                 # Auto-generated augmented images
│
├── src/                           # Core source code
│   ├── data_preprocessing.py      # LAB conversion & data loading
│   ├── data_augmentation.py       # Augmentation strategies
│   ├── model_architecture.py      # U-Net model definition
│   ├── training.py                # Training pipeline
│   ├── evaluation.py              # Metrics & evaluation
│   ├── utils.py                   # Helper functions
│   └── gui.py                     # Interactive GUI application
│
├── models/                        # Trained model storage
│   ├── baseline_model/            # Model without augmentation
│   └── augmented_model/           # Model with augmentation
│
├── results/                       # Training outputs & analysis
│   ├── baseline/                  # Baseline training results
│   ├── augmented/                 # Augmented training results
│   ├── comparisons/               # Model comparison charts
│   └── logs/                      # Training logs
│
├── notebooks/                     # Step-by-step tutorials
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_augmentation_experiments.ipynb
│   ├── 04_model_training_baseline.ipynb
│   ├── 05_model_training_augmented.ipynb
│   ├── 06_evaluation_comparison.ipynb
│   └── 07_gui_demo.ipynb
│
├── config/                        # Configuration files
│   ├── config.yaml               # Main project settings
│   └── augmentation_config.yaml  # Augmentation strategies
│
├── requirements.txt               # Python dependencies
└── README.md                      # This comprehensive guide
```


***

## 🧠 Model Architecture \& Technical Details

### U-Net Architecture

Our colorization model uses **U-Net**, a proven architecture for image-to-image translation:

- **Parameters**: 31,036,546 trainable parameters
- **Model Size**: 118.44 MB
- **Input**: Grayscale L channel (1×256×256)
- **Output**: Color AB channels (2×256×256)


### Image Processing Pipeline

#### 1. **Color Space Conversion (RGB → LAB)**

```
RGB Image → LAB Color Space
├── L Channel (Lightness): 0-100 → Grayscale information
├── A Channel (-128 to +127): Green-Red color axis  
└── B Channel (-128 to +127): Blue-Yellow color axis
```


#### 2. **Data Normalization**

```
L Channel: [0, 100] → [-1, 1]    # Model input (grayscale)
A Channel: [-128, 127] → [-1, 1] # Model target (color)
B Channel: [-128, 127] → [-1, 1] # Model target (color)
```


#### 3. **Model Processing Flow**

```
Input Color Image (RGB)
        ↓
Convert to LAB Color Space
        ↓
Split: L Channel (input) + AB Channels (target)
        ↓
Normalize values to [-1, 1]
        ↓
U-Net Model: L Channel → Predicted AB Channels
        ↓
Combine: L (original) + AB (predicted)
        ↓
Convert LAB → RGB
        ↓
Colorized Output Image
```


### Data Augmentation Strategies

The project implements **three augmentation strategies**:

#### **Light Strategy** (30% geometric, 20% photometric)

- Horizontal flip, rotation ±15°, random scaling


#### **Medium Strategy** (50% geometric, 40% photometric)

- All light augmentations + brightness/contrast + noise


#### **Heavy Strategy** (70% geometric, 60% photometric)

- All medium augmentations + advanced distortions


- **Baseline Model**
    - U-Net architecture
    - Trained on original grayscale→color pairs
    - Saved under `models/baseline_model/`
- **Augmented Model**
    - Same architecture, trained with augmented data
    - Augmentation strategies configurable in `augmentation_config.yaml`
    - Saved under `models/augmented_model/`


***

## 📊 Project Results \& Performance

Based on the provided outputs, here are the key findings:

### Dataset Statistics

- **Total Images**: 27 (9 train, 9 validation, 9 test)
- **Average Resolution**: 3062×3239 pixels
- **Average File Size**: 2.78MB
- **Format**: JPEG, RGB color mode
- **Quality Issues**: Large images (>4000px) require resizing


### Training Results

#### **Baseline Model (No Augmentation)**

- **Training Time**: 17.9 minutes (11 epochs)
- **Best Validation Loss**: 0.4998
- **PSNR**: 7.16 dB (Poor performance)
- **SSIM**: 0.6881 (Good structural similarity)
- **Issue**: Overfitting detected (val loss > train loss)


#### **Augmented Model (With Augmentation)**

- **Training Time**: 38.6 minutes (11 epochs)
- **Best Validation Loss**: 0.2086 (**58.27% improvement!**)
- **PSNR**: 9.31 dB (+30.08% improvement)
- **SSIM**: 0.5076 (-26.23% decrease)
- **Result**: **Augmentation successfully improved model performance**


### Performance Analysis

| Metric | Baseline | Augmented | Improvement |
| :-- | :-- | :-- | :-- |
| **Validation Loss** | 0.4998 | 0.2086 | **+58.27%** ✅ |
| **PSNR (dB)** | 7.16 | 9.31 | **+30.08%** ✅ |
| **SSIM** | 0.6881 | 0.5076 | **-26.23%** ❌ |
| **Training Time** | 17.9 min | 38.6 min | 2.15× longer |

**Key Insights:**

- ✅ **Data augmentation significantly reduced overfitting**
- ✅ **Validation loss improved dramatically**
- ✅ **PSNR improved, indicating better color accuracy**
- ⚠️ **SSIM decreased, suggesting some structural detail loss**
- 📊 **Trade-off between color accuracy and structural preservation**

***

## 🎮 How to Use the GUI

After training models, test them interactively:

1. **Launch GUI**: Run notebook `07_gui_demo.ipynb`
2. **Select Model**: Choose baseline or augmented from dropdown
3. **Load Image**: Select any color image from your computer
4. **Colorize**: Click to process (converts to grayscale → colorized)
5. **Save Result**: Export the colorized image

### GUI Features

- **Real-time colorization**: See results instantly
- **Model comparison**: Switch between baseline/augmented models
- **Image preview**: Side-by-side grayscale input and color output
- **Export functionality**: Save results in PNG/JPEG format

#### Simple

1. **Load Model**: Select a trained model (baseline or augmented)
2. **Load Image**: Choose a color photo; converted to grayscale for input
3. **Colorize**: Run the model to produce a colorized output
4. **Save Result**: Export the colorized image as PNG or JPEG
5. **Fallback GUI**: If CustomTkinter isn’t available, a basic Tkinter interface is used
***

## 📈 Evaluation Metrics Explained

### **PSNR (Peak Signal-to-Noise Ratio)**

- **Range**: Higher is better
- **Good**: >20 dB, **Acceptable**: 15-20 dB, **Poor**: <15 dB
- **Measures**: Pixel-level color accuracy
- **Project Result**: 7.16 → 9.31 dB (improvement but still low)


### **SSIM (Structural Similarity Index)**

- **Range**: 0-1 (higher is better)
- **Good**: >0.8, **Acceptable**: 0.6-0.8, **Poor**: <0.6
- **Measures**: Structural and perceptual similarity
- **Project Result**: 0.6881 → 0.5076 (decreased)


### **MSE/MAE (Mean Squared/Absolute Error)**

- **Range**: Lower is better
- **Measures**: Pixel-wise prediction error
- **Project Result**: Both improved with augmentation

***

## 🔧 Configuration \& Customization

### Main Configuration (`config/config.yaml`)

```yaml
data:
  input_size: [256, 256]    # Image resolution
  batch_size: 16            # Training batch size
  color_space: "LAB"        # Color space for processing

model:
  architecture: "unet"      # Model architecture
  pretrained: true          # Use pretrained encoder

training:
  epochs: 100               # Maximum training epochs
  learning_rate: 0.001      # Learning rate
  patience: 10              # Early stopping patience
```


### Augmentation Configuration (`config/augmentation_config.yaml`)

```yaml
current_strategy: "medium"  # light, medium, heavy

strategies:
  medium:
    geometric_prob: 0.5     # 50% chance of geometric augmentation
    photometric_prob: 0.4   # 40% chance of color augmentation
    advanced_prob: 0.2      # 20% chance of advanced augmentation
```


***

## 🚨 Troubleshooting

### Common Issues \& Solutions

#### **"No models found" in GUI**

- Ensure you've completed training (notebooks 04 \& 05)
- Check `models/` folder contains `.pth` files


#### **CUDA out of memory**

- Reduce `batch_size` in `config.yaml` (try 8 or 4)
- Use CPU training (slower but works with any hardware)


#### **Poor colorization quality**

- **Small dataset**: Add more training images (aim for 500+)
- **Overfitting**: Increase augmentation intensity
- **Underfitting**: Train for more epochs or use larger model


#### **Import errors**

- Activate virtual environment: `venv\Scripts\activate`
- Install missing packages: `pip install [package_name]`


### Performance Optimization

#### **For Better Results:**

1. **Larger Dataset**: Use 1000+ diverse, high-quality images
2. **Longer Training**: Remove early stopping, train 50-100 epochs
3. **Advanced Augmentation**: Use "heavy" strategy
4. **Model Architecture**: Try ResNet-UNet or deeper networks
5. **Loss Functions**: Experiment with perceptual loss

#### **For Faster Training:**

1. **GPU**: Use CUDA-enabled PyTorch
2. **Mixed Precision**: Enable in config (already on)
3. **Smaller Images**: Use 128×128 input size
4. **Fewer Workers**: Reduce `num_workers` if CPU-limited

***

## 🎓 Learning Outcomes

After completing this project, you'll understand:

1. **Color Space Theory**: RGB vs LAB color spaces and their applications
2. **Data Augmentation**: How diverse training data improves generalization
3. **Deep Learning Pipeline**: End-to-end training, evaluation, and deployment
4. **U-Net Architecture**: Encoder-decoder networks for image translation
5. **Overfitting vs Generalization**: Why augmentation helps model performance
6. **Evaluation Metrics**: PSNR, SSIM, and their interpretations
7. **Model Comparison**: Statistical evaluation of different approaches

***
## Requirements to Interact with Models

- Python 3.8+
- PyTorch (CPU or GPU)
- CustomTkinter (for GUI) or fallback Tkinter
- Jupyter Notebook (for interactive notebooks)
- Adequate disk space for dataset and models (10+ GB)


### Tools \& Libraries

- **PyTorch**: Deep learning framework
- **Albumentations**: Advanced image augmentations
- **OpenCV**: Computer vision operations
- **scikit-image**: Image processing utilities

### Datasets for Expansion

- **COCO Dataset**: Large-scale object detection dataset
- **ImageNet**: Diverse natural images
- **Places365**: Scene-centric images
- **Flickr8K/30K**: User-uploaded photographs

## 🎯 Next Steps \& Improvements

### Immediate Improvements

1. **Expand Dataset**: Collect 1000+ diverse images
2. **Hyperparameter Tuning**: Optimize learning rate, batch size
3. **Advanced Architectures**: Try Pix2Pix, CycleGAN, or Vision Transformers

### Advanced Features

1. **Perceptual Loss**: Use VGG-based loss functions
2. **Attention Mechanisms**: Add self-attention to U-Net
3. **Multi-Scale Training**: Train on multiple resolutions
4. **Web Deployment**: Deploy model using Flask/FastAPI

### Research Directions

1. **Zero-Shot Colorization**: Generalize to unseen image types
2. **Controllable Colorization**: User-guided color placement
3. **Video Colorization**: Extend to temporal consistency
4. **Style Transfer Integration**: Combine colorization with artistic styles

***

**Happy Colorizing! 🎨**

*Transform your grayscale memories into vibrant, colorful images with the power of deep learning and data augmentation.*

***


