# Chest X-Ray Classification with Explainable AI (XAI)

A deep learning project that classifies chest X-ray images as either **NORMAL** or **PNEUMONIA** using transfer learning with three state-of-the-art models and Grad-CAM visualization for explainability.

## 📋 Project Overview

This project implements medical image classification using:

- **Inception ResNetV2**
- **VGG16**
- **Xception**

Each model is trained with transfer learning on ImageNet weights, fine-tuned on the chest X-ray dataset, and evaluated using confusion matrices and ROC curves. An ensemble approach combines predictions from all three models for improved accuracy.

### Key Features

- ✅ Transfer learning with pre-trained ImageNet weights
- ✅ Grad-CAM implementation for model interpretability
- ✅ Class imbalance handling with weighted loss
- ✅ Ensemble learning for improved predictions
- ✅ ROC curves and confusion matrices for model evaluation

## 📁 Dataset Structure

```
Dataset/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── test/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── val/
        ├── NORMAL/
        └── PNEUMONIA/
```

The dataset contains chest X-ray images categorized into two classes:

- **NORMAL**: Healthy chest X-rays
- **PNEUMONIA**: Chest X-rays showing pneumonia

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow/Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/XAI_code.git
cd XAI_code
```

2. Create a virtual environment:

```bash
python -m venv ml_env
source ml_env/Scripts/activate  # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Prepare the dataset:
   - Place the chest X-ray dataset in the appropriate location:
     - **Google Colab**: Upload to `Google Drive/Dataset/chest_xray/`
     - **Local**: Place in `./Dataset/chest_xray/`
   - Ensure it follows the structure shown above

### Running on Google Colab (Recommended)

1. Upload the notebook (`XAI.ipynb`) to Google Colab
2. Upload the `chest_xray` dataset to your Google Drive under `Dataset/chest_xray/`
3. In Colab: **Runtime > Change runtime type > GPU (T4)**
4. Run all cells sequentially — the notebook will mount Google Drive and load the dataset automatically

### Running Locally

```bash
jupyter notebook XAI.ipynb
```

Update `DATASET_BASE` in the notebook to point to your local dataset path.

## 📊 Model Architecture

### Base Configuration

- **Input Size**: 299 × 299 pixels (RGB)
- **Image Preprocessing**:
  - Rescaling: 1/255
  - Data augmentation: zoom, shift, brightness adjustments
  - Target size normalization

### Training Configuration

- **Batch Size**: 32
- **Training Epochs**: 3 (initial)
- **Fine-tuning Epochs**: 5
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Early Stopping**: Patience=10, min_delta=0.001

### Class Weights

Balanced class weights are computed to handle dataset imbalance between NORMAL and PNEUMONIA classes.

## 🔍 Explainability: Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) generates visual explanations of model predictions:

- Highlights the regions of the image that most influenced the classification
- Implemented for each model with their respective last convolutional layers:
  - **Inception**: `conv_7b_ac`
  - **VGG16**: `block5_conv3`
  - **Xception**: `block14_sepconv2_act`

## 📈 Performance Metrics

### Model Results

| Model | Test Accuracy |
|---|---|
| InceptionResNetV2 | 87.66% |
| VGG16 | 92.47% |
| Xception | 87.34% |

### Evaluation Outputs

1. **Accuracy & Loss**: Training and validation curves
2. **Confusion Matrix**: Per-model and ensemble predictions
3. **ROC Curves**: AUC-ROC for binary classification
4. **Sensitivity & Specificity**: Clinical relevance metrics

## 🔗 Ensemble Learning

The ensemble combines predictions from all three models by averaging their output probabilities:

```python
ensemble_preds = (inception_preds + vgg16_preds + xception_preds) / 3.0
```

This typically provides more robust and reliable predictions than individual models.

## 📝 Notebook Structure

1. **Setup & Configuration**: Import libraries and set hyperparameters
2. **Data Loading**: Load and preprocess chest X-ray images
3. **Data Visualization**: Display sample images and class distribution
4. **Model Definition**: Create custom models on top of pre-trained bases
5. **Training**: Train each model with initial weights and fine-tune
6. **Evaluation**: Compute metrics and generate visualizations
7. **Explainability**: Generate and visualize Grad-CAM heatmaps
8. **Ensemble**: Combine predictions and evaluate ensemble performance

## 🛠️ Customization

### Modify Training Parameters

Edit the **Overall Settings** cell:

```python
FINE_TUNING_EPOCHS = 5
TRAINING_EPOCHS = 3
BATCH_SIZE = 32
image_height = 299
image_width = 299
```

### Change Data Augmentation

Modify the `ImageDataGenerator` parameters:

```python
gen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05]
)
```

### Add Custom Models

Extend the notebook by creating additional base models before the "Model" section.

## 📚 Dependencies

See [requirements.txt](requirements.txt) for the full list. Install with:

```bash
pip install -r requirements.txt
```

> **Note**: When running on Google Colab, all dependencies are pre-installed.

## ⚠️ Notes for Google Colab Users

- The notebook is configured to run on **Google Colab with GPU (T4)** out of the box
- Google Drive is mounted automatically in the setup cell
- Dataset path is set to `/content/drive/MyDrive/Dataset/chest_xray`
- Ensure your dataset is placed at that path in Google Drive
- Use **Runtime > Change runtime type > GPU** for faster training

## 📖 References

- Grad-CAM: [Why Should I Trust You?](https://arxiv.org/abs/1610.02055)
- Inception ResNetV2: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)
- VGG16: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- Xception: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## 👤 Author

Created for educational and research purposes in medical image analysis and explainable AI.

---

**Disclaimer**: This project is for educational purposes only and should not be used as a substitute for professional medical diagnosis.
