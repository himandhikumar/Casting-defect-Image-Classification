# Casting Product Image Classification

## Overview
This project focuses on classifying casting product images to distinguish between defective and non-defective parts. The goal is to automate quality inspection in industrial manufacturing using deep learning techniques. The notebook includes data preprocessing, model training, and evaluation steps using PyTorch.

## Dataset
The dataset used is the **"Real-Life Industrial Dataset of Casting Product"** from Kaggle:
- [Dataset Link](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
- Contains images of casting products labeled as `def_front` (defective) and `ok_front` (non-defective).
- Image size: 512x512 pixels.

## Notebook Structure
1. **Data Preparation**
   - Download and extract the dataset from Kaggle.
   - Organize data into training and testing directories.
   - Visualize sample images.

2. **Data Preprocessing**
   - Apply transformations (resizing, normalization, etc.) using `torchvision.transforms`.
   - Create `DataLoader` for batch processing.

3. **Model Architecture**
   - Uses a convolutional neural network (CNN) built with PyTorch.
   - Pretrained models (e.g., ResNet) can also be explored for transfer learning.

4. **Training**
   - Define loss function (cross-entropy) and optimizer (Adam).
   - Train the model on the dataset and validate performance.

5. **Evaluation**
   - Metrics: Accuracy, precision, recall, and F1-score.
   - Visualize training/validation loss and accuracy curves.

6. **Results**
   - Achieved high accuracy in classifying defective vs. non-defective castings.
   - Model can be deployed for real-time quality inspection.

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- pandas
- matplotlib
- kagglehub (for dataset download)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/casting-product-classification.git
   cd casting-product-classification
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision pandas matplotlib kagglehub
   ```

3. Download the dataset:
   - Ensure you have Kaggle API credentials (`kaggle.json`).
   - Run the dataset download cell in the notebook or use:
     ```bash
     kaggle datasets download -d ravirajsinh45/real-life-industrial-dataset-of-casting-product
     unzip real-life-industrial-dataset-of-casting-product.zip
     ```

## Usage
1. Open the Colab notebook:
   ```bash
   Colab notebook Casting_product_image_classification.ipynb
   ```

2. Execute cells sequentially to:
   - Preprocess data.
   - Train the model.
   - Evaluate performance.

## Results
Example output:
- Validation accuracy: **98.85%%**


## Future Improvements
- Experiment with more complex architectures (e.g., EfficientNet).
- Implement data augmentation for better generalization.
- Deploy as a web app using Flask/FastAPI.

## License
This project is open-source under the [MIT License](LICENSE).

---

### Key Features to Highlight:
- **Industrial Application**: Solves a real-world problem in manufacturing quality control.
- **Modular Code**: Easy to extend for similar image classification tasks.
- **Visualizations**: Includes sample images and performance metrics.
