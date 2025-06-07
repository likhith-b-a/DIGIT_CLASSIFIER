# MNIST Digit Classifier

A PyTorch implementation of a neural network for classifying handwritten digits from the MNIST dataset.

## Overview

This project implements a fully connected neural network to classify handwritten digits (0-9) from the famous MNIST dataset. The model achieves approximately **97.8% accuracy** on the validation set after just 5 epochs of training.

## Features

- **Simple Architecture**: 3-layer fully connected neural network
- **Fast Training**: Reaches high accuracy in just 5 epochs
- **GPU Support**: Automatically detects and uses CUDA if available
- **Data Visualization**: Includes utilities for visualizing MNIST images
- **Clean Implementation**: Well-structured PyTorch code with clear separation of concerns

## Model Architecture

The neural network consists of:

```
Input Layer:    784 neurons (28×28 flattened images)
Hidden Layer 1: 512 neurons + ReLU activation
Hidden Layer 2: 512 neurons + ReLU activation
Output Layer:   10 neurons (one for each digit class)
```

## Requirements

```python
torch
torchvision
matplotlib
```

## Installation

1. Clone this repository or download the notebook
2. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib
   ```
3. Run the Jupyter notebook `MNIST_DIGIT_CLASSIFIER.ipynb`

## Usage

### Training the Model

The notebook automatically:
1. Downloads the MNIST dataset
2. Preprocesses the data (converts to tensors, normalizes)
3. Creates data loaders with batch size of 32
4. Initializes the neural network
5. Trains for 5 epochs using Adam optimizer
6. Validates the model after each epoch

### Key Components

**Data Loading:**
```python
train_set = torchvision.datasets.MNIST("./data/", train=True, download=True)
valid_set = torchvision.datasets.MNIST("./data/", train=False, download=True)
```

**Model Definition:**
```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
```

**Training Loop:**
- Uses CrossEntropyLoss for multi-class classification
- Adam optimizer with default learning rate
- Batch-wise training with gradient updates

## Results

After 5 epochs of training:

| Epoch | Train Loss | Train Accuracy | Valid Loss | Valid Accuracy |
|-------|------------|----------------|------------|----------------|
| 0     | 372.04     | 94.01%         | 38.67      | 96.20%         |
| 1     | 154.73     | 97.45%         | 22.20      | 97.71%         |
| 2     | 105.71     | 98.27%         | 22.65      | 97.84%         |
| 3     | 86.17      | 98.53%         | 28.04      | 97.41%         |
| 4     | 64.12      | 98.91%         | 24.92      | 97.86%         |

## Project Structure

```
MNIST_DIGIT_CLASSIFIER/
├── MNIST_DIGIT_CLASSIFIER.ipynb    # Main notebook with complete implementation
├── data/                           # MNIST dataset (auto-downloaded)
└── README.md                       # This file
```

## Key Functions

- `get_batch_accuracy()`: Calculates accuracy for a batch of predictions
- `train()`: Training loop for one epoch
- `validate()`: Validation loop for one epoch

## Dataset Information

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels (grayscale)
- **Classes**: 10 (digits 0-9)
- **Data Type**: Normalized float32 tensors (0.0 to 1.0)


## Acknowledgments

- MNIST dataset: Yann LeCun, Corinna Cortes, and Christopher J.C. Burges
- PyTorch framework for deep learning implementation
- This project uses skills I learned from the NVIDIA Deep Learning Institute course.
