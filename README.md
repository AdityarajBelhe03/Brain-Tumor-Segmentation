
# Brain Tumor Segmentation using Advanced Deep Learning Algorithms.

## Project Overview

The U-Net model used in this project is highly effective for biomedical image segmentation. We have achieved a 99.30% testing accuracy, demonstrating the model's robustness and precision. 

## Table of Contents

- [Directories](#directories)
  - [.ipynb_checkpoints](#ipynb_checkpoints)
  - [__pycache__](#pycache)
  - [files](#files)
  - [logs](#logs)
  - [results](#results)
  - [splited](#splited)
  - [test](#test)
  - [test_data/test](#test_datatest)
  - [train](#train)
- [Notebooks](#notebooks)
  - [data.ipynb](#dataipynb)
  - [dataset.ipynb](#datasetipynb)
  - [eval.ipynb](#evalipynb)
  - [metrics.ipynb](#metricshipynb)
  - [model2.ipynb](#model2ipynb)
  - [predict.ipynb](#predictipynb)
  - [train.ipynb](#trainipynb)
- [Scripts](#scripts)
  - [model.py](#modelpy)
  - [predict.py](#predictpy)
  - [train.py](#trainpy)
## Dependencies

- **Python 3**: The programming language used.
- **TensorFlow >= 2.0**: The primary deep learning library.
- **Keras**: High-level neural networks API, now part of TensorFlow.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting and visualization.
- **scikit-learn**: For additional machine learning tools.
- **Pillow**: For image processing.
- **OpenCV**: For image processing and computer vision tasks.
## Data Preparation

Data augmentation and train-test split are performed using the data_preparation.py script.

```bash
 python scripts/data_preparation.py
 ```
## Training the Model

The U-Net model is trained using the train_model.py script. The model architecture is defined to capture complex features and spatial information effectively.

```bash
python scripts/train_model.py
```
## Evaluating the Model

Model evaluation is performed using the evaluate_model.py script. This script assesses the model's performance on the test set.

```bash
python scripts/evaluate_model.py
```
## Making Predictions

To make predictions on new MRI images, use the predict.py script. This script will output the segmented tumor regions.

```bash
python scripts/predict.py --image_path path_to_image
```
## Model Architecture

The U-Net architecture is composed of an encoder (downsampling) path and a decoder (upsampling) path. Skip connections between the encoder and decoder help in retaining spatial information lost during downsampling.

### Detailed Model Explanation:

#### Encoder: Captures context using convolutional layers followed by max-pooling layers.
#### Bottleneck: Connects the encoder and decoder, capturing the most abstract features.
#### Decoder: Reconstructs the spatial dimensions using transposed convolutions and skip connections.
#### Final Layer: Outputs the segmented image using a 1x1 convolution with a sigmoid activation function.

### Model Summary

Below is a summary of the U-Net model architecture:

| Layer Type          | Output Shape       | Parameters      |
|---------------------|--------------------|-----------------|
| Input Layer         | (512, 512, 3)      | 0               |
| Conv2D + ReLU       | (512, 512, 32)     | 896             |
| Conv2D + ReLU       | (512, 512, 32)     | 9248            |
| MaxPooling2D        | (256, 256, 32)     | 0               |
| Conv2D + ReLU       | (256, 256, 64)     | 18496           |
| Conv2D + ReLU       | (256, 256, 64)     | 36928           |
| MaxPooling2D        | (128, 128, 64)     | 0               |
| Conv2D + ReLU       | (128, 128, 128)    | 73856           |
| Conv2D + ReLU       | (128, 128, 128)    | 147584          |
| MaxPooling2D        | (64, 64, 128)      | 0               |
| Conv2D + ReLU       | (64, 64, 256)      | 295168          |
| Conv2D + ReLU       | (64, 64, 256)      | 590080          |
| MaxPooling2D        | (32, 32, 256)      | 0               |
| Conv2D + ReLU       | (32, 32, 512)      | 1180160         |
| Conv2D + ReLU       | (32, 32, 512)      | 2359808         |
| MaxPooling2D        | (16, 16, 512)      | 0               |
| Conv2D + ReLU       | (16, 16, 1024)     | 4719616         |
| Conv2D + ReLU       | (16, 16, 1024)     | 9438208         |
| Conv2DTranspose     | (32, 32, 512)      | 2097664         |
| Conv2D + ReLU       | (32, 32, 512)      | 4719104         |
| Conv2D + ReLU       | (32, 32, 512)      | 2359808         |
| Conv2DTranspose     | (64, 64, 256)      | 524544          |
| Conv2D + ReLU       | (64, 64, 256)      | 1180160         |
| Conv2D + ReLU       | (64, 64, 256)      | 590080          |
| Conv2DTranspose     | (128, 128, 128)    | 131200          |
| Conv2D + ReLU       | (128, 128, 128)    | 295040          |
| Conv2D + ReLU       | (128, 128, 128)    | 147584          |
| Conv2DTranspose     | (256, 256, 64)     | 32832           |
| Conv2D + ReLU       | (256, 256, 64)     | 73792           |
| Conv2D + ReLU       | (256, 256, 64)     | 36928           |
| Conv2DTranspose     | (512, 512, 32)     | 8224            |
| Conv2D + ReLU       | (512, 512, 32)     | 18464           |
| Conv2D + ReLU       | (512, 512, 32)     | 9248            |
| Conv2D (Output)     | (512, 512, 1)      | 33              |

## Results

The U-Net model achieved impressive performance metrics on the test dataset. Below are the key results:

| Metric      | Value  |
|-------------|--------|
| Accuracy    | 99.30% |
| F1 Score    | 0.71   |
| Jaccard Index | 0.63   |
| Recall      | 0.72   |
| Precision   | 0.85   |


## References

1. [BRATS2017 1st Solution, Ensembles of Multiple Models and Architectures for Robust Brain Tumor Segmentation](https://arxiv.org/pdf/1711.01468.pdf)
2. [BRATS2017 2nd Solution, Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks](https://arxiv.org/abs/1710.08047)
3. [BRATS2017 3rd Solution, 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation](https://arxiv.org/abs/1606.06650)
4. [Ischemic Stroke Lesion Segmentation from MRI using UNet with Transfer Learning and Morphological Post-Processing](https://arxiv.org/abs/1911.01314)
5. [Deep Learning for Brain MRI Segmentation: State of the Art and Future Directions](https://arxiv.org/abs/2007.09075)
## Authors

- [@Prathamesh Uravane](https://github.com/upratham)
- [@Vedant Ganthade ](https://github.com/vedantganthade)
- [@Abhiraj Gadade](https://github.com/AbhirajGadade)



