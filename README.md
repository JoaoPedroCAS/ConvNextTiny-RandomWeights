# ConvNeXt-Tiny Feature Extraction with Random Weights

This project explores the performance of feature extraction using the ConvNeXt-Tiny architecture with random weight initialization. The primary goal is to investigate how different random initializations impact the feature extraction and classification accuracy, particularly for texture classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)

## Project Overview

The ConvNeXt-Tiny model is used to extract features from a dataset of texture images. In this project, the model is modified to remove the classification layers, and the weights of its convolutional and linear layers are randomly initialized within a specified range. The extracted features are then fed into a Linear Discriminant Analysis (LDA) classifier to evaluate the impact of different random initializations on classification performance.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/convnext-tiny-random-weights.git
    cd convnext-tiny-random-weights
    ```

2. **Required Libraries:**
   - `torch`
   - `torchvision`
   - `numpy`
   - `scikit-learn`
   - `PIL`
   - `psutil`

   ```bash
   pip install torch torchvision numpy scikit-learn pillow psutil
   ```

## Dataset

The project assumes a directory structure for the dataset where each subfolder represents a class label. Update the `DATASET_PATH` variable in the code to point to your local dataset directory.

**Example structure:**

```
texturas/
├── class_1/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
├── class_2/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ...
└── ...
```

## Usage

1. **Configure the Dataset Path:**

   Update the `DATASET_PATH` and `RESULTS_PATH` variables in the code with the path to your dataset and results directories.

2. **Run the Script:**

   Execute the main script to start the feature extraction and classification process:

   ```bash
   python main.py
   ```

3. **Parameters:**

   - `BATCH_SIZE`: Number of images processed in each batch during feature extraction.
   - `N_COMPONENTS`: Number of principal components to retain in PCA.
   - `a_std_range`: Range for uniform random initialization of weights.

## Results

The project evaluates the impact of random weight initialization by:

1. Extracting features using a modified ConvNeXt-Tiny model.
2. Applying PCA to reduce dimensionality.
3. Classifying the features using LDA.
4. Saving metrics such as accuracy, F1-Score, recall, and precision for each random initialization.

Results are saved in the `RESULTS_PATH` directory as text files.