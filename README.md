LeafDiseaseClassifier
Overview
This repository contains a comprehensive Python script for classifying leaf images using deep learning, leveraging the MobileNetV2 architecture. The project focuses on building a robust and accurate classification model through techniques like transfer learning, extensive data augmentation, hyperparameter tuning with KerasTuner, and K-Fold cross-validation. The goal is to provide a reliable solution for identifying different leaf types or diseases based on visual characteristics.

Features
Transfer Learning: Utilizes the pre-trained MobileNetV2 model as a feature extractor.

Custom Classification Head: Implements a custom dense layer network with regularization, batch normalization, and various activation functions (ReLU, LeakyReLU, ELU) for optimal performance.

Extensive Data Augmentation: Employs ImageDataGenerator with various transformations (rotation, shifts, shear, zoom, flip, brightness) to enhance model generalization and prevent overfitting.

Hyperparameter Tuning: Integrates KerasTuner (Hyperband) to efficiently search for optimal learning rates, improving model training stability and performance.

K-Fold Cross-Validation: Implements K-Fold cross-validation to provide a more robust and unbiased estimate of the model's performance, reducing reliance on a single train-validation split.

Ensemble Prediction: Combines predictions from multiple K-Fold models to potentially improve overall accuracy and robustness.

Comprehensive Evaluation: Generates detailed classification reports, confusion matrices, F1-scores, Precision-Recall curves, and ROC curves.

Visualizations: Includes functions to visualize sample images, class distributions, augmented samples, training history (accuracy and loss), and hyperparameter tuning results.

Best Model Saving: Automatically saves the best performing model (either the initial MobileNetV2 or the best K-Fold model) based on validation accuracy.

Dataset Structure
The script expects the dataset to be organized into a main leafs directory, with subdirectories for Train, Validation, and Test sets. Each of these sets should contain subdirectories named after the respective classes.

leafs/
├── Train/
│   ├── ClassA/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── ClassB/
│   │   ├── imageX.jpg
│   │   └── ...
│   └── ...
├── Validation/
│   ├── ClassA/
│   │   ├── image_val1.jpg
│   │   └── ...
│   └── ...
└── Test/
    ├── ClassA/
    │   ├── image_test1.jpg
    │   └── ...
    └── ...

Requirements
To run this code, you need to have the following Python libraries installed. You can install them using pip:

pip install tensorflow scikit-learn pandas numpy matplotlib seaborn pillow keras-tuner

It's recommended to use a virtual environment to manage dependencies.

Usage
Prepare your dataset: Ensure your leaf image dataset is structured as described in the "Dataset Structure" section within a directory named leafs in the same location as the script.

Run the script:

python leafs.py

The script will perform the following steps:

Load and preprocess the data.

Visualize data samples, class distributions, and augmented images.

Train an initial MobileNetV2 model.

Perform K-Fold cross-validation with hyperparameter tuning for each fold.

Generate various plots for training history, ROC curves, and hyperparameter results.

Evaluate the ensemble model on the test set.

Save the best overall model as best_overall_leaf_model.h5.

Model Architecture
The model utilizes MobileNetV2 as a base, with its initial layers frozen to leverage pre-trained weights. A custom classification head is appended, consisting of:

GlobalMaxPooling2D

Multiple Dense layers with varying units (512, 256, 128, 64, 32, 16)

BatchNormalization layers for stable training

Dropout layers for regularization

LeakyReLU and ELU activation functions in intermediate layers

ReLU activation in some intermediate layers

Softmax activation in the final output layer for multi-class classification.

L2 regularization applied to dense layers to prevent overfitting.

Hyperparameter Tuning
KerasTuner's Hyperband algorithm is used to tune the learning rate for the Adam optimizer within each K-Fold. The learning rate is set with a CosineDecay schedule.

K-Fold Cross-Validation
The script performs 5-fold stratified cross-validation. This means the dataset is split into 5 subsets, and the model is trained 5 times. In each iteration, a different subset is used for validation, and the remaining 4 are used for training. This helps in getting a more reliable performance metric and ensures the model generalizes well across different data partitions.

Results and Evaluation
After training, the script will output:

Test accuracy and loss for the initial MobileNetV2 model.

Validation accuracy and loss for each K-Fold.

Average K-Fold validation accuracy and loss.

A detailed classification report, confusion matrix, and weighted F1-score for the ensemble model on the test set.

Visualizations
The script automatically generates and displays several plots:

Sample images from training and test sets.

Class distribution bar charts for training, validation, and test sets.

Examples of augmented images.

Training and validation accuracy/loss plots for the initial model and each K-Fold model.

ROC curves for the initial model and each K-Fold model (on their respective validation sets) and for the ensemble model on the test set.

Precision-Recall curves for the ensemble model on the test set.

Hyperparameter tuning results (Validation Accuracy vs. Learning Rate) for each fold.

Comparative bar charts for initial vs. K-Fold accuracies and losses.

Contributing
Feel free to fork this repository, open issues, or submit pull requests. Contributions are welcome!
