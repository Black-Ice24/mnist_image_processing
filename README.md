# README for MNIST Image Processing Notebook

This Jupyter Notebook file contains a Python code to process the MNIST dataset of handwritten digits. The code uses the RandomForestClassifier from the scikit-learn library to build a classification model.

## Purpose of the Notebook
The purpose of this notebook is to demonstrate how to load and preprocess the MNIST dataset and train a random forest model to classify handwritten digits.

## Dataset
The MNIST database is a widely used benchmark dataset for image processing tasks. It contains 60,000 training images and 10,000 test images of handwritten digits. The images are size-normalized and centred in a fixed-size image. The dataset is available for download, or you can access it using a library like Keras.

## Steps
The notebook follows the following steps:

1. Load the MNIST dataset.
2. Split the training data into a training and development(test) set.
3. Use the RandomForestClassifier to create a classification model.
4. Pick one parameter to tune, and explain why you chose this parameter.
5. Choose a value for the parameter to test on the test data and explain why.
6. Print the confusion matrix for the Random Forest model on the test set.
7. Report which classes the model struggles with the most.
8. Report the accuracy, precision, recall, and f1-score.

## How to Use the Notebook
To use this notebook, you will need a Python environment with the necessary libraries installed, such as scikit-learn, Keras, and Matplotlib. You can install these libraries using pip or conda.

After installing the required libraries, open the notebook in Jupyter Notebook or JupyterLab and run each cell in order. The notebook will load the MNIST dataset, split it into training and development sets, train a Random Forest model, and evaluate its performance on the test set.

## Conclusion
This notebook demonstrates how to load and preprocess the MNIST dataset and train a Random Forest model to classify handwritten digits. It also shows how to evaluate the model's performance using metrics like accuracy, precision, recall, and f1-score, and the confusion matrix. By following the steps in this notebook, you can learn how to preprocess image data, train a machine learning model, and evaluate its performance.
