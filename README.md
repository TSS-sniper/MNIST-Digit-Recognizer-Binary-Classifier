# MNIST Digit Recognizer Binary Classifier #
## Introduction ##
* MNIST ( Modified National Institute of Standards and Technology ) Digit Recognizer is a popular dataset in the field of machine learning and computer vision. 
It is a collection of 70,000 grayscale images of handwritten digits, each of which is 28x28 pixels in size. The dataset is divided into two parts: 60,000 images for training and 10,000 images for testing. 
* The goal is to develop a machine learning model that can accurately classify each image into the corresponding digit, ranging from 0 to 9. The MNIST dataset has been widely used as a benchmark for evaluating the performance of various machine learning algorithms and deep learning models, and it has played an important role in advancing the field of computer vision. 
* The popularity of this dataset is due to its simplicity and the fact that it provides a challenging yet manageable problem that can be used to teach and learn the basics of machine learning and image processing.

## What is Classification? ##
* Classification is one of the most important and commonly used techniques in data science. It refers to the process of categorizing or labelling data into predefined classes or categories based on certain features or attributes. 
Classification aims to develop a predictive model that can accurately classify new, unseen data points into one of the pre-defined classes. 
This technique is widely used in various fields such as finance, marketing, healthcare, and natural language processing.
* Binary classification is a type of supervised learning problem in which the goal is to predict one of two possible outcomes or classes for each input observation. The output variable in binary classification is typically binary, meaning it takes on one of two values such as 0 or 1, true or false, positive or negative, etc.

## Objective: ##
To create a Binary Classification model for a particular value in MNIST Digit Recognizer Dataset using three different classification algorithms.
The main objective of the MNIST Digit Recognizer Dataset is to develop algorithms that can correctly classify the images into their respective digit categories.

Classification Algorithms Used in the Project:

1. Logistic Regression
2. SVM (Support Vector Machine)
3. Decision Tree

## Dataset Description ##
* The Digit Dataset/Digit Recognizer Dataset contains files train.csv and test.csv containing grey-scale images of hand-drawn digits, from zero through nine.
* The MNIST (Modified National Institute of Standards and Technology) dataset is a widely used benchmark dataset in the field of machine learning and computer vision. It consists of a set of 70,000 images of handwritten digits (0-9) that have been normalized, centred and scaled to fit into a 28x28 pixel grid.
* The dataset is divided into two parts: 60,000 images are used for training i.e. train.csv and 10,000 images are used for testing i.e. test.csv.
* The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel values of the associated image.
* The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.
* Each pixel has a single pixel value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel value is an integer between 0 and 255, inclusive.

# Methodology Used: #
1. Load the MNIST dataset: The MNIST dataset can be loaded using the scikit-learn library.
2. Preprocess the data: The dataset should be preprocessed by flattening the 28x28 pixel images to a 1D array of length 784 and normalizing the pixel values to a range of 0 to 1. The data should also be split into training and testing sets.
3. Train the model: Train the Logistic Regression, SVM, and Decision Trees model on the training dataset.
4. Evaluate the model: Evaluate the model on the testing dataset using metrics such as accuracy and confusion matrix.
5. Make predictions: Use the trained model to make predictions on new, unseen data.
6. Compare the models: Compare the performance of the Logistic Regression, SVM, and Decision Trees models on the testing dataset. Choose the model with the highest accuracy.
