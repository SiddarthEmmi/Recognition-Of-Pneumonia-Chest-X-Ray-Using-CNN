# Recognition-Of-Pneumonia-Chest-X-Ray-Using-CNN
This project focuses on classifying chest X-rays as normal or pneumonia-affected using a Convolutional Neural Network (CNN). The dataset consists of 4,538 X-ray images, and the model is designed with multiple convolutional and pooling layers, followed by fully connected layers to classify the images. The model achieves a high accuracy of 97.94% through data preprocessing techniques like resizing and normalization, along with hyperparameter tuning. Performance metrics such as accuracy and loss are visualized using graphs. This project demonstrates the potential of deep learning in medical image classification, and instructions for setting up and running the model are provided for easy replication. Contributions to improve the model are also welcome.

# Overview
This project focuses on building a Convolutional Neural Network (CNN) for chest X-ray (CXR) classification, specifically to identify pneumonia in X-ray images. Using a dataset of 4,538 chest X-rays, the model is trained to distinguish between normal and pneumonia-affected lungs. By optimizing the network with algorithms like Adam and RMSprop, the project achieves high accuracy through hyperparameter tuning and model variation testing.

The key objectives are:

Efficiently processing medical images for disease detection.
Optimizing model performance with advanced tuning techniques.
Demonstrating high accuracy in the detection of pneumonia.

# Dataset
The dataset consists of 4,538 chest X-ray images, divided into two categories:

Normal: Healthy lungs
Pneumonia: Lungs affected by pneumonia
Each category contains images organized into respective folders. The dataset is preprocessed to convert the images to a suitable size and normalized to enhance model performance.

# Model Architecture
The CNN architecture used in this project consists of several layers:

* Convolutional Layers: Extract important features from the input images.
* ReLU Activation: Introduces non-linearity into the model to improve learning.
* Pooling Layers: Reduces the dimensionality of the feature maps, retaining key information.
* Fully Connected Layers: Flatten the output and classify the images into normal or pneumonia classes.
* Softmax Output: Generates probabilities for each class (normal, pneumonia).


Project Associates: Siddartha S Emmi, Shreeraja H M, Shashank N M, Shaik Mohaddis
