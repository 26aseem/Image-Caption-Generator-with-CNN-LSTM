# Image-Caption-Generator-with-CNN-LSTM


## Introduction
You saw an image and your brain readily deduced what it represented, but can a computer deduce what it represents? We can now develop models that can generate captions for images thanks to advances in deep learning techniques, the availability of large datasets, and computer power. This is what we'll do in this Python-based project, where we'll combine Convolutional Neural Networks and a sort of Recurrent Neural Network (LSTM) deep learning approaches.

Image caption generator is a task that involves computer vision and natural language processing concepts to recognize the context of an image and describe them in a natural language like English.


## Objective
The objective of our project is to learn the concepts of a CNN and LSTM model and build a working model of Image caption generator by implementing CNN with LSTM.

In this Python project, we will be implementing the caption generator using CNN (Convolutional Neural Networks) and LSTM (Long short term memory). The image features will be extracted from Xception which is a CNN model trained on the imagenet dataset and then we feed the features into the LSTM model which will be responsible for generating the image captions.

## Dataset Used
For the image caption generator, we will be using the Flickr_8K dataset. The Flickr_8k_text folder contains file Flickr8k.token which is the main file of our dataset that contains image name and their respective captions
* Link for the image dataset: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
* Link for the image caption labels: https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

## Methodology & Model used
- Convolutional neural networks are a type of deep neural network that can process data in the form of a 2D matrix. Images are easily represented as a 2D matrix, and CNN is an   excellent tool for working with them. CNN is mostly used to classify images and determine whether they depict a bird, a jet, or Superman, among other things.
- Long short term memory (LSTM) is a form of RNN (recurrent neural network) that is particularly well adapted to sequence prediction challenges.
- So, to make our image caption generator model, we will be merging these architectures. It is also called a CNN-RNN model.
* CNN is used for extracting features from the image. We will use the pre-trained model Xception.
* LSTM will use the information from CNN to help generate a description of the image.
