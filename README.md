# Image-Caption-Generator-with-CNN-LSTM

## Project Link
https://image-caption-generator-aseem.herokuapp.com


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
- CNN is used for extracting features from the image. We will use the pre-trained model Xception.
- LSTM will use the information from CNN to help generate a description of the image.
- Caption text is converted to speech using the gTTS (Google Text-to-Speech) library of Python.
- The generated caption is read aloud automatically for the user.

<img width="462" alt="Screenshot 2021-11-21 211848" src="https://user-images.githubusercontent.com/43955843/142769551-d57c5e5f-355e-40cd-b2fe-06821299cf96.png">

   Figure 1: Methodology and Model architecture


## Create Data generator
To make this task into a supervised learning task, we have to provide input and output to the model for training. We have to train our model on 6000 images and each image will contain 2048 length feature vector and caption is also represented as numbers. This amount of data for 6000 images is not possible to hold into memory so we will be using a generator method that will yield batches. The generator will yield the input and output sequence.
The input to our model is [x1, x2] and the output will be y, where x1 is the 2048 feature vector of that image, x2 is the input text sequence and y is the output text sequence that the model has to predict.

## Defining the CNN-LSTM model
To define the structure of the model, we will be using the Keras Model from Functional API. It will consist of three major parts:
- Feature Extractor – The feature extracted from the image has a size of 2048, with a dense layer, we will reduce the dimensions to 256 nodes.
- Sequence Processor – An embedding layer will handle the textual input, followed by the LSTM layer.
- Decoder – By merging the output from the above two layers, we will process by the dense layer to make the final prediction. The final layer will contain the number of nodes equal to our vocabulary size.


![model](https://user-images.githubusercontent.com/43955843/142769502-082e87f0-b979-467f-9ffa-9d8d9ed05bb4.png)

             Figure 2: CNN-LSTM based architecture used in the project



## Training the model
To train the model, we will be using the 6000 training images by generating the input and output sequences in batches and fitting them to the model using model.fit_generator() method.

## Testing the model
The model has been trained, now, we will make a separate file testing_caption_generator.py which will load the model and generate predictions. The predictions contain the max length of index values so we will use the same tokenizer.p pickle file to get the words from their index values.

## Output
An interactive python based GUI has been developed using streamlit package and deployed on Heroku platform.
The screenshots of the website are displayed below.

<img width="960" alt="image" src="https://user-images.githubusercontent.com/43955843/142769289-9c4520f7-9679-4f75-9ae9-a3cafdc16cf6.png">

                               Figure 3: Interactive GUI using streamlit package




<img width="628" alt="image" src="https://user-images.githubusercontent.com/43955843/142769327-ca309dd0-e7c3-4677-b855-96540e57ff88.png">

                   Figure 4: Sample image uploaded for caption generation




<img width="574" alt="image" src="https://user-images.githubusercontent.com/43955843/142769345-6475e726-58bf-4df2-8052-782ada66ee93.png">

                Figure 5: Caption generated for the sample image
