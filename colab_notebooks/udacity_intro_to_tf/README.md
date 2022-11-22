# Udacity Intro to TensorFlow for Deep Learning Course
[Course link](https://classroom.udacity.com/courses/ud187)


## Lesson 1. Welcome to the Course
Self-explanatory.


## Lesson 2. Introduction to Machine Learning
This lesson introduced the basis of machine learning.
- Main cycle of typical machine learning project
    - Get data: Split into training, testing data
    - Preprocess data: Clean the dataset or anyother preparation necessary for the model
    - Train model: Define and train model on the training data
    - Analyse trained model: Analyse trained model and optimize as needed.
    - Evaluate model: Evaluate model on the test dataset
    - Deploy model: Deploy the trained model into application
- How to construct and train a model on tensorflow?
    - Define the layers for the model.
    - define loss, optimizer and metrics for the model.
- Model is trained on training data, it's performace is measured on a validation data and the final model is evaluated on a test data.
- Model trained to solve a _regression problem_ of converting degree celsiys to degree fahrenheit.


## Lesson 3. Fashion MNIST
- Implemented a neural network to solve a classification problem, classifying articles of cloths.
- Introduced activation functions to solve more complex problems.
- Introduced to Fashion MNIST dataset


## Lesson 4. Introduction to CNNS
- Introduced to convolutional and maxpooling layer
- Built and trained a CNN model to classify cloths from fashion MNIST data
- Applied convolutional operation on a greyscale image


## Lesson 5. Going Further with CNNs
- Built and trained a CNN model to classify images as to containing a dog or cat
- Applied convolutional operation on colored images
- Introduced to the problem of overfitting
  - Occurs when the model fails to generalise well to data outside of the training data
- Introduced to different methods to prevent overfitting in trained model
    - Data Augmentation: Add variation to the images in the training data
    - Model analysis: Validation set used to track performance of model after each training epoch. Model makes prediction on the validation set and it is never used to train the model. (Early Stopping)
    - Dropout and Dropconnection: Randomly disabling neurons and connection in the model during each epoch
    - Regularization: Constraining the model weights and bias
    - Simplify the model
    - Get more data

*Side Notes*.
- How effective are the image augmentations that were applied to the training dataset? A closer look at the augmentation, show the side-effects of the augmentation especially around the boundaries of the image, so how do convolutional neural networks get around this?
- The only model analysis i have currently is
    - plot of model metrics vs epoch to determine early stopping
- I know that there are other metrics or ways of analysing a model but it would be good to refresh on them and possibly document them here


## Lesson 6 & 7. Transfer Learning & Save and Loading Models
Summary

**Transfer learning**   
This is an approach that applies a pre-trained model to solve a different task. An example would be a model trained on the ImageNet dataset could be used as a starting point for image classification. The general idea is that *there would be an improvment of learning in a new task through the transfer of knowledge from a related task that has already been learned*, so the learnt task needs to be general but also related to the new task.

\
Benefits
- Re-uses expert model trained on large datasets with much larger computational resources. So a much better starting point
- Pre-trained model is essentially used as a feature extraction layer in our new model.
- Less data and tme would be needed during re-training of pre-trained model.
- Much better learning curve, so the model's improvement during trainings is much larger and better.

\
Approach to transfer learning
- Develop model approach. Train the initial model and then re-train to solve the main task.
- Pre-trained model approach. Download online available models and then re-train to solve the main task.

\
Model re-training   
The parameters of the transfered model would be frozen during model re-training, so that it's not updated by back propagation.
We would also need to select how much of the transfered model we want to use, in general the last layer (classification/regression layer) of the transferred model would be replaced with our own layer to solve our own problem. 
We would also have to ensure that our input data matches the type of data the transferred model expects at the input layer.

\
Example transfer learning with expert models trained on ImageNet
- ImageNet dataset contains colored images with a size of 224x224 so we would have to resize our own input image to meet the requirement,
- The ImageNet dataset contains 1001 class so the final classification layer would have to be removed and replaced with our, own layer to solve our own problem.


*KEY NOTES:*
- The transfered model needs to be trained on a related task, we can't just pick any random model to have an effective transfer learning
- There is also an open question as to how much of the pre-trained model to use.


**Saving and loading model**   
Models can be saved using tensorflow API (saved model format) or Keras API (save model as a HDF5 file). This is generally straightforward, the key difference between the 2 approach is that with the Keras API method you would need to know the model structure when reloading the model from file. With the TensorFlow API method no prior knowledge is needed to load the model from the saved model format.


## Lesson 8. Time Series Forecasting

## Resource
Useful link
- [Tensorflow API docs](https://www.tensorflow.org/api_docs/python/tf)
