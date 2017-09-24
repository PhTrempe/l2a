# l2a (Learn to Add)

## Description

This is a small project which aims to show an example of applied machine 
learning in Python 3 with the Keras library and its TensorFlow backend to 
train a neural network model for it to learn to add two integers.

The project also aims to follow the 7 Steps of Machine Learning presented by 
Google in [this](https://www.youtube.com/watch?v=nKW8Ndu7Mjw) YouTube video.

1. Gathering Data (`dataset_generator.py`)
1. Preparing Data (`dataset_preparer.py`)
1. Choosing a Model (`model_builder.py`)
1. Training (`trainer.py`)
1. Evaluation (`trainer.py`)
1. Hyperparameter Tuning (`hyperparameters.py`)
1. Prediction (`predictor.py`)

# Usage

## Installing Dependencies

    conda install numpy scipy
    pip install tensorflow tensorflow-gpu keras h5py

## Running the Training Process

    python trainer.py

This will first generate a dataset if none exists yet. 
It will then prepare the dataset if no prepared dataset exists yet.
After that, it will build the model using the model builder 
(cf. `model_builder.py`) if no model exists yet. 
If an existing model is found, this model will be loaded to continue its 
training.
Once the prepared dataset and model are loaded, the training process is started.
N.B. Feel free to cancel the training process at any point, since it will be
possible to resume it later on by running the trainer again.

## Visualizing Training with TensorBoard

    tensorboard --logdir=./logs

## Running Predictions

This will build the model using the model builder 
(cf. `model_builder.py`) if no model exists yet. 
If an existing model is found, this model will be loaded.
Once the model is loaded, it is used to make predictions on given inputs.

    python predictor.py
