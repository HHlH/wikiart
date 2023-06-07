
# Models


This repository contains code for training two CNN models for image classification using PyTorch. 

# Data Set
 The models are trained on the WikiArt dataset of images labeled with 10 genres.

Link to the original data set: https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

# Training
To train the models, run the provided code. Make sure to specify the directory of your data set in load_dir. Set the appropriate values for variables like load_dir, num_workers, num_epoch, batch_size, and learning_rate before training.

The training loop outputs the training accuracy for each batch and the validation accuracy for each epoch.

You can modify the code to save the trained models or visualize the training progress as needed.
