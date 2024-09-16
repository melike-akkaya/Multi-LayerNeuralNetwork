The project was developed as part of my coursework in the Fundamentals of Machine Learning - Fall 2023.


# Flower Species Classification using Multi-Layer Neural Network

This project involves the implementation of a Multi-Layer Neural Network (MLP) from scratch for the classification of 15 different flower species. The dataset is processed and fed into the neural network for training, followed by evaluation on the test set.

## Dataset

The dataset consists of images of 15 different flower species:
- **Classes:** Aster, Calendula, California Poppy, Coreopsis, Dandelion, Daisy, Iris, Lavender, Lily, Marigold, Orchid, Poppy, Rose, Sunflower, Tulip.
- **Train, Validation, Test splits:**
  - 700 images per species for training.
  - 150 images per species for validation.
  - 150 images per species for testing.

Images are resized to 64x64 pixels and converted to grayscale, with pixel values normalized to the range [0, 1].


## Multi-Layer Neural Network Architecture
- Input Layer: Processes 64x64 grayscale images (flattened into 4096 features).
- Hidden Layers: Configurable number of hidden layers with tunable number of neurons per layer.
- Activation Functions: Sigmoid, ReLU, and Tanh were experimented with.
- Output Layer: 15 neurons, one for each flower class.
- Loss Function Negative log-likelihood was used as the loss function.
- Gradient Descent: Mini-batch gradient descent was used for weight updates.
- Batch Size: Different batch sizes were experimented with (16, 32, 64, 128).
- Learning Rate: Initial learning rates between 0.005 and 0.02 were tried, with decay applied to improve convergence.
- Number of hidden layers: Tested with one and two hidden layers.
- Learning Rate Decay: Applied at the end of each epoch to ensure smoother convergence.
- The model's performance was evaluated based on accuracy, precision, recall, F1-score and confustion matrix. Results were recorded for different configurations of batch size, learning rate, and number of hidden layers.


## How to Run
Download the dataset and extract it to the appropriate directory (flowers15/train, flowers15/val, flowers15/test).
Run the provided Jupyter Notebook to preprocess the data, train the model, and evaluate its performance.


## Requirements
Python 3.x
Numpy
OpenCV
Matplotlib
Scikit-learn
Seaborn
PIL