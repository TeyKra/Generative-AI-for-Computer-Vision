import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_mnist():
    # Load the MNIST dataset
    (x_train, _), (x_test, _) = mnist.load_data()
    # Normalize the data and convert to float32
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    # Add a channel to make dimensions (batch_size, 28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    return x_train, x_test
