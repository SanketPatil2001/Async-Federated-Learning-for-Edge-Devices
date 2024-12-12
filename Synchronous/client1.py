import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Helper functions
def plot_class_distribution(y):
    """Plot the distribution of classes in the target labels."""
    counts = Counter(y)
    labels, values = zip(*sorted(counts.items()))  # Sorted order of classes

    # Visualize the class distribution
    sns.barplot(x=labels, y=values, palette="viridis")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.show()

def filter_data_to_distribution(dist, x, y):
    """Filter the data to match the given class distribution."""
    counts = np.zeros(len(dist), dtype=int)
    mask = np.zeros(len(y), dtype=bool)

    for i in range(len(y)):
        if counts[y[i]] < dist[y[i]]:
            mask[i] = True
            counts[y[i]] += 1

    return x[mask], y[mask]

# Load and compile the Keras model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

# Define desired class distribution for the training set
desired_distribution = [4000, 4000, 4000, 3000, 10, 10, 10, 10, 4000, 10]
x_train, y_train = filter_data_to_distribution(desired_distribution, x_train, y_train)

# Visualize class distribution
plot_class_distribution(y_train)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Return model weights."""
        return model.get_weights()

    def fit(self, parameters, config):
        """Train the model and return updated weights."""
        model.set_weights(parameters)
        result = model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test), verbose=0)
        print("Fit history:", result.history)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model and return loss and accuracy."""
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("Evaluation accuracy:", accuracy)
        return loss, len(x_test), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address=f"localhost:{sys.argv[1]}",
    client=FlowerClient(),
    grpc_max_message_length=1024 * 1024 * 1024  # Set max message length to 1GB
)
