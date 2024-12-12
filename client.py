import flwr as fl
from tensorflow import keras
import os
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# Load model
def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Load data and split it by class labels
def load_data(client_id):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define which classes belong to which client
    if client_id == 0:
        # Client 0 gets classes 1, 2, 3, 7
        class_labels = [0, 1, 2, 3, 7]
    else:
        # Client 1 gets classes 4, 5, 6, 8
        class_labels = [4, 5, 6, 8, 9]

    # Filter training data by class labels
    x_train_filtered = x_train[np.isin(y_train, class_labels)]
    y_train_filtered = y_train[np.isin(y_train, class_labels)]
    x_test_filtered = x_test[np.isin(y_test, class_labels)]
    y_test_filtered = y_test[np.isin(y_test, class_labels)]
    
    return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered

# Method to plot class distribution in training data
def plot_class_distribution(y_train, client_id):
    counts = Counter(y_train)
    labels, values = zip(*sorted(counts.items()))  # Sort classes by label
    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title(f"Data Distribution for Client {client_id}")
    plt.show()

# Define Flower client
class AsyncClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, client_id):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.client_id = client_id
        self.history = {
            "training_loss": [],
            "training_accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
        self.eval_accuracy = []
        self.eval_loss = []

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.x_train, self.y_train, 
            validation_data=(self.x_test, self.y_test),
            epochs=1, verbose=0
        )
        # Collect metrics
        self.history["training_loss"].append(history.history["loss"][-1])
        self.history["training_accuracy"].append(history.history["accuracy"][-1])
        self.history["val_loss"].append(history.history["val_loss"][-1])
        self.history["val_accuracy"].append(history.history["val_accuracy"][-1])

        print(f"Training Accuracy: {history.history['accuracy'][-1]} - Training Loss: {history.history['loss'][-1]}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][-1]} - Validation Loss: {history.history['val_loss'][-1]}")
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Eval Accuracy: {accuracy} - Eval Loss: {loss}")
        
        # Store evaluation metrics
        self.eval_accuracy.append(accuracy)
        self.eval_loss.append(loss)
        
        return loss, len(self.x_test), {"accuracy": accuracy}

    def plot_training_metrics(self):
        rounds = list(range(1, len(self.history["training_loss"]) + 1))
        # Training Accuracy
        plt.figure()
        plt.plot(rounds, self.history["training_accuracy"], label="Training Accuracy")
        plt.plot(rounds, self.history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy per Round")
        plt.legend()
        plt.show()

        # Training Loss
        plt.figure()
        plt.plot(rounds, self.history["training_loss"], label="Training Loss")
        plt.plot(rounds, self.history["val_loss"], label="Validation Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Round")
        plt.legend()
        plt.show()

    def plot_data_distribution(self):
        # Plot data distribution per client
        plot_class_distribution(self.y_train, self.client_id)

    def plot_eval_metrics(self):
        # After all rounds, plot evaluation accuracy and loss
        rounds = list(range(1, len(self.eval_accuracy) + 1))
        # Plot Evaluation Accuracy
        plt.figure()
        plt.plot(rounds, self.eval_accuracy, label="Eval Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.title("Evaluation Accuracy per Round")
        plt.legend()
        plt.show()

        # Plot Evaluation Loss
        plt.figure()
        plt.plot(rounds, self.eval_loss, label="Eval Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss per Round")
        plt.legend()
        plt.show()

# Client process function
def run_client(client_id):
    model = get_model()
    x_train, y_train, x_test, y_test = load_data(client_id)
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    client = AsyncClient(model, x_train, y_train, x_test, y_test, client_id)
    fl.client.start_numpy_client(server_address=server_address, client=client)
    client.plot_data_distribution()  # Show data distribution for the client
    client.plot_training_metrics()   # Plot training metrics
    client.plot_eval_metrics()       # Plot evaluation metrics after all rounds

if __name__ == "__main__":
    client_processes = []
    num_threads= 2 
    for thread_id in range(num_threads): 
        client_process = multiprocessing.Process(target=run_client, args=(thread_id,)) 
        client_process.start()
        client_processes.append(client_process)
    for client_process in client_processes:
        client_process.join()

