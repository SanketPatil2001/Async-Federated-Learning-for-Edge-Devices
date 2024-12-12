# import flwr as fl
# from tensorflow import keras
# import numpy as np
# import sys
# import multiprocessing

# # Load model
# def get_model():
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(128, activation="relu"),
#         keras.layers.Dense(10, activation="softmax")
#     ])
#     model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
#     return model

# # Load data
# def load_data(client_id):
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#     x_train, x_test = x_train / 255.0, x_test / 255.0

#     # Split data by client ID
#     num_clients = 2  # Adjust as needed
#     split_size = len(x_train) // num_clients
#     start = client_id * split_size
#     end = start + split_size

#     return x_train[start:end], y_train[start:end], x_test, y_test

# # Define Flower client
# class AsyncClient(fl.client.NumPyClient):
#     def __init__(self, model, x_train, y_train, x_test, y_test):
#         self.model = model
#         self.x_train = x_train
#         self.y_train = y_train
#         self.x_test = x_test
#         self.y_test = y_test

#     def get_parameters(self, config):
#         return self.model.get_weights()

#     def fit(self, parameters, config):
#         self.model.set_weights(parameters)
#         self.model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
#         return self.model.get_weights(), len(self.x_train), {}

#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
#         return loss, len(self.x_test), {"accuracy": accuracy}

# # Client process function
# def run_client(client_id):
#     model = get_model()
#     x_train, y_train, x_test, y_test = load_data(client_id)
#     fl.client.start_numpy_client(
#         server_address="localhost:8080", 
#         client=AsyncClient(model, x_train, y_train, x_test, y_test)
#     )

# if __name__ == "__main__":
#     # Start multiple clients concurrently using multiprocessing
#     client_processes = []
#     num_clients = 2  # Adjust based on how many clients you want
#     for client_id in range(num_clients):
#         client_process = multiprocessing.Process(target=run_client, args=(client_id,))
#         client_process.start()
#         client_processes.append(client_process)
    
#     # Wait for all client processes to finish
#     for client_process in client_processes:
#         client_process.join()


import flwr as fl
from tensorflow import keras
import os
import multiprocessing

# Load model
def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Load data
def load_data(client_id):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    num_clients = 2
    split_size = len(x_train) // num_clients
    start = client_id * split_size
    end = start + split_size
    return x_train[start:end], y_train[start:end], x_test, y_test

# Define Flower client
class AsyncClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, verbose=0)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Client process function
def run_client(client_id):
    model = get_model()
    x_train, y_train, x_test, y_test = load_data(client_id)
    server_address = os.getenv("SERVER_ADDRESS", "127.0.0.1:8080")
    fl.client.start_numpy_client(
        server_address=server_address, 
        client=AsyncClient(model, x_train, y_train, x_test, y_test)
    )

if __name__ == "__main__":
    client_processes = []
    num_clients = 2
    for client_id in range(num_clients):
        client_process = multiprocessing.Process(target=run_client, args=(client_id,))
        client_process.start()
        client_processes.append(client_process)
    for client_process in client_processes:
        client_process.join()
