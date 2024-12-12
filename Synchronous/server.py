import flwr as fl
import sys
import numpy as np

class CustomFedAvgStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, round_num, results, failures):
        # Aggregate weights using the base class's method
        aggregated_weights = super().aggregate_fit(round_num, results, failures)
        
        if aggregated_weights is not None:
            # Save the aggregated weights to a file after each round
            print(f"Saving aggregated weights for round {round_num}...")
            np.savez(f"aggregated_weights_round_{round_num}.npz", *aggregated_weights)
        
        return aggregated_weights

# Create a custom strategy instance
strategy = CustomFedAvgStrategy()

# Start the Flower server for 5 rounds of federated learning
server_address = f"localhost:{sys.argv[1]}"
fl.server.start_server(
    server_address=server_address,
    config=fl.server.ServerConfig(num_rounds=5),
    grpc_max_message_length=1024*1024*1024,  # Maximum message size of 1GB
    strategy=strategy
)
