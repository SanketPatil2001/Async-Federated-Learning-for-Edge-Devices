import flwr as fl

class AsyncFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            print(f"Aggregated weights at round {rnd}")
        return aggregated_weights

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=AsyncFedAvg(),
    config=fl.server.ServerConfig(num_rounds=10),
)



