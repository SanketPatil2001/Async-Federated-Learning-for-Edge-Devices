import flwr as fl
import matplotlib.pyplot as plt

class AsyncFedAvg(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__()
        self.metrics = {
            "rounds": [],
            "eval_accuracy": [],
            "eval_loss": []
        }

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_metrics = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_metrics:
            # Collect metrics for plotting
            self.metrics["rounds"].append(rnd)
            self.metrics["eval_loss"].append(aggregated_metrics[0])  # Loss
            self.metrics["eval_accuracy"].append(aggregated_metrics[1])  # Accuracy
            print(f"Round {rnd} - Eval Accuracy: {aggregated_metrics[1]} - Eval Loss: {aggregated_metrics[0]}")
        return aggregated_metrics

    def plot_metrics(self):
        rounds = self.metrics["rounds"]
        # Plot Evaluation Accuracy
        plt.figure()
        plt.plot(rounds, self.metrics["eval_accuracy"], label="Eval Accuracy")
        plt.xlabel("Rounds")
        plt.ylabel("Accuracy")
        plt.title("Evaluation Accuracy per Round")
        plt.legend()
        plt.show()

        # Plot Evaluation Loss
        plt.figure()
        plt.plot(rounds, self.metrics["eval_loss"], label="Eval Loss")
        plt.xlabel("Rounds")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss per Round")
        plt.legend()
        plt.show()

fl.server.start_server(
    server_address="0.0.0.0:8080",
    strategy=AsyncFedAvg(),
    config=fl.server.ServerConfig(num_rounds=10),
)