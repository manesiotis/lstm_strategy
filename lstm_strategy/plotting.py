import matplotlib.pyplot as plt
import os

def plot_predictions(actual, predicted, title="LSTM Prediction", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual", color='black')
    plt.plot(predicted, label="Predicted", color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
