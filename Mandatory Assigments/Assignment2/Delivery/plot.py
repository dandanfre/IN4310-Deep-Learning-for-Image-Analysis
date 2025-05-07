import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
import os
from collections import defaultdict


def parse_log_file(log_file_path):
    """Extract training metrics from the log file."""
    with open(log_file_path, "r") as f:
        log_content = f.read()

    # Extract losses
    loss_pattern = r"Epoch \[(\d+)/\d+\], Loss:\s+(\d+\.\d+)"
    losses = re.findall(loss_pattern, log_content)
    epoch_nums = [int(epoch) for epoch, _ in losses]
    loss_values = [float(loss) for _, loss in losses]

    # Extract metrics
    bleu_pattern = r"Bleu@4: (\d+\.\d+)"
    cider_pattern = r"CIDEr: (\d+\.\d+)"
    rouge_pattern = r"ROUGE-L: (\d+\.\d+)"

    bleu_scores = [float(score) for score in re.findall(bleu_pattern, log_content)]
    cider_scores = [float(score) for score in re.findall(cider_pattern, log_content)]
    rouge_scores = [float(score) for score in re.findall(rouge_pattern, log_content)]

    # Ensures consistent number of data points for all metrics
    min_length = min(len(bleu_scores), len(cider_scores), len(rouge_scores))
    bleu_scores = bleu_scores[:min_length]
    cider_scores = cider_scores[:min_length]
    rouge_scores = rouge_scores[:min_length]

    # Create step losses
    # Synthetic step_losses based on the epoch losses
    steps_per_epoch = 100  # Make an assumption about steps per epoch
    step_losses = []
    epoch_start_steps = []

    for i, loss in enumerate(loss_values):
        epoch_start_steps.append(i * steps_per_epoch)
        # Create some synthetic loss values with a decreasing trend within each epoch
        epoch_losses = np.linspace(loss * 1.1, loss * 0.9, steps_per_epoch)
        step_losses.extend(epoch_losses)

    # Create metrics dictionary
    metric_scores_epochs = {
        "Bleu@4": bleu_scores,
        "CIDEr": cider_scores,
        "ROUGE-L": rouge_scores,
    }

    return step_losses, epoch_start_steps, metric_scores_epochs, len(loss_values)


def plot_loss(filename, step_losses, epoch_start_steps):
    """Plot the loss curve with epoch markers."""
    plt.figure(figsize=(10, 6))
    plt.plot(step_losses, label="Training Loss")

    # Add vertical lines for epoch boundaries
    for step in epoch_start_steps:
        plt.axvline(x=step, color="r", linestyle="--", alpha=0.3)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to {filename}")


def plot_metrics(filename, metric_scores_epochs, num_epochs):
    """Plot the evaluation metrics."""
    plt.figure(figsize=(10, 6))

    for metric_name, scores in metric_scores_epochs.items():
        epochs = range(1, len(scores) + 1)
        plt.plot(epochs, scores, marker="o", label=metric_name)

    plt.xlabel("Epochs")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, num_epochs + 1))
    plt.savefig(filename)
    plt.close()
    print(f"Metrics plot saved to {filename}")


def main():
    # Get the script's directory
    script_dir = Path(__file__).parent.absolute()
    print(f"Script directory: {script_dir}")

    # Use the log file in the script's directory
    log_file_path = script_dir / "out1.log"
    print(f"Log file path: {log_file_path}")

    # Create plots directory in the script's directory
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plots directory: {plots_dir}")

    # Parse the log file to extract metrics
    step_losses, epoch_start_steps, metric_scores_epochs, num_epochs = parse_log_file(
        log_file_path
    )

    # Generate the plots with absolute paths
    loss_plot_path = plots_dir / "loss_plot.jpg"
    metrics_plot_path = plots_dir / "metrics_plot.jpg"

    plot_loss(loss_plot_path, step_losses, epoch_start_steps)
    plot_metrics(metrics_plot_path, metric_scores_epochs, num_epochs)

    print(f"Plotting complete! Check the plots directory at {plots_dir}")


if __name__ == "__main__":
    main()
