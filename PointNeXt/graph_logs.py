import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np

def parse_log_file(log_file_path):
    """
    Parses the stdout.log file to extract training and validation metrics per epoch.
    """
    train_data_by_epoch = {}
    val_data_by_epoch = {}

    with open(log_file_path, 'r') as f:
        for line in f:
            # Capture final epoch summary line
            summary_match = re.search(r"Epoch (\d+) LR ([\d.]+) train_loss ([\d.]+), val_mse ([\d.]+), best val mse ([\d.]+)", line)
            if summary_match:
                epoch = int(summary_match.group(1))
                lr = float(summary_match.group(2))
                train_loss = float(summary_match.group(3))
                val_mse_summary = float(summary_match.group(4)) 
                
                train_data_by_epoch[epoch] = {'loss': train_loss, 'lr': lr, 'mse': np.nan, 'mae': np.nan} 
                val_data_by_epoch[epoch] = {'loss': np.nan, 'mse': val_mse_summary, 'mae': np.nan} 

            # Capture accurate val data (if val_freq is > 1, this line won't appear every epoch)
            val_detail_match = re.search(r"Val Epoch \[(\d+)\] Loss ([\d.]+) MSE ([\d.]+) MAE ([\d.]+)", line)
            if val_detail_match:
                epoch = int(val_detail_match.group(1))
                val_loss = float(val_detail_match.group(2))
                val_mse = float(val_detail_match.group(3))
                val_mae = float(val_detail_match.group(4))
                val_data_by_epoch[epoch] = {'loss': val_loss, 'mse': val_mse, 'mae': val_mae}

    # Convert dictionaries to sorted lists
    train_epochs = sorted(train_data_by_epoch.keys())
    train_losses = [train_data_by_epoch[e]['loss'] for e in train_epochs]
    train_mses = [train_data_by_epoch[e].get('mse', np.nan) for e in train_epochs] 
    train_maes = [train_data_by_epoch[e].get('mae', np.nan) for e in train_epochs] 
    train_lrs = [train_data_by_epoch[e]['lr'] for e in train_epochs]

    val_epochs = sorted(val_data_by_epoch.keys())
    val_losses = [val_data_by_epoch[e]['loss'] for e in val_epochs]
    val_mses = [val_data_by_epoch[e]['mse'] for e in val_epochs]
    val_maes = [val_data_by_epoch[e]['mae'] for e in val_epochs]


    return {
        'train_epochs': train_epochs,
        'train_losses': train_losses,
        'train_mses': train_mses,
        'train_maes': train_maes,
        'train_lrs': train_lrs,
        'val_epochs': val_epochs,
        'val_losses': val_losses,
        'val_mses': val_mses,
        'val_maes': val_maes,
    }

def plot_metrics(metrics_data, title_prefix="Training Metrics"):
    """
    Plots training and validation metrics.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12)) # 3 plots: Loss, MSE, MAE

    # Plot Loss
    if metrics_data['train_epochs'] and metrics_data['train_losses']:
        axes[0].plot(metrics_data['train_epochs'], metrics_data['train_losses'], label='Train Loss')
    if metrics_data['val_epochs'] and metrics_data['val_losses']:
        axes[0].plot(metrics_data['val_epochs'], metrics_data['val_losses'], label='Val Loss', marker='o')
    axes[0].set_title(f'{title_prefix} - Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot MSE
    if metrics_data['train_epochs'] and metrics_data['train_mses']: 
        axes[1].plot(metrics_data['train_epochs'], metrics_data['train_mses'], label='Train MSE')
    if metrics_data['val_epochs'] and metrics_data['val_mses']:
        axes[1].plot(metrics_data['val_epochs'], metrics_data['val_mses'], label='Val MSE', marker='o')
    axes[1].set_title(f'{title_prefix} - MSE over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True)

    # Plot MAE
    if metrics_data['train_epochs'] and metrics_data['train_maes']: 
        axes[2].plot(metrics_data['train_epochs'], metrics_data['train_maes'], label='Train MAE')
    if metrics_data['val_epochs'] and metrics_data['val_maes']:
        axes[2].plot(metrics_data['val_epochs'], metrics_data['val_maes'], label='Val MAE', marker='o')
    axes[2].set_title(f'{title_prefix} - MAE over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('MAE')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def plot_2d_histogram(predictions_path, targets_path, title="Predicted vs. Actual Angle"):
    """
    Loads saved predictions and targets and plots a 2D histogram.
    """
    try:
        predictions = np.load(predictions_path)
        targets = np.load(targets_path)
    except FileNotFoundError:
        print(f"Error: Prediction or target file not found.")
        print(f"  Predictions: {predictions_path}")
        print(f"  Targets: {targets_path}")
        return

    # Handle cases where predictions/targets might be scalar arrays (if only one sample was saved)
    if predictions.ndim == 0: 
        predictions = np.array([predictions.item()])
    if targets.ndim == 0:
        targets = np.array([targets.item()])
        
    # Ensure they are 1D arrays for histogramming
    predictions = predictions.flatten()
    targets = targets.flatten()

    if len(predictions) != len(targets) or len(predictions) == 0:
        print("Error: Mismatched or empty prediction/target arrays.")
        return

    plt.figure(figsize=(8, 8))
    
    # Create the 2D histogram
    # bins can be adjusted based on the range of your angles or data distribution
    # cmin=1 ensures that only bins with at least one sample are colored
    plt.hist2d(targets, predictions, bins=50, cmap='viridis', cmin=1) 
    plt.colorbar(label='Density of Samples')
    
    # Add a diagonal line for perfect prediction (y=x)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction') # Red dashed line

    plt.title(title)
    plt.xlabel('Actual Angle')
    plt.ylabel('Predicted Angle')
    plt.grid(True)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box') # Ensures the x and y axes have the same scale
    plt.show()

if __name__ == "__main__":
    # --- IMPORTANT: Configure this ---
    # The base components of your log directory structure:
    LOG_ROOT_DIR = "log_regression" # This should match 'root_dir' in your YAML (corrected typo)
    TASK_NAME = "regression_events" # This should match 'task_name' in your YAML
    EXP_NAME_PREFIX = "pointnext_opang_regression" # This is the prefix of your run_name

    # You need to provide the EXACT full name of your timestamped run directory.
    # Copy this name from the 'run_dir' line in your console output when you run main.py
    # Example from your output: regression_events-train-pointnext_opang_regression-ngpus1-seed0-20250728-161840-EqsuYrWNpLBN7Hrst9RBSj
    ACTUAL_RUN_TIMESTAMPED_FOLDER = "regression_events-train-pointnext_opang_regression-ngpus1-seed0-20250728-163143-dzs9hXcUBGvkXVmLafhyUS" # <--- UPDATE THIS EXACTLY!

    # Construct the base_run_dir using os.path.join for platform compatibility
    # This assumes plot_logs.py is run from the 'eplusminus' directory.
    base_run_dir = os.path.join(
        LOG_ROOT_DIR,
        TASK_NAME,
        ACTUAL_RUN_TIMESTAMPED_FOLDER
    )

    # --- Plot Training Metrics (Loss/MSE/MAE over Epochs) ---
    log_file = os.path.join(base_run_dir, "stdout.log") 
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        print(f"  Attempted path: {os.path.abspath(log_file)}") # Print full resolved path for debugging
        sys.exit(1)

    print(f"Parsing log file: {log_file}")
    metrics = parse_log_file(log_file)

    if metrics['train_epochs'] or metrics['val_epochs']:
        plot_metrics(metrics, title_prefix="PointNeXt Regression Training")
    else:
        print("No metrics found in log file to plot.")

    # --- Plot 2D Histogram (Predicted vs. Actual Angle) ---
    predictions_file = os.path.join(base_run_dir, "best_epoch_test_predictions.npy")
    targets_file = os.path.join(base_run_dir, "best_epoch_test_targets.npy")

    if os.path.exists(predictions_file) and os.path.exists(targets_file):
        print(f"\nPlotting 2D histogram from: {predictions_file} and {targets_file}")
        plot_2d_histogram(predictions_file, targets_file, title="Angle Prediction Histogram (Best Epoch)")
    else:
        print(f"\nError: Prediction/Target .npy files not found for 2D histogram.")
        print(f"Ensure training completed successfully and files were saved to: {base_run_dir}")