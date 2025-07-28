import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np

def parse_log_file(log_file_path):
    #Parses the stdout.log file to extract training and validation metrics per epoch.
    train_epochs = []
    train_losses = []
    train_mses = []
    train_maes = []
    train_lrs = [] # Learning rates

    val_epochs = []
    val_losses = []
    val_mses = []
    val_maes = []

    with open(log_file_path, 'r') as f:
        for line in f:
            # Regex to match the 'Train Epoch [X/Y] Loss Z MSE A MAE B' line (from tqdm description)
            train_tqdm_match = re.search(r"Train Epoch \[(\d+)/\d+\] Loss ([\d.]+) MSE ([\d.]+) MAE ([\d.]+)", line)
            # Regex to match the 'Epoch X LR YYYY train_loss ZZZZ, val_mse AAAA, best val mse BBBB' line (from logging.info after epoch)
            train_epoch_summary_match = re.search(r"Epoch (\d+) LR ([\d.]+) train_loss ([\d.]+), val_mse ([\d.]+), best val mse ([\d.]+)", line)


            if train_tqdm_match:
                # We prioritize the summary line for epoch metrics if available, as it's definitive for the epoch end.
                # However, if train_tqdm_match is captured, we need to ensure we don't duplicate.
                # A common pattern is to collect all logs per epoch.
                pass # We'll rely on the train_epoch_summary_match below for final epoch metrics

            if train_epoch_summary_match:
                epoch = int(train_epoch_summary_match.group(1))
                lr = float(train_epoch_summary_match.group(2))
                train_loss = float(train_epoch_summary_match.group(3))
                # Note: train_mse and val_mse from this line are from the summary, might not be exact end-of-epoch values
                # if tqdm description updates frequently, but are what's logged in info.
                
                # Check if epoch already exists (to avoid duplicates if parsed from different log messages)
                if not train_epochs or train_epochs[-1] != epoch:
                    train_epochs.append(epoch)
                    train_losses.append(train_loss)
                    train_lrs.append(lr)
                    # MSE and MAE for training are typically logged in tqdm. We'll use the final value from summary.
                    # Or, better, collect them from the tqdm description lines if they are consistent.
                    # For simplicity, if these aren't consistent, we only plot LR here
                    train_mses.append(np.nan) # Placeholder if not directly in summary line
                    train_maes.append(np.nan) # Placeholder if not directly in summary line


            # Match validation epoch line: "Val Epoch [X] Loss Y MSE A MAE B"
            val_match = re.search(r"Val Epoch \[(\d+)\] Loss ([\d.]+) MSE ([\d.]+) MAE ([\d.]+)", line)
            if val_match:
                epoch = int(val_match.group(1))
                loss = float(val_match.group(2))
                mse = float(val_match.group(3))
                mae = float(val_match.group(4))
                # Check if epoch already exists (to avoid duplicates)
                if not val_epochs or val_epochs[-1] != epoch:
                    val_epochs.append(epoch)
                    val_losses.append(loss)
                    val_mses.append(mse)
                    val_maes.append(mae)

    # Final pass to ensure all data points align.
    # Sometimes parsing order can result in val metrics before train for the same epoch.
    # It's safer to store into dictionaries then convert to sorted lists
    # For now, assuming relatively ordered log.
    
    # If train_mses/maes were not reliably parsed, replace with LR values.
    # The current train_mod.py logs train_loss, val_mse, val_mae in the summary logging.info.
    # So let's re-align the parsing based on this.

    # Revised Parsing Strategy: Capture all relevant info from the final logging.info line per epoch
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
                # Note: The validation metrics logged in this summary line are for the *last* validation, which might not be every epoch
                val_mse_summary = float(summary_match.group(4)) 
                
                train_data_by_epoch[epoch] = {'loss': train_loss, 'lr': lr, 'mse': np.nan, 'mae': np.nan} # MSE/MAE from tqdm will need separate parsing
                val_data_by_epoch[epoch] = {'loss': np.nan, 'mse': val_mse_summary, 'mae': np.nan} # Only MSE is in summary

            # Capture tqdm update line for accurate train MSE/MAE per batch, then get last one
            # This is tricky for per-epoch final value.
            # Simpler: just get final epoch values from the summary lines or just from val.

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
    train_mses = [train_data_by_epoch[e].get('mse', np.nan) for e in train_epochs] # May be nan if not parsed
    train_maes = [train_data_by_epoch[e].get('mae', np.nan) for e in train_epochs] # May be nan
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
    if metrics_data['train_epochs'] and metrics_data['train_mses']: # May be empty if not parsed
        axes[1].plot(metrics_data['train_epochs'], metrics_data['train_mses'], label='Train MSE')
    if metrics_data['val_epochs'] and metrics_data['val_mses']:
        axes[1].plot(metrics_data['val_epochs'], metrics_data['val_mses'], label='Val MSE', marker='o')
    axes[1].set_title(f'{title_prefix} - MSE over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True)

    # Plot MAE
    if metrics_data['train_epochs'] and metrics_data['train_maes']: # May be empty if not parsed
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

# --- Main execution to run the plotting ---
if __name__ == "__main__":
    # --- IMPORTANT: Configure this ---
    # You need to specify the path to your stdout.log file.
    # It will be under: log_regression/regression_events/pointnext_opang_regression/[TIMESTAMP]/stdout.log
    # Example:
	log_file = "log_rregression/regression_events/regression_events-train-pointnext_opang_regression-ngpus1-seed0-20250728-091107-YEMyfkw9Ac4u2bZgSL97Hp/regression_events-train-pointnext_opang_regression-ngpus1-seed0-20250728-091107-YEMyfkw9Ac4u2bZgSL97Hp.log"
    #log_file = "log_regression/regression_events/pointnext_opang_regression/20250724-114447-6qicz2VTx3fkpNcDTaTgkt/stdout.log" # <--- UPDATE THIS PATH
    
    # Make sure the log file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        sys.exit(1)

    print(f"Parsing log file: {log_file}")
    metrics = parse_log_file(log_file)

    if metrics['train_epochs'] or metrics['val_epochs']:
        plot_metrics(metrics, title_prefix="PointNeXt Regression Training")
    else:
        print("No metrics found in log file to plot.")