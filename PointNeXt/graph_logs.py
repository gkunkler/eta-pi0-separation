import matplotlib.pyplot as plt
import re
import os
import sys
import numpy as np
from torchmetrics import ROC, AUROC
import torch
from coloredlines import colored_line

# Should be the same as in slim_data.py
interaction_descriptions = [([22, 22], 0),
                            ([111, 211, -211], 1),
                            ([111, 111, 111], 2),
                            ([111, 111], 3),
                            ([111], 4)]

def parse_log_file(log_file_path):
    """
    Parses the stdout.log file to extract training and validation metrics per epoch.
    """
    train_data_by_epoch = {}
    val_data_by_epoch = {}

    with open(log_file_path, 'r') as f:
        for line in f:
            # Capture final epoch summary line
            summary_match = re.search(r"TRAINING PROGRESS: Epoch (\d+) LR ([\d.]+) train_loss ([\d.]+), train_accuracy ([\d.]+), train_precision ([\d.]+)", line)
            if summary_match:
                epoch = int(summary_match.group(1))
                lr = float(summary_match.group(2))
                train_loss = float(summary_match.group(3))
                train_accuracy = float(summary_match.group(4)) 
                train_precision = float(summary_match.group(5)) 
                
                train_data_by_epoch[epoch] = {'loss': train_loss, 'lr': lr, 'accuracy': train_accuracy, 'precision': train_precision} 
                # val_data_by_epoch[epoch] = {'loss': np.nan, 'mse': val_mse_summary, 'mae': np.nan} 

            # Capture accurate val data (if val_freq is > 1, this line won't appear every epoch)
            val_detail_match = re.search(r"VALIDATION RESULTS: Epoch (\d+) val_loss ([\d.]+), val_accuracy ([\d.]+), val_precision ([\d.]+)", line)
            if val_detail_match:
                epoch = int(val_detail_match.group(1))
                val_loss = float(val_detail_match.group(2))
                val_accuracy = float(val_detail_match.group(3))
                val_precision = float(val_detail_match.group(4))
                val_data_by_epoch[epoch] = {'loss': val_loss, 'accuracy': val_accuracy, 'precision': val_precision}

    # Convert dictionaries to sorted lists
    train_epochs = sorted(train_data_by_epoch.keys())
    train_losses = [train_data_by_epoch[e]['loss'] for e in train_epochs]
    train_accuracies = [train_data_by_epoch[e].get('accuracy', np.nan) for e in train_epochs] 
    train_precisions = [train_data_by_epoch[e].get('precision', np.nan) for e in train_epochs] 
    train_lrs = [train_data_by_epoch[e]['lr'] for e in train_epochs]

    val_epochs = sorted(val_data_by_epoch.keys())
    val_losses = [val_data_by_epoch[e]['loss'] for e in val_epochs]
    val_accuracies = [val_data_by_epoch[e]['accuracy'] for e in val_epochs]
    val_precisions = [val_data_by_epoch[e]['precision'] for e in val_epochs]


    return {
        'train_epochs': train_epochs,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'train_precisions': train_precisions,
        'train_lrs': train_lrs,
        'val_epochs': val_epochs,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_precisions': val_precisions,
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
    # axes[0].set_yscale('log')

    # Plot Accuracy
    if metrics_data['train_epochs'] and metrics_data['train_accuracies']: 
        axes[1].plot(metrics_data['train_epochs'], metrics_data['train_precisions'], label='Train Accuracy')
    if metrics_data['val_epochs'] and metrics_data['val_accuracies']:
        axes[1].plot(metrics_data['val_epochs'], metrics_data['val_accuracies'], label='Val Accuracy', marker='o')
    axes[1].set_title(f'{title_prefix} - Accuracy over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Precision
    if metrics_data['train_epochs'] and metrics_data['train_precisions']: 
        axes[2].plot(metrics_data['train_epochs'], metrics_data['train_precisions'], label='Train Precision')
    if metrics_data['val_epochs'] and metrics_data['val_precisions']:
        axes[2].plot(metrics_data['val_epochs'], metrics_data['val_precisions'], label='Val Precision', marker='o')
    axes[2].set_title(f'{title_prefix} - Precision over Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Precision')
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

def plot_predictions(predictions_path, targets_path, descriptions_path, title="Eta Pi0 Separation ROC"):

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

    # Get the descriptions if it exists
    if os.path.exists(descriptions_file):
        try:
            descriptions = np.load(descriptions_path)
            
        except:
            print(f"Descriptions not found: {descriptions_path}")
            descriptions = np.ones(len(targets))*-1
        if descriptions.ndim == 0:
            descriptions = np.array([descriptions.item()])
    else:
        descriptions = np.ones(len(targets))*-1

    descriptions = descriptions.flatten()

    # print(f'predictions: {predictions}')
    # print(f'targets: {targets}')
    # print(f'descriptions: {descriptions}')

    if len(predictions) != len(targets) or len(predictions) == 0 or len(descriptions) != len(targets):
        print("Error: Mismatched or empty prediction/target/description arrays.")
        return

    # Separate by target value

    
    labels=[]
    separated_predictions = []
    for particles, description in interaction_descriptions:
        print(f'{particles}, {description}')
        separated_predictions.append(predictions[(targets==1) & (descriptions == description)])
        labels.append(f'1 (Eta) - {particles}')
    separated_predictions.append(predictions[(targets==1) & (descriptions == -1)])
    labels.append(f'1 (Eta) - other')
    separated_predictions.append(predictions[targets==0])
    labels.append('0 (Pi0)')

    # predictions_0 = predictions[targets==0]
    

    # Create bins
    bins = np.linspace(0,1,20)
    bin_centers = (bins[:-1]+bins[1:])/2
    bin_width = np.diff(bins)[0]

    # Create the histograms
    # hist_0 = np.histogram(predictions_0, bins)[0]
    # hist_1 = np.histogram(predictions_1, bins)[0]

    fig, ax = plt.subplots(dpi=200)
    ax.hist(tuple(separated_predictions), bins, stacked=True, label=labels)

    # ax.hist(hist_1, bins, stacked=True, label='1', color='b')

    plt.title(title)
    plt.xlabel('Model Prediction')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

def plot_roc(predictions_path, targets_path, title="Eta Pi0 Separation Results"):

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

    fig, ax = plt.subplots(dpi=200, figsize=(5,4))

    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(torch.tensor(predictions, dtype=torch.float), torch.tensor(targets, dtype=torch.int))

    auroc = AUROC(task="binary")
    auc = auroc(torch.tensor(predictions, dtype=torch.float), torch.tensor(targets, dtype=torch.int)).item()

    # ax.plot(fpr, tpr)
    line = colored_line(fpr, tpr, thresholds, ax, linewidth=3, cmap="plasma")
    cb = fig.colorbar(line, )

    ax.set_xlabel('1 - Eta Selection Purity (FP Rate)')
    ax.set_ylabel('Eta Selection Efficiency (TP Rate)')
    ax.set_title(f'Area Under Curve (AUC) = {auc:.2f}')
    cb.set_label('Threshold')
    ax.set_aspect('equal')

    ax.plot([0,1], [0,1], linestyle='--', c='k', linewidth=1)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    fig.suptitle('Receiver Operating Characteristic (ROC)')

    plt.show()

if __name__ == "__main__":
    # --- IMPORTANT: Configure this ---
    # The base components of your log directory structure:
    LOG_ROOT_DIR = "eta-pi0-classification" # This should match 'root_dir' in your YAML (corrected typo)
    TASK_NAME = "classification_events" # This should match 'task_name' in your YAML
    # TASK_NAME = "good runs"
    EXP_NAME_PREFIX = "pointnext_eta-pi0-classification" # This is the prefix of your run_name

    # You need to provide the EXACT full name of your timestamped run directory.
    # Copy this name from the 'run_dir' line in your console output when you run main.py
    # Example from your output: regression_events-train-pointnext_opang_regression-ngpus1-seed0-20250728-161840-EqsuYrWNpLBN7Hrst9RBSj
    ACTUAL_RUN_TIMESTAMPED_FOLDER = "classification_events-train-pointnext_eta-pi0-classification-ngpus1-seed0-20250808-111411-hx9mKTB8fu6ndjxc646bK7" # <--- UPDATE THIS EXACTLY!



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
        plot_metrics(metrics, title_prefix="PointNeXt Classification Training")
    else:
        print("No metrics found in log file to plot.")

    predictions_file = os.path.join(base_run_dir, "best_epoch_test_predictions.npy")
    targets_file = os.path.join(base_run_dir, "best_epoch_test_targets.npy")
    descriptions_file = os.path.join(base_run_dir, "best_epoch_test_descriptions.npy")

    if os.path.exists(predictions_file) and os.path.exists(targets_file):
        print(f"\nPlotting final predictions from: {predictions_file} and {targets_file}")
        plot_predictions(predictions_file, targets_file, descriptions_file)
        plot_roc(predictions_file, targets_file)
    else:
        print(f"\nError: Prediction/Target .npy files not found for 2D histogram.")
        print(f"Ensure training completed successfully and files were saved to: {base_run_dir}")