import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

def plot_attention_weights(attention_weights, ecg_signal, save_path=None):
    """
    Plot attention weights overlaid on ECG signal
    
    Args:
        attention_weights (torch.Tensor): Attention weights of shape [sequence_length]
        ecg_signal (torch.Tensor): ECG signal of shape [sequence_length]
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot ECG signal
    plt.plot(ecg_signal.cpu().numpy(), 'b-', label='ECG Signal')
    
    # Create heatmap of attention weights
    weights = attention_weights.cpu().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min())  # Normalize to [0,1]
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('attention', ['white', 'red'])
    
    # Overlay attention weights
    plt.imshow(np.tile(weights, (50, 1)), 
              aspect='auto', 
              cmap=cmap, 
              alpha=0.3,
              extent=[0, len(ecg_signal), ecg_signal.min(), ecg_signal.max()])
    
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Attention Weights')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(confusion_matrix, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix (np.ndarray): Confusion matrix
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation metrics
    
    Args:
        train_losses (list): Training losses
        val_losses (list): Validation losses
        train_accs (list): Training accuracies
        val_accs (list): Validation accuracies
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Losses')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracies')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 