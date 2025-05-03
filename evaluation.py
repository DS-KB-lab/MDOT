import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
from typing import Dict, Tuple, List

class ECGEvaluator:
    def __init__(self, num_classes: int):
        """
        Initialize ECG Evaluator
        
        Args:
            num_classes (int): Number of classes in the classification task
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
        self.confusion = np.zeros((self.num_classes, self.num_classes))
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new batch of predictions and targets
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth labels
        """
        preds = predictions.cpu().numpy()
        targs = targets.cpu().numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(targs)
        
        # Update confusion matrix
        batch_confusion = confusion_matrix(targs, preds, labels=range(self.num_classes))
        self.confusion += batch_confusion
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dict[str, float]: Dictionary containing all computed metrics
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, predictions),
            'precision': precision_score(targets, predictions, average='macro'),
            'recall': recall_score(targets, predictions, average='macro'),
            'f1': f1_score(targets, predictions, average='macro')
        }
        
        # Compute class-wise metrics
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = precision_score(targets, predictions, labels=[i], average='micro')
            metrics[f'recall_class_{i}'] = recall_score(targets, predictions, labels=[i], average='micro')
            metrics[f'f1_class_{i}'] = f1_score(targets, predictions, labels=[i], average='micro')
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get the confusion matrix
        
        Returns:
            np.ndarray: Confusion matrix
        """
        return self.confusion
    
    def print_report(self, class_names: List[str] = None):
        """
        Print a detailed report of all metrics
        
        Args:
            class_names (List[str], optional): Names of the classes
        """
        metrics = self.compute_metrics()
        
        print("\nECG Classification Report")
        print("=" * 50)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision: {metrics['precision']:.4f}")
        print(f"Macro Recall: {metrics['recall']:.4f}")
        print(f"Macro F1-Score: {metrics['f1']:.4f}")
        
        if class_names:
            print("\nClass-wise Metrics:")
            print("-" * 50)
            for i, name in enumerate(class_names):
                print(f"\nClass: {name}")
                print(f"Precision: {metrics[f'precision_class_{i}']:.4f}")
                print(f"Recall: {metrics[f'recall_class_{i}']:.4f}")
                print(f"F1-Score: {metrics[f'f1_class_{i}']:.4f}")
        
        print("\nConfusion Matrix:")
        print("-" * 50)
        print(self.confusion) 