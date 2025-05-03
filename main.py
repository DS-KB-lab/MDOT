import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from ecg_datasets import ECGDataset
from evaluation import ECGEvaluator
from visualization import plot_training_metrics, plot_confusion_matrix
from KD_base_model.basemodel import MDOT
from KD_base_model.criterion import LabelSmoothFocalLoss
from KD_base_model.train_teacher import train_teacher
from KD_base_model.train_student import train_student

def parse_args():
    parser = argparse.ArgumentParser(description='MDOT ECG Classification')
    parser.add_argument('--dataset', type=str, default='mitbih', choices=['mitbih', 'chapman'],
                      help='Dataset to use (default: mitbih)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                      help='Learning rate (default: 1e-4)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (default: cuda if available)')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save results (default: results)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    train_dataset = ECGDataset(dataset_type=args.dataset, split='train')
    val_dataset = ECGDataset(dataset_type=args.dataset, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize models
    print("Initializing models...")
    teacher_model = train_teacher(train_loader, val_loader, args)
    student_model = train_student(train_loader, val_loader, teacher_model, args)
    
    # Initialize evaluator
    evaluator = ECGEvaluator(num_classes=train_dataset.num_classes)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    criterion = LabelSmoothFocalLoss()
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        # Training
        student_model.train()
        train_loss = 0
        evaluator.reset()
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Training'):
            ecg_data = batch['ecg'].to(args.device)
            labels = batch['label'].to(args.device)
            
            # Forward pass
            outputs = student_model(ecg_data)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            evaluator.update(outputs.argmax(dim=1), labels)
        
        train_loss /= len(train_loader)
        train_metrics = evaluator.compute_metrics()
        train_losses.append(train_loss)
        train_accs.append(train_metrics['accuracy'])
        
        # Validation
        student_model.eval()
        val_loss = 0
        evaluator.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Validation'):
                ecg_data = batch['ecg'].to(args.device)
                labels = batch['label'].to(args.device)
                
                outputs = student_model(ecg_data)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                evaluator.update(outputs.argmax(dim=1), labels)
        
        val_loss /= len(val_loader)
        val_metrics = evaluator.compute_metrics()
        val_losses.append(val_loss)
        val_accs.append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save plots
        plot_training_metrics(train_losses, val_losses, train_accs, val_accs,
                            os.path.join(args.save_dir, 'training_metrics.png'))
        
        # Save confusion matrix
        confusion = evaluator.get_confusion_matrix()
        plot_confusion_matrix(confusion, train_dataset.class_names,
                            os.path.join(args.save_dir, 'confusion_matrix.png'))
        
        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    # Final evaluation
    print("\nFinal Evaluation:")
    evaluator.print_report(train_dataset.class_names)

if __name__ == '__main__':
    main() 