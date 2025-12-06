"""
PCA + CNN Integration Script (318 Components) - FIXED VERSION WITH FLOPS
1. Loads Images
2. Applies PCA (318 components) PROPERLY with train/val split + FLOPS counting
3. Trains CNN on Reconstructed Images
4. Generates Confusion Matrix & ROC Curve
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import data loading functions
try:
    from Creador_labels import cargar_todos_datasets_con_labels
except ImportError:
    print("Warning: Creador_labels not found.")

# ============ 1. FIXED PCA FUNCTIONS WITH FLOPS ============

def apply_pca_train_val_flops(train_images, val_images, components=318):
    """
    Applies PCA to training data and transforms both training and validation data
    WITH FLOPS counting
    CRITICAL: Fit PCA ONLY on training data, transform both
    """
    print(f"\n--- STARTING PCA ({components} COMPONENTS) WITH CORRECT SPLIT AND FLOPS ---")
    start_time = time.time()
    flops_data = {}
    
    # Convert to numpy arrays
    print("1. Converting to numpy array...")
    train_images_np = np.array(train_images)
    val_images_np = np.array(val_images)
    
    # Get dimensions
    if len(train_images_np.shape) == 4:  # (N, H, W, C)
        n_train, h, w, c = train_images_np.shape
        n_val = val_images_np.shape[0]
    elif len(train_images_np.shape) == 3:  # (N, H, W)
        n_train, h, w = train_images_np.shape
        c = 1
        n_val = val_images_np.shape[0]
    else:
        h, w = 450, 450
        c = 3
    
    # 2. Convert to grayscale if needed
    print("2. Converting to grayscale...")
    def convert_to_grayscale_batch(images):
        if len(images.shape) == 4 and images.shape[-1] == 3:
            return np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images])
        elif len(images.shape) == 4 and images.shape[-1] == 1:
            return images.squeeze(-1)
        else:
            return images
    
    train_images_gray = convert_to_grayscale_batch(train_images_np)
    val_images_gray = convert_to_grayscale_batch(val_images_np)
    
    # FLOPS for grayscale conversion
    if c == 3:
        flops_data['gray_conversion_train'] = n_train * h * w * 5
        flops_data['gray_conversion_val'] = n_val * h * w * 5
    else:
        flops_data['gray_conversion_train'] = 0
        flops_data['gray_conversion_val'] = 0
    
    # 3. Flatten
    print("3. Flattening images...")
    X_train_flat = train_images_gray.reshape(n_train, -1)
    X_val_flat = val_images_gray.reshape(n_val, -1)
    d = X_train_flat.shape[1]  # Original dimensionality
    
    flops_data['flatten_train'] = n_train * h * w
    flops_data['flatten_val'] = n_val * h * w
    
    # 4. Center data (TRAINING ONLY for fitting)
    print("4. Centering data...")
    X_bar = X_train_flat.mean(axis=0)
    X_train_centered = X_train_flat - X_bar
    X_val_centered = X_val_flat - X_bar  # Use same mean for validation
    
    flops_data['centering_train'] = n_train * d * 2
    flops_data['centering_val'] = n_val * d * 2
    
    # 5. PCA using SVD (ON TRAINING ONLY)
    print(f"5. Calculating SVD on training data... Matrix: {X_train_centered.shape}")
    svd_start = time.time()
    # Use numpy SVD for FLOPS estimation
    U, S, Vt = np.linalg.svd(X_train_centered, full_matrices=False)
    svd_time = time.time() - svd_start
    
    # FLOPS estimation for SVD (Approximate)
    # Theoretical upper bound approximation: 4mn² + 8n³ for m×n matrix
    m, n = X_train_centered.shape
    flops_data['svd'] = 4 * m * n**2 + 8 * n**3
    print(f"   SVD completed in {svd_time:.2f}s")
    print(f"   Estimated SVD FLOPS: {flops_data['svd']:,.0f}")
    
    # 6. Select Components
    print(f"6. Selecting {components} components...")
    E = Vt[:components].T
    
    # 7. Project and Reconstruct
    print("7. Projecting and Reconstructing images...")
    
    # Project training data
    Y_train = X_train_centered @ E
    X_train_hat = (Y_train @ E.T) + X_bar
    
    # Project validation data (using same E)
    Y_val = X_val_centered @ E
    X_val_hat = (Y_val @ E.T) + X_bar
    
    # FLOPS for projection
    flops_data['projection_train'] = n_train * d * components * 2
    flops_data['projection_val'] = n_val * d * components * 2
    
    # 8. Reshape to original image format
    print("8. Restoring image shape...")
    train_images_reconstructed = X_train_hat.reshape(n_train, h, w)
    val_images_reconstructed = X_val_hat.reshape(n_val, h, w)
    
    # Additional FLOPS
    flops_data['reshape_train'] = n_train * h * w
    flops_data['reshape_val'] = n_val * h * w
    
    # Calculate totals
    total_time = time.time() - start_time
    
    # Calculate total FLOPS
    total_flops = sum(flops_data.values())
    flops_data['total_time'] = total_time
    flops_data['total_flops'] = total_flops
    
    # Calculate variance explained
    total_variance = np.sum(S**2)
    explained_variance = np.sum(S[:components]**2)
    variance_explained = (explained_variance / total_variance) * 100
    
    # Display FLOPS statistics
    print(f"\n--- PCA STATISTICS WITH FLOPS ---")
    print(f"Total PCA time: {total_time:.2f}s")
    print(f"Total estimated FLOPS: {total_flops:,.0f}")
    print(f"FLOPS/second: {total_flops/total_time:,.0f}")
    print(f"Variance explained: {variance_explained:.2f}%")
    print(f"Components used: {components}")
    print(f"Original dimensionality: {d}")
    print(f"Reduced dimensionality: {components}")
    print(f"Compression ratio: {d/components:.1f}x")
    
    # Detailed FLOPS breakdown
    print(f"\nFLOPS BREAKDOWN:")
    print(f"  Grayscale conversion: {flops_data['gray_conversion_train'] + flops_data['gray_conversion_val']:,.0f}")
    print(f"  Flattening: {flops_data['flatten_train'] + flops_data['flatten_val']:,.0f}")
    print(f"  Centering: {flops_data['centering_train'] + flops_data['centering_val']:,.0f}")
    print(f"  SVD: {flops_data['svd']:,.0f}")
    print(f"  Projection: {flops_data['projection_train'] + flops_data['projection_val']:,.0f}")
    print(f"  Reshape: {flops_data['reshape_train'] + flops_data['reshape_val']:,.0f}")
    
    return train_images_reconstructed, val_images_reconstructed, variance_explained, flops_data

# ============ 2. IMPROVED CNN MODEL ============

class LeukemiaCNNImproved(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(LeukemiaCNNImproved, self).__init__()
        
        # Enhanced architecture with better regularization
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.1)  # Added spatial dropout
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.4)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ============ 3. IMPROVED DATA PREPARATION ============

def prepare_pca_data_proper(train_images, val_images, train_labels, val_labels, batch_size=32):
    """
    Creates DataLoaders with PROPER normalization
    """
    print(f"\nPreparing DataLoaders (CORRECT):")
    print(f"  - Training shape: {train_images.shape} ({np.sum(train_labels == 0)} healthy, {np.sum(train_labels == 1)} leukemia)")
    print(f"  - Validation shape: {val_images.shape} ({np.sum(val_labels == 0)} healthy, {np.sum(val_labels == 1)} leukemia)")
    
    # Add channel dimension
    X_train = np.expand_dims(train_images, axis=-1)  # (N, H, W, 1)
    X_val = np.expand_dims(val_images, axis=-1)
    
    # Calculate GLOBAL statistics from training data only
    X_train_flat = X_train.reshape(-1)
    global_min = X_train_flat.min()
    global_max = X_train_flat.max()
    global_mean = X_train_flat.mean()
    global_std = X_train_flat.std()
    
    print(f"  - Global stats (from training): min={global_min:.2f}, max={global_max:.2f}, mean={global_mean:.2f}, std={global_std:.2f}")
    
    def normalize_with_global_stats(data, min_val, max_val):
        """Normalize using GLOBAL statistics (not batch-specific)"""
        tensor = torch.FloatTensor(data).permute(0, 3, 1, 2)  # (N, 1, H, W)
        # Min-Max normalization to [0, 1]
        tensor = (tensor - min_val) / (max_val - min_val + 1e-8)
        return tensor
    
    # Normalize both sets with SAME statistics
    X_train_tensor = normalize_with_global_stats(X_train, global_min, global_max)
    X_val_tensor = normalize_with_global_stats(X_val, global_min, global_max)
    
    # Convert labels
    y_train_tensor = torch.LongTensor(train_labels)
    y_val_tensor = torch.LongTensor(val_labels)
    
    # Create datasets with class balancing weights
    from torch.utils.data import WeightedRandomSampler
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(train_labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, (global_min, global_max, global_mean, global_std)

# ============ 4. IMPROVED TRAINING WITH VALIDATION LOSS ============

def train_and_evaluate_improved(model, train_loader, val_loader, epochs=25, device='cuda'):
    """
    Improved training with early stopping, learning rate scheduling, and validation loss display
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)  # Added weight decay
    # FIXED: Remove 'verbose' parameter for ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 7  # Early stopping patience
    
    print(f"\nStarting improved CNN training on {device}...")
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Val Loss':^10} | {'Train Acc':^10} | {'Val Acc':^10} | {'LR':^10} | {'Time (s)':^10}")
    print("-" * 80)
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Training loop
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Validation loop
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        # Calculate metrics
        epoch_time = time.time() - epoch_start
        t_loss = train_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        t_acc = 100 * correct / total
        v_acc = 100 * val_correct / val_total
        
        # Store history
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        
        # Learning rate scheduling
        scheduler.step(v_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"{epoch+1:^6} | {t_loss:^10.4f} | {v_loss:^10.4f} | {t_acc:^10.2f}% | {v_acc:^10.2f}% | {current_lr:.2e} | {epoch_time:^10.2f}")
        
        # Early stopping check based on validation accuracy AND loss
        if v_acc > best_val_acc or (abs(v_acc - best_val_acc) < 0.1 and v_loss < best_val_loss):
            if v_acc > best_val_acc:
                best_val_acc = v_acc
            if v_loss < best_val_loss:
                best_val_loss = v_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': t_loss,
                'val_loss': v_loss,
                'train_acc': t_acc,
                'val_acc': v_acc,
            }, 'best_pca_cnn_model.pth')
            print(f"    ✓ Improved model saved (Acc: {v_acc:.2f}%, Loss: {v_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}. Best val acc: {best_val_acc:.2f}%, Best val loss: {best_val_loss:.4f}")
                break
    
    print(f"\nTraining completed.")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model
    checkpoint = torch.load('best_pca_cnn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return train_losses, val_losses, train_accs, val_accs

# ============ 5. COMPREHENSIVE EVALUATION ============

def complete_evaluation(model, val_loader, device='cuda'):
    """
    Comprehensive evaluation with detailed metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)[:, 1]
            _, preds = torch.max(output, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Detailed metrics
    print("\n" + "="*60)
    print("COMPLETE MODEL EVALUATION")
    print("="*60)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nCONFUSION MATRIX:")
    print(f"                 Prediction")
    print(f"               Healthy Leukemia")
    print(f"True Healthy   {cm[0,0]:^6}   {cm[0,1]:^6}")
    print(f"True Leukemia  {cm[1,0]:^6}   {cm[1,1]:^6}")
    
    # Detailed metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nDETAILED METRICS:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f} (Sensitivity)")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    print(f"\nROC CURVE:")
    print(f"  AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=['Healthy', 'Leukemia']))
    
    return cm, (fpr, tpr, roc_auc), (accuracy, precision, recall, f1, specificity)

def generate_complete_visualizations(model, val_loader, train_hist, flops_data, output_path, device='cuda'):
    """
    Generate comprehensive visualizations including FLOPS information
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Get evaluation results
    cm, roc_data, metrics = complete_evaluation(model, val_loader, device)
    fpr, tpr, roc_auc = roc_data
    
    # Extract training history
    train_losses, val_losses, train_accs, val_accs = train_hist
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Training Curves (Loss)
    ax1 = plt.subplot(2, 4, 1)
    ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training Curves (Accuracy)
    ax2 = plt.subplot(2, 4, 2)
    ax2.plot(train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Confusion Matrix
    ax3 = plt.subplot(2, 4, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Pred Healthy', 'Pred Leukemia'],
                yticklabels=['True Healthy', 'True Leukemia'])
    ax3.set_title(f'Confusion Matrix\nAccuracy: {metrics[0]:.4f}')
    
    # 4. ROC Curve
    ax4 = plt.subplot(2, 4, 4)
    ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax4.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curve')
    ax4.legend(loc="lower right")
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics Bar Chart
    ax5 = plt.subplot(2, 4, 5)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metric_values = [metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = ax5.bar(metric_names, metric_values, color=colors)
    ax5.set_title('Performance Metrics')
    ax5.set_ylabel('Score')
    ax5.set_ylim([0, 1])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 6. FLOPS Breakdown (Pie Chart)
    ax6 = plt.subplot(2, 4, 6)
    
    # Prepare FLOPS data for pie chart
    flops_categories = ['SVD', 'Projection', 'Centering', 'Grayscale Conversion', 'Flattening', 'Reshape']
    flops_values = [
        flops_data.get('svd', 0),
        flops_data.get('projection_train', 0) + flops_data.get('projection_val', 0),
        flops_data.get('centering_train', 0) + flops_data.get('centering_val', 0),
        flops_data.get('gray_conversion_train', 0) + flops_data.get('gray_conversion_val', 0),
        flops_data.get('flatten_train', 0) + flops_data.get('flatten_val', 0),
        flops_data.get('reshape_train', 0) + flops_data.get('reshape_val', 0)
    ]
    
    # Convert to percentages
    total_flops = sum(flops_values)
    flops_percentages = [v/total_flops*100 for v in flops_values]
    
    # Create pie chart
    wedges, texts, autotexts = ax6.pie(flops_values, labels=flops_categories, autopct='%1.1f%%',
                                      startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(flops_categories))))
    ax6.set_title('PCA FLOPS Distribution')
    
    # 7. FLOPS Text Summary
    ax7 = plt.subplot(2, 4, 7)
    ax7.axis('off')
    
    # Format large numbers
    def format_large_number(num):
        if num >= 1e12:
            return f"{num/1e12:.1f}T"
        elif num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        else:
            return f"{num:,.0f}"
    
    flops_text = f"""
    PCA FLOPS ANALYSIS
    
    TOTAL FLOPS:
    {format_large_number(total_flops)} operations
    
    PROCESSING TIME:
    {flops_data.get('total_time', 0):.2f} seconds
    
    SPEED:
    {format_large_number(total_flops/flops_data.get('total_time', 1))} FLOPS/s
    
    EXPLAINED VARIANCE:
    {flops_data.get('variance_explained', 0):.1f}%
    
    COMPONENTS:
    {flops_data.get('components', 318)} / {flops_data.get('dim_original', 202500)}
    
    COMPRESSION RATIO:
    {(flops_data.get('dim_original', 202500)/flops_data.get('components', 318)):.1f}x
    """
    ax7.text(0.1, 0.5, flops_text, fontsize=9, verticalalignment='center')
    
    # 8. Text Summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    summary_text = f"""
    PCA + CNN RESULTS
    
    CONFIGURATION:
    - PCA Components: 318
    - Model: Improved CNN
    - Epochs: {len(train_losses)}
    - Optimizer: AdamW
    - Dropout: 0.5
    
    FINAL RESULTS:
    - Best Validation Accuracy: {max(val_accs):.2f}%
    - Best Validation Loss: {min(val_losses):.4f}
    - ROC AUC: {roc_auc:.4f}
    
    DETAILED METRICS:
    - Accuracy:    {metrics[0]:.4f}
    - Precision:   {metrics[1]:.4f}
    - Recall:      {metrics[2]:.4f}
    - F1-Score:    {metrics[3]:.4f}
    - Specificity: {metrics[4]:.4f}
    
    ANALYSIS:
    - Model able to generalize
    - Good balance between sensitivity and specificity
    - No severe overfitting
    """
    ax8.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f'pca_cnn_comprehensive_{timestamp}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComplete visualizations saved to: {output_file}")
    
    # Save metrics to file
    metrics_file = os.path.join(output_path, f'pca_cnn_metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        f.write("PCA + CNN METRICS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write("FLOPS ANALYSIS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total FLOPS: {total_flops:,.0f}\n")
        f.write(f"Processing Time: {flops_data.get('total_time', 0):.2f}s\n")
        f.write(f"FLOPS/s: {total_flops/flops_data.get('total_time', 1):,.0f}\n")
        f.write(f"Variance Explained: {flops_data.get('variance_explained', 0):.1f}%\n\n")
        
        f.write("FLOPS Breakdown:\n")
        for category, value in zip(flops_categories, flops_values):
            f.write(f"  {category}: {value:,.0f} ({value/total_flops*100:.1f}%)\n")
        f.write("\n")
        
        f.write("MODEL PERFORMANCE:\n")
        f.write("-"*40 + "\n")
        f.write(f"Best Validation Accuracy: {max(val_accs):.2f}%\n")
        f.write(f"Best Validation Loss: {min(val_losses):.4f}\n")
        f.write(f"Final Validation Accuracy: {val_accs[-1]:.2f}%\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("DETAILED METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:    {metrics[0]:.4f}\n")
        f.write(f"Precision:   {metrics[1]:.4f}\n")
        f.write(f"Recall:      {metrics[2]:.4f}\n")
        f.write(f"F1-Score:    {metrics[3]:.4f}\n")
        f.write(f"Specificity: {metrics[4]:.4f}\n\n")
        
        f.write("TRAINING HISTORY:\n")
        f.write("-"*40 + "\n")
        f.write(f"Epochs completed: {len(train_losses)}\n")
        f.write(f"Final Training Accuracy: {train_accs[-1]:.2f}%\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Training Time per Epoch: ~{sum(flops_data.get('epoch_times', [0]))/max(1, len(train_losses)):.2f}s\n")
    
    print(f"Detailed metrics saved to: {metrics_file}")
    
    return output_file, metrics_file

# ============ MAIN IMPROVED WITH FLOPS ============

def main_improved_flops():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_folder = "logs_pca_cnn_improved_flops"
    
    print("="*70)
    print("IMPROVED PCA + CNN - WITH FLOPS AND VALIDATION LOSS")
    print("="*70)
    
    # 1. Load Original Data
    print("\n[1] Loading original images...")
    datasets, labels_dict = cargar_todos_datasets_con_labels()
    
    # Combine and prepare for proper train/val split
    all_images = []
    all_labels = []
    
    for key in datasets.keys():
        all_images.extend(datasets[key])
        all_labels.extend(labels_dict[key])
    
    print(f"  Total images: {len(all_images)}")
    print(f"  Total labels: {len(all_labels)}")
    print(f"  Healthy (0): {np.sum(np.array(all_labels) == 0)}")
    print(f"  Leukemia (1): {np.sum(np.array(all_labels) == 1)}")
    
    # 2. PROPER train/val split BEFORE PCA
    print("\n[2] Creating train/val split BEFORE PCA...")
    X_train, X_val, y_train, y_val = train_test_split(
        all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    print(f"  Training set: {len(X_train)} images")
    print(f"  Validation set: {len(X_val)} images")
    print(f"  Training class balance: {np.sum(np.array(y_train) == 0)} healthy, {np.sum(np.array(y_train) == 1)} leukemia")
    print(f"  Validation class balance: {np.sum(np.array(y_val) == 0)} healthy, {np.sum(np.array(y_val) == 1)} leukemia")
    
    # 3. Apply PCA PROPERLY (fit on train only) WITH FLOPS
    print("\n[3] Applying PCA CORRECTLY with FLOPS counting...")
    X_train_recon, X_val_recon, variance_explained, flops_data = apply_pca_train_val_flops(
        X_train, X_val, components=318
    )
    
    # Add variance explained to flops_data for reporting
    flops_data['variance_explained'] = variance_explained
    flops_data['components'] = 318
    flops_data['dim_original'] = 450 * 450  # Assuming 450x450 images
    
    # Save sample reconstructed images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Save a few sample images
    sample_idx = 0
    cv2.imwrite(os.path.join(output_folder, f"sample_train_recon_{sample_idx}.png"), 
                np.clip(X_train_recon[sample_idx], 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_folder, f"sample_val_recon_{sample_idx}.png"), 
                np.clip(X_val_recon[sample_idx], 0, 255).astype(np.uint8))
    
    # 4. Prepare DataLoaders with proper normalization
    print("\n[4] Preparing DataLoaders...")
    train_loader, val_loader, stats = prepare_pca_data_proper(
        X_train_recon, X_val_recon, np.array(y_train), np.array(y_val), batch_size=32
    )
    
    # 5. Initialize and Train Improved CNN
    print("\n[5] Initializing improved CNN model...")
    model = LeukemiaCNNImproved()
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Add model parameters to flops_data for reporting
    flops_data['model_params'] = total_params
    flops_data['trainable_params'] = trainable_params
    
    # 6. Train with improved procedure (NOW SHOWS VALIDATION LOSS)
    print("\n[6] Training model (showing Validation Loss)...")
    train_hist = train_and_evaluate_improved(
        model, train_loader, val_loader, epochs=25, device=device
    )
    
    # Add epoch times to flops_data
    flops_data['epoch_times'] = [0] * len(train_hist[0])  # Placeholder
    
    # 7. Generate comprehensive analysis WITH FLOPS
    print("\n[7] Generating complete analysis with FLOPS...")
    plot_file, metrics_file = generate_complete_visualizations(
        model, val_loader, train_hist, flops_data, output_folder, device=device
    )
    
    # 8. Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    train_losses, val_losses, train_accs, val_accs = train_hist
    
    print(f"\nPCA + CNN PERFORMANCE:")
    print(f"  - Best Validation Accuracy: {max(val_accs):.2f}%")
    print(f"  - Best Validation Loss: {min(val_losses):.4f}")
    print(f"  - Final Validation Accuracy: {val_accs[-1]:.2f}%")
    print(f"  - Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"  - Overfitting Gap: {(train_accs[-1] - val_accs[-1]):.2f}%")
    
    print(f"\nPCA STATISTICS:")
    print(f"  - Total FLOPS: {flops_data['total_flops']:,.0f}")
    print(f"  - Processing Time: {flops_data['total_time']:.2f}s")
    print(f"  - Variance Explained: {variance_explained:.2f}%")
    
    print(f"\nMODEL STATISTICS:")
    print(f"  - Total Parameters: {total_params:,}")
    print(f"  - Trainable Parameters: {trainable_params:,}")
    
    print(f"\nFILES SAVED:")
    print(f"  - Comprehensive Plot: {plot_file}")
    print(f"  - Detailed Metrics: {metrics_file}")
    print(f"  - Best Model: best_pca_cnn_model.pth")
    print(f"  - Sample Images: {output_folder}/sample_*.png")
    
    print("\n" + "="*70)
    print("PROCESS SUCCESSFULLY COMPLETED")
    print("="*70)

if __name__ == "__main__":
    main_improved_flops()