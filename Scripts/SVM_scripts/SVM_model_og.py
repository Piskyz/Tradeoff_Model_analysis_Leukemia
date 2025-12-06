"""
SVM Classifier for Leukemia Detection (Classical ML Approach)
Pipeline: Resize -> Flatten -> Scaler -> PCA -> SVM
Input: Images (Resized to 128x128 for efficiency)
Output: Binary Classification (Healthy vs Leukemia)
"""

import os
import time
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ML Libraries
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# Data Loading Imports
try:
    from Creador_labels import cargar_todos_datasets_con_labels
except ImportError:
    print("Warning: Creador_labels not found. Using fallback.")

from Carga_imagenes import cargar_training_all_original, cargar_training_hem_original


def process_and_flatten_images(images, target_size=(128, 128)):
    """
    Resizes and flattens images for Classical ML.
    Original (450x450) -> Resized (128x128) -> Flattened (16384 features)
    """
    processed_data = []
    
    for img in images:
        # 1. Convert to Grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
            
        # 2. Resize (SVMs are slow with huge feature vectors)
        resized = cv2.resize(gray, target_size)
        
        # 3. Flatten (2D Matrix -> 1D Vector)
        flattened = resized.flatten()
        processed_data.append(flattened)
        
    return np.array(processed_data)

def prepare_data_svm(datasets):
    """
    Combines all data into a standard X (features) and y (labels) format.
    """
    print("Preparando datos para SVM...")
    
    # 1. Extract Images
    hem_images = [] # Healthy (0)
    all_images = [] # Leukemia (1)
    
    for key in datasets:
        if 'hem' in key:
            hem_images.extend(datasets[key])
        elif 'all' in key:
            all_images.extend(datasets[key])
            
    print(f"  - Imágenes Sanas (Healthy): {len(hem_images)}")
    print(f"  - Imágenes Leucemia (ALL): {len(all_images)}")
    
    # 2. Process (Resize & Flatten)
    print("  - Procesando imágenes (Resize 128x128 + Flatten)...")
    X_hem = process_and_flatten_images(hem_images)
    X_all = process_and_flatten_images(all_images)
    
    # 3. Create Labels
    y_hem = np.zeros(len(X_hem)) # Label 0
    y_all = np.ones(len(X_all))  # Label 1
    
    # 4. Concatenate
    X = np.concatenate([X_hem, X_all], axis=0)
    y = np.concatenate([y_hem, y_all], axis=0)
    
    return X, y

def train_svm_pipeline(X_train, y_train):
    """
    Creates and trains the ML Pipeline: Scaler -> PCA -> SVM
    """
    print("\nIniciando entrenamiento del Pipeline SVM...")
    print("  1. StandardScaler: Normalizando datos...")
    print("  2. PCA: Reduciendo dimensionalidad (95% varianza)...")
    print("  3. SVM: Entrenando clasificador (Kernel RBF)...")
    
    # Define the Pipeline
    # PCA n_components=0.95 means "Keep enough components to explain 95% of variance"
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)), 
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
    
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    # Check how many PCA components were kept
    n_components = pipeline.named_steps['pca'].n_components_
    print(f"\nEntrenamiento completado en {training_time:.2f} segundos.")
    print(f"PCA redujo las features de {X_train.shape[1]} a {n_components} componentes.")
    
    return pipeline, training_time

def evaluate_model(model, X_test, y_test, training_time, output_path="logs_svm"):
    """
    Evaluates the model and generates reports/plots compatible with previous models.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print("\nEvaluando modelo en Test Set...")
    start_time = time.time()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilities for ROC
    
    inference_time = time.time() - start_time
    avg_inference_time = (inference_time / len(X_test)) * 1000 # ms
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Healthy', 'Leukemia'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # --- SAVE REPORT ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_path, f"reporte_svm_{timestamp}.txt")
    
    with open(report_file, "w") as f:
        f.write("="*60 + "\n")
        f.write("REPORTE DE RENDIMIENTO - SVM CLASSIFIER\n")
        f.write("="*60 + "\n\n")
        f.write(f"Modelo: Support Vector Machine (RBF Kernel)\n")
        f.write(f"Pipeline: Resize(128x128) -> Scaler -> PCA(0.95) -> SVM\n")
        f.write(f"Fecha: {timestamp}\n\n")
        
        f.write("-" * 60 + "\nTIEMPOS\n" + "-" * 60 + "\n")
        f.write(f"Tiempo de Entrenamiento: {training_time:.2f} s\n")
        f.write(f"Tiempo de Inferencia (Test): {inference_time:.2f} s\n")
        f.write(f"Latencia promedio por imagen: {avg_inference_time:.2f} ms\n\n")
        
        f.write("-" * 60 + "\nMETRICAS\n" + "-" * 60 + "\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"ROC AUC Score: {roc_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"Reporte guardado en: {report_file}")
    
    # --- SAVE ROC PLOT ---
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - SVM')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plot_file = os.path.join(output_path, f"roc_svm_{timestamp}.png")
    plt.savefig(plot_file)
    print(f"Gráfica ROC guardada en: {plot_file}")

    return acc, roc_auc

def main():
    # 1. Load Data
    print("Cargando datasets...")
    try:
        datasets, _ = cargar_todos_datasets_con_labels()
    except:
        print("Using fallback loading...")
        # Adjust paths if necessary (e.g. "../data/...")
        datasets = {
            'fold_0_hem': cargar_training_hem_original("../data/training_data/fold_0/hem/", max_imagenes=200),
            'fold_0_all': cargar_training_all_original("../data/training_data/fold_0/all/", max_imagenes=200)
        }

    # 2. Prepare Data
    X, y = prepare_data_svm(datasets)
    
    # 3. Split Data (Standard 80/20 Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDatos divididos:")
    print(f"  - Train: {len(X_train)}")
    print(f"  - Test:  {len(X_test)}")
    
    # 4. Train
    model_pipeline, training_time = train_svm_pipeline(X_train, y_train)
    
    # 5. Evaluate
    evaluate_model(model_pipeline, X_test, y_test, training_time)
    
    # 6. Save Model
    model_filename = "leukemia_svm_pipeline.pkl"
    joblib.dump(model_pipeline, model_filename)
    print(f"\nModelo guardado exitosamente como: {model_filename}")

if __name__ == "__main__":
    main()