"""
Đánh giá model và tính toán các metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.utils import to_categorical
import os


def evaluate_classification_model(model, X_test, y_test, class_names=None):
    """
    Đánh giá model classification
    
    Args:
        model: Model đã train
        X_test: Test features
        y_test: Test labels
        class_names: Tên các class
    
    Returns:
        dict: Các metrics đánh giá
    """
    # Predictions
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
    else:
        # Keras model
        y_prob = model.predict(X_test)
        if y_prob.shape[1] == 1:  # Binary classification
            y_pred = (y_prob > 0.5).astype(int).flatten()
            y_prob = np.column_stack([1-y_prob.flatten(), y_prob.flatten()])
        else:  # Multi-class
            y_pred = np.argmax(y_prob, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Multi-class vs binary classification
    if len(np.unique(y_test)) > 2:
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC for multi-class
        if y_prob.shape[1] > 2:
            y_test_binary = to_categorical(y_test, num_classes=y_prob.shape[1])
            roc_auc = roc_auc_score(y_test_binary, y_prob, average='weighted', multi_class='ovr')
        else:
            roc_auc = None
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC AUC for binary classification
        if y_prob.shape[1] == 2:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            roc_auc = roc_auc_score(y_test, y_prob.flatten())
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification Report
    if class_names is not None:
        target_names = class_names
    else:
        target_names = [f'Class_{i}' for i in range(len(np.unique(y_test)))]
    
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'probabilities': y_prob
    }
    
    # Print results
    print("=== KẾT QUẢ ĐÁNH GIÁ MODEL ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return results


def plot_confusion_matrix(cm, class_names=None, title='Confusion Matrix', save_path=None):
    """
    Vẽ confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: Tên các class
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu confusion matrix vào {save_path}")
    
    plt.show()


def plot_roc_curve(y_test, y_prob, title='ROC Curve', save_path=None):
    """
    Vẽ ROC curve
    
    Args:
        y_test: True labels
        y_prob: Predicted probabilities
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(8, 6))
    
    # Binary classification
    if len(np.unique(y_test)) == 2:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob.flatten()
        
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = roc_auc_score(y_test, y_score)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    
    # Multi-class classification
    else:
        y_test_binary = to_categorical(y_test, num_classes=y_prob.shape[1])
        
        for i in range(y_prob.shape[1]):
            fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_prob[:, i])
            roc_auc = roc_auc_score(y_test_binary[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, lw=2, 
                    label=f'Class {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu ROC curve vào {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_test, y_prob, title='Precision-Recall Curve', save_path=None):
    """
    Vẽ Precision-Recall curve
    
    Args:
        y_test: True labels
        y_prob: Predicted probabilities
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(8, 6))
    
    # Binary classification
    if len(np.unique(y_test)) == 2:
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_score = y_prob[:, 1]
        else:
            y_score = y_prob.flatten()
        
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        avg_precision = average_precision_score(y_test, y_score)
        
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
    
    # Multi-class classification
    else:
        y_test_binary = to_categorical(y_test, num_classes=y_prob.shape[1])
        
        for i in range(y_prob.shape[1]):
            precision, recall, _ = precision_recall_curve(y_test_binary[:, i], y_prob[:, i])
            avg_precision = average_precision_score(y_test_binary[:, i], y_prob[:, i])
            plt.plot(recall, precision, lw=2,
                    label=f'Class {i} (AP = {avg_precision:.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu PR curve vào {save_path}")
    
    plt.show()


def evaluate_regression_model(model, X_test, y_test):
    """
    Đánh giá model regression
    
    Args:
        model: Model đã train
        X_test: Test features
        y_test: Test labels
    
    Returns:
        dict: Các metrics đánh giá
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'predictions': y_pred
    }
    
    # Print results
    print("=== KẾT QUẢ ĐÁNH GIÁ MODEL REGRESSION ===")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return results


def generate_evaluation_report(model, X_test, y_test, class_names=None, 
                             model_name="Model", save_dir="outputs/figures"):
    """
    Tạo báo cáo đánh giá hoàn chỉnh
    
    Args:
        model: Model đã train
        X_test: Test features
        y_test: Test labels
        class_names: Tên các class
        model_name: Tên model
        save_dir: Thư mục lưu hình
    
    Returns:
        dict: Kết quả đánh giá
    """
    # Evaluate model
    results = evaluate_classification_model(model, X_test, y_test, class_names)
    
    # Plot confusion matrix
    cm_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plot_confusion_matrix(results['confusion_matrix'], class_names, 
                         f'{model_name} - Confusion Matrix', cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(save_dir, f"{model_name}_roc_curve.png")
    plot_roc_curve(y_test, results['probabilities'], 
                  f'{model_name} - ROC Curve', roc_path)
    
    # Plot PR curve
    pr_path = os.path.join(save_dir, f"{model_name}_pr_curve.png")
    plot_precision_recall_curve(y_test, results['probabilities'],
                               f'{model_name} - Precision-Recall Curve', pr_path)
    
    print(f"\nĐã tạo báo cáo đánh giá cho {model_name}")
    print(f"Các hình ảnh được lưu trong thư mục: {save_dir}")
    
    return results


def compare_models_performance(models_results, metric='accuracy'):
    """
    So sánh performance của nhiều model
    
    Args:
        models_results: Dict chứa kết quả của các model
        metric: Metric để so sánh
    
    Returns:
        pd.DataFrame: Bảng so sánh
    """
    comparison_data = []
    
    for model_name, results in models_results.items():
        if metric in results:
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'ROC AUC': results.get('roc_auc', 0) if results.get('roc_auc') else 0
            })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values(metric.title(), ascending=False)
    
    print(f"\n=== SO SÁNH PERFORMANCE THEO {metric.upper()} ===")
    print(df.to_string(index=False, float_format='%.4f'))
    
    return df 