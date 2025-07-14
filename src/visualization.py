"""
Visualization cho machine learning: learning curves, plots, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_learning_curve(model, X, y, title="Learning Curve", cv=5, 
                       train_sizes=np.linspace(0.1, 1.0, 10), save_path=None):
    """
    Vẽ learning curve để kiểm tra overfitting/underfitting
    
    Args:
        model: Model cần vẽ learning curve
        X: Features
        y: Labels
        title: Tiêu đề biểu đồ
        cv: Số fold cho cross validation
        train_sizes: Các kích thước training set
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes,
        scoring='accuracy', random_state=42
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu learning curve vào {save_path}")
    
    plt.show()


def plot_validation_curve(model, X, y, param_name, param_range, title="Validation Curve",
                         cv=5, scoring='accuracy', save_path=None):
    """
    Vẽ validation curve để tìm hyperparameter tối ưu
    
    Args:
        model: Model cần vẽ validation curve
        X: Features
        y: Labels
        param_name: Tên parameter cần tune
        param_range: Range của parameter
        title: Tiêu đề biểu đồ
        cv: Số fold cho cross validation
        scoring: Metric để đánh giá
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(10, 6))
    
    train_scores, test_scores = validation_curve(
        model, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, train_scores_mean, 'o-', color='r',
             label='Training score')
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color='r')
    
    plt.plot(param_range, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')
    
    plt.xlabel(param_name)
    plt.ylabel(f'{scoring.title()} Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu validation curve vào {save_path}")
    
    plt.show()


def plot_training_history(history, title="Training History", save_path=None):
    """
    Vẽ training history cho Deep Learning model
    
    Args:
        history: Training history từ Keras
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history:
        axes[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        axes[1].plot(history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu training history vào {save_path}")
    
    plt.show()


def plot_feature_importance(model, feature_names, title="Feature Importance", save_path=None):
    """
    Vẽ feature importance cho tree-based models
    
    Args:
        model: Model đã train (tree-based)
        feature_names: Tên các features
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model không hỗ trợ feature importance")
        return
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(len(importance)), importance[indices])
    plt.yticks(range(len(importance)), [feature_names[i] for i in indices])
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu feature importance vào {save_path}")
    
    plt.show()


def plot_model_comparison(results_df, metric='CV_Mean', title="Model Comparison", save_path=None):
    """
    Vẽ biểu đồ so sánh các model
    
    Args:
        results_df: DataFrame chứa kết quả các model
        metric: Metric để so sánh
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(results_df['Model'], results_df[metric])
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu model comparison vào {save_path}")
    
    plt.show()


def plot_data_distribution(data, columns, title="Data Distribution", save_path=None):
    """
    Vẽ phân phối dữ liệu
    
    Args:
        data: DataFrame chứa dữ liệu
        columns: List các cột cần vẽ
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(columns):
        if i < len(axes):
            if data[col].dtype in ['int64', 'float64']:
                axes[i].hist(data[col], bins=30, alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
            else:
                data[col].value_counts().plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Count of {col}')
                axes[i].tick_params(axis='x', rotation=45)
    
    # Hide unused subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu data distribution vào {save_path}")
    
    plt.show()


def plot_correlation_matrix(data, title="Correlation Matrix", save_path=None):
    """
    Vẽ correlation matrix
    
    Args:
        data: DataFrame chứa dữ liệu
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                center=0, square=True, cmap='coolwarm')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu correlation matrix vào {save_path}")
    
    plt.show()


def create_interactive_plot(data, x_col, y_col, color_col=None, title="Interactive Plot"):
    """
    Tạo biểu đồ tương tác với Plotly
    
    Args:
        data: DataFrame chứa dữ liệu
        x_col: Cột cho trục x
        y_col: Cột cho trục y
        color_col: Cột cho màu sắc
        title: Tiêu đề biểu đồ
    
    Returns:
        plotly figure
    """
    if color_col:
        fig = px.scatter(data, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = px.scatter(data, x=x_col, y=y_col, title=title)
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='closest'
    )
    
    return fig


def plot_class_distribution(y, class_names=None, title="Class Distribution", save_path=None):
    """
    Vẽ phân phối các class
    
    Args:
        y: Labels
        class_names: Tên các class
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu hình
    """
    plt.figure(figsize=(10, 6))
    
    unique, counts = np.unique(y, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    bars = plt.bar(class_names, counts)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu class distribution vào {save_path}")
    
    plt.show()


def save_all_plots(figures_dict, save_dir="outputs/figures"):
    """
    Lưu tất cả biểu đồ vào thư mục
    
    Args:
        figures_dict: Dict chứa tên và figure
        save_dir: Thư mục lưu
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for name, fig in figures_dict.items():
        save_path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu {name} vào {save_path}")
    
    print(f"\nĐã lưu tất cả biểu đồ vào thư mục: {save_dir}") 