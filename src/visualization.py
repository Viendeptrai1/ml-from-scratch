"""
Utilities để tạo visualizations cho phân tích dữ liệu ảnh và machine learning
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import cv2
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_palette("husl")

class ImageDataVisualizer:
    """Class để tạo visualizations cho dữ liệu ảnh"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize ImageDataVisualizer
        
        Args:
            figsize: Kích thước mặc định cho figures
        """
        self.figsize = figsize
        
    def plot_sample_images(self, X: np.ndarray, y: np.ndarray, 
                          class_names: List[str], n_samples: int = 3):
        """
        Hiển thị mẫu ảnh từ mỗi lớp
        
        Args:
            X: Mảng ảnh
            y: Mảng labels
            class_names: Tên các lớp
            n_samples: Số mẫu mỗi lớp
        """
        n_classes = len(class_names)
        fig, axes = plt.subplots(n_classes, n_samples, 
                                figsize=(4*n_samples, 4*n_classes))
        fig.suptitle('Mẫu Ảnh Từ Mỗi Lớp', fontsize=16, fontweight='bold')
        
        for i, class_name in enumerate(class_names):
            # Lấy indices của lớp hiện tại
            class_indices = np.where(y == i)[0]
            
            # Lấy n_samples ảnh ngẫu nhiên
            if len(class_indices) >= n_samples:
                selected_indices = np.random.choice(class_indices, n_samples, replace=False)
            else:
                selected_indices = class_indices
            
            for j, idx in enumerate(selected_indices):
                if j >= n_samples:
                    break
                    
                img = X[idx]
                
                # Chuẩn hóa ảnh để hiển thị
                if img.max() <= 1.0:
                    img_display = img
                else:
                    img_display = img / 255.0
                
                if n_classes == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                ax.imshow(img_display)
                ax.set_title(f'{class_name}\nSample {j+1}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_data_distribution(self, df: pd.DataFrame):
        """
        Vẽ biểu đồ phân bố dữ liệu ảnh
        
        Args:
            df: DataFrame chứa thông tin ảnh
        """
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        fig.suptitle('Phân Tích Phân Bố Dữ Liệu Ảnh', fontsize=16, fontweight='bold')
        
        # 1. Phân bố số lượng theo lớp
        class_counts = df['class'].value_counts()
        axes[0, 0].bar(class_counts.index, class_counts.values, 
                      color=sns.color_palette("husl", len(class_counts)))
        axes[0, 0].set_title('Phân Bố Số Lượng Theo Lớp')
        axes[0, 0].set_xlabel('Lớp')
        axes[0, 0].set_ylabel('Số Lượng')
        for i, v in enumerate(class_counts.values):
            axes[0, 0].text(i, v + 0.05, str(v), ha='center', va='bottom')
        
        # 2. Phân bố kích thước
        axes[0, 1].scatter(df['width'], df['height'], 
                          c=[sns.color_palette("husl", len(df['class'].unique()))[
                              list(df['class'].unique()).index(x)] for x in df['class']],
                          alpha=0.7)
        axes[0, 1].set_title('Phân Bố Kích Thước (Width vs Height)')
        axes[0, 1].set_xlabel('Width (pixels)')
        axes[0, 1].set_ylabel('Height (pixels)')
        
        # 3. Histogram tỷ lệ khung hình
        axes[0, 2].hist(df['aspect_ratio'], bins=15, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[0, 2].set_title('Phân Bố Tỷ Lệ Khung Hình')
        axes[0, 2].set_xlabel('Aspect Ratio')
        axes[0, 2].set_ylabel('Frequency')
        
        # 4. Phân bố dung lượng file
        axes[1, 0].hist(df['file_size_kb'], bins=15, alpha=0.7, 
                       color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Phân Bố Dung Lượng File')
        axes[1, 0].set_xlabel('File Size (KB)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Boxplot dung lượng theo lớp
        sns.boxplot(data=df, x='class', y='file_size_kb', ax=axes[1, 1])
        axes[1, 1].set_title('Dung Lượng File Theo Lớp')
        axes[1, 1].set_xlabel('Lớp')
        axes[1, 1].set_ylabel('File Size (KB)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Định dạng file
        format_counts = df['format'].value_counts() if 'format' in df.columns else df['extension'].value_counts()
        axes[1, 2].pie(format_counts.values, labels=format_counts.index, 
                      autopct='%1.1f%%', colors=sns.color_palette("Set3"))
        axes[1, 2].set_title('Phân Bố Định Dạng File')
        
        plt.tight_layout()
        plt.show()
    
    def plot_color_analysis(self, images: List[np.ndarray], 
                           class_names: List[str], titles: List[str] = None):
        """
        Phân tích và vẽ histogram màu sắc
        
        Args:
            images: List các ảnh để phân tích
            class_names: Tên các lớp
            titles: Tiêu đề cho từng ảnh
        """
        n_images = len(images)
        fig, axes = plt.subplots(n_images, 2, figsize=(12, 4*n_images))
        fig.suptitle('Phân Tích Màu Sắc', fontsize=16, fontweight='bold')
        
        colors = ['red', 'green', 'blue']
        
        for i, img in enumerate(images):
            # Hiển thị ảnh gốc
            if n_images == 1:
                ax_img, ax_hist = axes[0], axes[1]
            else:
                ax_img, ax_hist = axes[i, 0], axes[i, 1]
            
            # Chuẩn hóa ảnh để hiển thị
            if img.max() <= 1.0:
                img_display = img
            else:
                img_display = img / 255.0
            
            ax_img.imshow(img_display)
            title = titles[i] if titles else f'{class_names[i] if i < len(class_names) else "Image"}'
            ax_img.set_title(title)
            ax_img.axis('off')
            
            # Tính và vẽ histogram
            img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            for j, color in enumerate(colors):
                hist = cv2.calcHist([img_uint8], [j], None, [256], [0, 256])
                ax_hist.plot(hist, color=color, alpha=0.7, label=f'{color.capitalize()}')
            
            ax_hist.set_title(f'Histogram Màu Sắc - {title}')
            ax_hist.set_xlabel('Intensity')
            ax_hist.set_ylabel('Frequency')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_augmentation_examples(self, original_img: np.ndarray, 
                                  augmented_imgs: List[np.ndarray],
                                  titles: List[str]):
        """
        Hiển thị ví dụ về data augmentation
        
        Args:
            original_img: Ảnh gốc
            augmented_imgs: List ảnh đã augment
            titles: Tiêu đề cho từng ảnh
        """
        n_imgs = len(augmented_imgs) + 1
        cols = min(4, n_imgs)
        rows = (n_imgs + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        fig.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Hiển thị ảnh gốc
        img_display = original_img if original_img.max() <= 1.0 else original_img / 255.0
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Hiển thị ảnh đã augment
        for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            img_display = aug_img if aug_img.max() <= 1.0 else aug_img / 255.0
            axes[row, col].imshow(img_display)
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        # Ẩn các subplot không sử dụng
        for i in range(n_imgs, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()

class ModelEvaluationVisualizer:
    """Class để visualize kết quả đánh giá model"""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                             class_names: List[str], normalize: bool = False):
        """
        Vẽ confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Tên các lớp
            normalize: Có normalize không
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_training_history(history: Dict, metrics: List[str] = ['loss', 'accuracy']):
        """
        Vẽ biểu đồ training history
        
        Args:
            history: Dictionary chứa training history
            metrics: List các metrics cần vẽ
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in history:
                axes[i].plot(history[metric], label=f'Training {metric}')
                if f'val_{metric}' in history:
                    axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
                
                axes[i].set_title(f'{metric.capitalize()} Over Epochs')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                                  class_names: List[str]):
        """
        Hiển thị classification report dạng heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Tên các lớp
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # Tạo DataFrame cho heatmap
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=class_names)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics_df.T, annot=True, fmt='.3f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Score'})
        plt.title('Classification Metrics by Class')
        plt.xlabel('Classes')
        plt.ylabel('Metrics')
        plt.tight_layout()
        plt.show()
        
        # In text report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

def plot_feature_visualization(X: np.ndarray, y: np.ndarray, 
                              class_names: List[str], method: str = 'tsne'):
    """
    Visualize features sử dụng dimensionality reduction
    
    Args:
        X: Feature matrix
        y: Labels
        class_names: Tên các lớp
        method: Phương pháp giảm chiều ('tsne', 'pca')
    """
    # Flatten images nếu cần
    if len(X.shape) > 2:
        X_flat = X.reshape(X.shape[0], -1)
    else:
        X_flat = X
    
    # Áp dụng dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization'
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Visualization'
    else:
        raise ValueError("Method phải là 'tsne' hoặc 'pca'")
    
    print(f"Applying {method.upper()}...")
    X_reduced = reducer.fit_transform(X_flat)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("husl", len(class_names))
    
    for i, class_name in enumerate(class_names):
        mask = y == i
        plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], 
                   c=[colors[i]], label=class_name, alpha=0.7, s=60)
    
    plt.title(f'{title} of Image Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def save_all_plots(output_dir: str = '../outputs/figures'):
    """
    Lưu tất cả các plots đã tạo
    
    Args:
        output_dir: Thư mục output
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu current figure
    plt.savefig(os.path.join(output_dir, 'current_plot.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_dir}") 