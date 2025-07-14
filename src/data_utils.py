"""
Data utilities cho việc xử lý và preprocessing dữ liệu ảnh
"""

import os
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import cv2
from sklearn.model_selection import train_test_split
from pathlib import Path
import glob
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class ImageDataProcessor:
    """Class để xử lý và preprocessing dữ liệu ảnh"""
    
    def __init__(self, data_path: str, target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize ImageDataProcessor
        
        Args:
            data_path: Đường dẫn đến thư mục chứa dữ liệu
            target_size: Kích thước mục tiêu để resize (width, height)
        """
        self.data_path = data_path
        self.target_size = target_size
        self.classes = []
        self.image_paths = []
        self.labels = []
        
    def scan_dataset(self) -> pd.DataFrame:
        """
        Quét dataset và thu thập thông tin về các file ảnh
        
        Returns:
            DataFrame chứa thông tin chi tiết về các ảnh
        """
        image_info = []
        self.classes = [d for d in os.listdir(self.data_path) 
                       if os.path.isdir(os.path.join(self.data_path, d))]
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            image_files = glob.glob(os.path.join(class_path, "*.[jpJP][pnPeNgG]*"))
            
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode
                        
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    
                    image_info.append({
                        'path': img_path,
                        'class': class_name,
                        'class_idx': class_idx,
                        'filename': os.path.basename(img_path),
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height,
                        'file_size_kb': round(file_size, 2),
                        'color_mode': mode
                    })
                    
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Lỗi khi đọc {img_path}: {e}")
        
        return pd.DataFrame(image_info)
    
    def load_and_preprocess_image(self, image_path: str, normalize: bool = True) -> np.ndarray:
        """
        Load và preprocess một ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            normalize: Có normalize pixel values không
            
        Returns:
            Mảng numpy của ảnh đã được preprocess
        """
        try:
            # Load ảnh
            img = Image.open(image_path)
            
            # Convert sang RGB nếu cần
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Convert sang numpy array
            img_array = np.array(img)
            
            # Normalize nếu cần
            if normalize:
                img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"Lỗi khi xử lý {image_path}: {e}")
            return None
    
    def create_dataset(self, test_size: float = 0.2, val_size: float = 0.1, 
                      random_state: int = 42) -> Dict:
        """
        Tạo dataset đã được chia train/val/test
        
        Args:
            test_size: Tỷ lệ test set
            val_size: Tỷ lệ validation set (từ train set)
            random_state: Random seed
            
        Returns:
            Dictionary chứa X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Load tất cả ảnh
        X = []
        y = []
        
        print(f"Loading {len(self.image_paths)} images...")
        for i, img_path in enumerate(self.image_paths):
            img = self.load_and_preprocess_image(img_path)
            if img is not None:
                X.append(img)
                y.append(self.labels[i])
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(self.image_paths)} images")
        
        X = np.array(X)
        y = np.array(y)
        
        # Chia train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Chia train/validation
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
            )
        else:
            X_val, y_val = None, None
        
        dataset = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'classes': self.classes
        }
        
        print(f"\nDataset created:")
        print(f"Train: {len(X_train)} samples")
        if X_val is not None:
            print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        return dataset

class ImageAugmentor:
    """Class để thực hiện data augmentation"""
    
    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Xoay ảnh một góc nhất định"""
        if len(image.shape) == 3:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, matrix, (w, h))
        return image
    
    @staticmethod
    def flip_image(image: np.ndarray, horizontal: bool = True) -> np.ndarray:
        """Lật ảnh theo chiều ngang hoặc dọc"""
        if horizontal:
            return cv2.flip(image, 1)
        else:
            return cv2.flip(image, 0)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Điều chỉnh độ sáng của ảnh"""
        if image.dtype == np.float32:
            # Nếu đã normalize
            return np.clip(image * factor, 0.0, 1.0)
        else:
            # Nếu chưa normalize
            return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """Thêm noise vào ảnh"""
        noise = np.random.normal(0, noise_factor, image.shape)
        if image.dtype == np.float32:
            return np.clip(image + noise, 0.0, 1.0)
        else:
            return np.clip(image + noise * 255, 0, 255).astype(np.uint8)
    
    def augment_dataset(self, X: np.ndarray, y: np.ndarray, 
                       augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset với các kỹ thuật khác nhau
        
        Args:
            X: Mảng ảnh
            y: Mảng labels
            augment_factor: Số lần augment mỗi ảnh
            
        Returns:
            Tuple (X_augmented, y_augmented)
        """
        X_aug = [X]
        y_aug = [y]
        
        for i in range(augment_factor):
            X_new = []
            for img in X:
                # Random augmentation
                aug_img = img.copy()
                
                # Random rotation (-15 to 15 degrees)
                if np.random.random() > 0.5:
                    angle = np.random.uniform(-15, 15)
                    aug_img = self.rotate_image(aug_img, angle)
                
                # Random flip
                if np.random.random() > 0.5:
                    aug_img = self.flip_image(aug_img, horizontal=True)
                
                # Random brightness
                if np.random.random() > 0.5:
                    factor = np.random.uniform(0.8, 1.2)
                    aug_img = self.adjust_brightness(aug_img, factor)
                
                # Random noise
                if np.random.random() > 0.7:
                    aug_img = self.add_noise(aug_img, 0.05)
                
                X_new.append(aug_img)
            
            X_aug.append(np.array(X_new))
            y_aug.append(y)
        
        X_final = np.concatenate(X_aug, axis=0)
        y_final = np.concatenate(y_aug, axis=0)
        
        print(f"Dataset augmented from {len(X)} to {len(X_final)} samples")
        
        return X_final, y_final

def save_processed_data(data: Dict, output_path: str):
    """
    Lưu processed data vào file
    
    Args:
        data: Dictionary chứa dataset
        output_path: Đường dẫn output
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Lưu từng phần của dataset
    for key, value in data.items():
        if value is not None and key != 'classes':
            np.save(os.path.join(output_path, f"{key}.npy"), value)
    
    # Lưu class names
    if 'classes' in data:
        with open(os.path.join(output_path, "classes.txt"), 'w') as f:
            for class_name in data['classes']:
                f.write(f"{class_name}\n")
    
    print(f"Processed data saved to {output_path}")

def load_processed_data(data_path: str) -> Dict:
    """
    Load processed data từ file
    
    Args:
        data_path: Đường dẫn đến processed data
        
    Returns:
        Dictionary chứa dataset
    """
    data = {}
    
    # Load numpy files
    for file_path in glob.glob(os.path.join(data_path, "*.npy")):
        key = os.path.splitext(os.path.basename(file_path))[0]
        data[key] = np.load(file_path)
    
    # Load class names
    classes_file = os.path.join(data_path, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            data['classes'] = [line.strip() for line in f.readlines()]
    
    print(f"Processed data loaded from {data_path}")
    return data 