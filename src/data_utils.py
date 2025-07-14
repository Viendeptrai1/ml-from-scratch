"""
Tiện ích xử lý và tiền xử lý dữ liệu
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import pickle


def load_csv_data(file_path, **kwargs):
    """
    Load dữ liệu từ file CSV
    
    Args:
        file_path (str): Đường dẫn đến file CSV
        **kwargs: Các tham số cho pandas.read_csv
    
    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu
    """
    try:
        data = pd.read_csv(file_path, **kwargs)
        print(f"Đã load thành công {data.shape[0]} dòng, {data.shape[1]} cột từ {file_path}")
        return data
    except Exception as e:
        print(f"Lỗi khi load file {file_path}: {e}")
        return None


def preprocess_numerical_data(data, features, method='standard'):
    """
    Tiền xử lý dữ liệu số
    
    Args:
        data (pd.DataFrame): Dữ liệu gốc
        features (list): Danh sách các cột số cần xử lý
        method (str): Phương pháp scaling ('standard', 'minmax')
    
    Returns:
        tuple: (dữ liệu đã xử lý, scaler object)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method phải là 'standard' hoặc 'minmax'")
    
    data_scaled = data.copy()
    data_scaled[features] = scaler.fit_transform(data[features])
    
    return data_scaled, scaler


def preprocess_categorical_data(data, features):
    """
    Tiền xử lý dữ liệu categorical
    
    Args:
        data (pd.DataFrame): Dữ liệu gốc
        features (list): Danh sách các cột categorical
    
    Returns:
        tuple: (dữ liệu đã xử lý, dict các encoder)
    """
    data_encoded = data.copy()
    encoders = {}
    
    for feature in features:
        encoder = LabelEncoder()
        data_encoded[feature] = encoder.fit_transform(data[feature].astype(str))
        encoders[feature] = encoder
    
    return data_encoded, encoders


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Chia dữ liệu thành train/test
    
    Args:
        X: Features
        y: Target
        test_size (float): Tỷ lệ test set
        random_state (int): Random seed
        stratify: Stratify parameter
    
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, 
                          random_state=random_state, stratify=stratify)


def load_image_data(image_dir, target_size=(224, 224)):
    """
    Load và resize ảnh từ thư mục
    
    Args:
        image_dir (str): Đường dẫn thư mục chứa ảnh
        target_size (tuple): Kích thước ảnh sau resize
    
    Returns:
        tuple: (mảng ảnh, danh sách tên file)
    """
    images = []
    filenames = []
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            try:
                img = Image.open(img_path)
                img = img.resize(target_size)
                img_array = np.array(img)
                
                # Chuyển grayscale sang RGB nếu cần
                if len(img_array.shape) == 2:
                    img_array = np.stack([img_array] * 3, axis=-1)
                
                images.append(img_array)
                filenames.append(filename)
            except Exception as e:
                print(f"Lỗi khi load ảnh {filename}: {e}")
    
    return np.array(images), filenames


def save_processed_data(data, file_path):
    """
    Lưu dữ liệu đã xử lý
    
    Args:
        data: Dữ liệu cần lưu
        file_path (str): Đường dẫn file lưu
    """
    try:
        if file_path.endswith('.csv'):
            data.to_csv(file_path, index=False)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError("Chỉ hỗ trợ định dạng .csv và .pkl")
        
        print(f"Đã lưu dữ liệu vào {file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu dữ liệu: {e}")


def handle_missing_values(data, strategy='mean'):
    """
    Xử lý missing values
    
    Args:
        data (pd.DataFrame): Dữ liệu có missing values
        strategy (str): Chiến lược xử lý ('mean', 'median', 'mode', 'drop')
    
    Returns:
        pd.DataFrame: Dữ liệu đã xử lý missing values
    """
    data_clean = data.copy()
    
    if strategy == 'drop':
        data_clean = data_clean.dropna()
    else:
        for column in data_clean.columns:
            if data_clean[column].isnull().sum() > 0:
                if data_clean[column].dtype in ['int64', 'float64']:
                    if strategy == 'mean':
                        data_clean[column].fillna(data_clean[column].mean(), inplace=True)
                    elif strategy == 'median':
                        data_clean[column].fillna(data_clean[column].median(), inplace=True)
                else:
                    # Categorical data - dùng mode
                    data_clean[column].fillna(data_clean[column].mode()[0], inplace=True)
    
    return data_clean 