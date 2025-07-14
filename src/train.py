"""
Hàm training model và cross validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
import pickle
import os
import json
from datetime import datetime

from models import get_model, DeepLearningModels


def train_ml_model(model, X_train, y_train, X_test=None, y_test=None):
    """
    Train model Machine Learning
    
    Args:
        model: Model cần train
        X_train: Training features
        y_train: Training labels
        X_test: Test features (optional)
        y_test: Test labels (optional)
    
    Returns:
        dict: Kết quả training
    """
    print(f"Đang train model {type(model).__name__}...")
    
    # Training
    model.fit(X_train, y_train)
    
    # Evaluation
    train_score = model.score(X_train, y_train)
    results = {
        'model_name': type(model).__name__,
        'train_accuracy': train_score,
        'model': model
    }
    
    if X_test is not None and y_test is not None:
        test_score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        results['test_accuracy'] = test_score
        results['classification_report'] = classification_report(y_test, y_pred)
        
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
    else:
        print(f"Train Accuracy: {train_score:.4f}")
    
    return results


def cross_validate_model(model, X, y, cv_folds=5, scoring='accuracy'):
    """
    Cross validation cho model
    
    Args:
        model: Model cần validate
        X: Features
        y: Labels
        cv_folds: Số fold cho CV
        scoring: Metric để đánh giá
    
    Returns:
        dict: Kết quả cross validation
    """
    print(f"Đang thực hiện {cv_folds}-fold cross validation...")
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    results = {
        'cv_scores': scores,
        'mean_score': scores.mean(),
        'std_score': scores.std(),
        'model_name': type(model).__name__
    }
    
    print(f"CV Scores: {scores}")
    print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return results


def hyperparameter_tuning(model_type, param_grid, X, y, cv_folds=5):
    """
    Hyperparameter tuning với GridSearchCV
    
    Args:
        model_type: Loại model
        param_grid: Grid các hyperparameter
        X: Features
        y: Labels
        cv_folds: Số fold cho CV
    
    Returns:
        dict: Kết quả tuning và best model
    """
    print(f"Đang thực hiện hyperparameter tuning cho {model_type}...")
    
    # Tạo base model
    base_model = get_model(model_type)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return results


def train_deep_learning_model(model, X_train, y_train, X_val=None, y_val=None,
                            epochs=100, batch_size=32, callbacks=None):
    """
    Train Deep Learning model
    
    Args:
        model: Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Số epochs
        batch_size: Batch size
        callbacks: Keras callbacks
    
    Returns:
        dict: Kết quả training
    """
    print("Đang train Deep Learning model...")
    
    # Prepare validation data
    validation_data = None
    if X_val is not None and y_val is not None:
        validation_data = (X_val, y_val)
    
    # Default callbacks
    if callbacks is None:
        callbacks = DeepLearningModels.get_callbacks()
    
    # Training
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )
    
    results = {
        'model': model,
        'history': history.history,
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1]
    }
    
    if validation_data is not None:
        results['final_val_loss'] = history.history['val_loss'][-1]
        results['final_val_accuracy'] = history.history['val_accuracy'][-1]
        
        print(f"Final Train Accuracy: {results['final_train_accuracy']:.4f}")
        print(f"Final Validation Accuracy: {results['final_val_accuracy']:.4f}")
    else:
        print(f"Final Train Accuracy: {results['final_train_accuracy']:.4f}")
    
    return results


def compare_models(model_configs, X_train, y_train, X_test=None, y_test=None):
    """
    So sánh nhiều model
    
    Args:
        model_configs: List các config model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        pd.DataFrame: Bảng so sánh kết quả
    """
    results = []
    
    for config in model_configs:
        model_type = config['type']
        params = config.get('params', {})
        
        try:
            # Tạo model
            model = get_model(model_type, **params)
            
            # Cross validation
            cv_results = cross_validate_model(model, X_train, y_train)
            
            # Training và test
            train_results = train_ml_model(model, X_train, y_train, X_test, y_test)
            
            result = {
                'Model': model_type,
                'CV_Mean': cv_results['mean_score'],
                'CV_Std': cv_results['std_score'],
                'Train_Accuracy': train_results['train_accuracy']
            }
            
            if 'test_accuracy' in train_results:
                result['Test_Accuracy'] = train_results['test_accuracy']
            
            results.append(result)
            
        except Exception as e:
            print(f"Lỗi khi train model {model_type}: {e}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('CV_Mean', ascending=False)
    
    print("\n=== KẾT QUẢ SO SÁNH MODEL ===")
    print(results_df.to_string(index=False))
    
    return results_df


def save_model(model, model_name, save_dir='outputs/models'):
    """
    Lưu model đã train
    
    Args:
        model: Model cần lưu
        model_name: Tên model
        save_dir: Thư mục lưu
    """
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Keras model
    if hasattr(model, 'save'):
        file_path = os.path.join(save_dir, f"{model_name}_{timestamp}.h5")
        model.save(file_path)
    # Sklearn model
    else:
        file_path = os.path.join(save_dir, f"{model_name}_{timestamp}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Đã lưu model vào {file_path}")
    return file_path


def load_model(model_path):
    """
    Load model đã lưu
    
    Args:
        model_path: Đường dẫn đến model
    
    Returns:
        Model object
    """
    if model_path.endswith('.h5'):
        model = keras.models.load_model(model_path)
    elif model_path.endswith('.pkl'):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        raise ValueError("Chỉ hỗ trợ file .h5 và .pkl")
    
    print(f"Đã load model từ {model_path}")
    return model


def save_training_log(results, log_file='outputs/logs/training_log.json'):
    """
    Lưu log training
    
    Args:
        results: Kết quả training
        log_file: File log
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Load existing logs
    logs = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = json.load(f)
    
    # Add new log
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2, default=str)
    
    print(f"Đã lưu training log vào {log_file}") 