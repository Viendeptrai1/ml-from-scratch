"""
Định nghĩa các model Machine Learning và Deep Learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


class MLModels:
    """Các model Machine Learning cơ bản"""
    
    @staticmethod
    def get_knn_model(n_neighbors=5):
        """K-Nearest Neighbors"""
        return KNeighborsClassifier(n_neighbors=n_neighbors)
    
    @staticmethod
    def get_svm_model(kernel='rbf', C=1.0):
        """Support Vector Machine"""
        return SVC(kernel=kernel, C=C, probability=True)
    
    @staticmethod
    def get_logistic_regression(C=1.0, max_iter=1000):
        """Logistic Regression"""
        return LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    
    @staticmethod
    def get_decision_tree(max_depth=None, min_samples_split=2):
        """Decision Tree"""
        return DecisionTreeClassifier(max_depth=max_depth, 
                                    min_samples_split=min_samples_split,
                                    random_state=42)
    
    @staticmethod
    def get_random_forest(n_estimators=100, max_depth=None):
        """Random Forest"""
        return RandomForestClassifier(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    random_state=42)
    
    @staticmethod
    def get_gradient_boosting(n_estimators=100, learning_rate=0.1):
        """Gradient Boosting"""
        return GradientBoostingClassifier(n_estimators=n_estimators,
                                        learning_rate=learning_rate,
                                        random_state=42)
    
    @staticmethod
    def get_naive_bayes():
        """Naive Bayes"""
        return GaussianNB()


class DeepLearningModels:
    """Các model Deep Learning"""
    
    @staticmethod
    def get_mlp_model(input_dim, num_classes, hidden_layers=[128, 64]):
        """
        Multi-Layer Perceptron
        
        Args:
            input_dim (int): Số features đầu vào
            num_classes (int): Số classes
            hidden_layers (list): Kích thước các hidden layer
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(0.3))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.3))
        
        # Output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss=loss,
                     metrics=metrics)
        
        return model
    
    @staticmethod
    def get_cnn_model(input_shape, num_classes):
        """
        Convolutional Neural Network cho ảnh
        
        Args:
            input_shape (tuple): Shape của ảnh đầu vào (height, width, channels)
            num_classes (int): Số classes
        """
        model = Sequential()
        
        # Conv Block 1
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Conv Block 2
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Conv Block 3
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        
        # Fully Connected Layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # Output layer
        if num_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(num_classes, activation='softmax'))
            loss = 'categorical_crossentropy'
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss=loss,
                     metrics=['accuracy'])
        
        return model
    
    @staticmethod
    def get_advanced_cnn_model(input_shape, num_classes):
        """
        CNN nâng cao với residual connections
        """
        inputs = keras.Input(shape=input_shape)
        
        # Initial conv layer
        x = Conv2D(64, (7, 7), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D((3, 3), padding='same')(x)
        
        # Residual blocks
        for filters in [64, 128, 256]:
            # Residual connection
            shortcut = x
            
            # Conv block
            x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            
            # Adjust shortcut if needed
            if shortcut.shape[-1] != filters:
                shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)
            
            # Add residual connection
            x = keras.layers.Add()([x, shortcut])
            x = keras.layers.Activation('relu')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Dropout(0.25)(x)
        
        # Global Average Pooling
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        
        # Output
        if num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
        else:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'categorical_crossentropy'
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss=loss,
                     metrics=['accuracy'])
        
        return model
    
    @staticmethod
    def get_callbacks():
        """Các callback hữu ích cho training"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7
            )
        ]
        return callbacks


# Factory function để lấy model dễ dàng
def get_model(model_type, **kwargs):
    """
    Factory function để tạo model
    
    Args:
        model_type (str): Loại model ('knn', 'svm', 'rf', 'mlp', 'cnn', etc.)
        **kwargs: Các tham số cho model
    
    Returns:
        Model object
    """
    ml_models = MLModels()
    dl_models = DeepLearningModels()
    
    model_map = {
        'knn': ml_models.get_knn_model,
        'svm': ml_models.get_svm_model,
        'logistic': ml_models.get_logistic_regression,
        'decision_tree': ml_models.get_decision_tree,
        'random_forest': ml_models.get_random_forest,
        'gradient_boosting': ml_models.get_gradient_boosting,
        'naive_bayes': ml_models.get_naive_bayes,
        'mlp': dl_models.get_mlp_model,
        'cnn': dl_models.get_cnn_model,
        'advanced_cnn': dl_models.get_advanced_cnn_model
    }
    
    if model_type not in model_map:
        raise ValueError(f"Model type '{model_type}' không được hỗ trợ")
    
    return model_map[model_type](**kwargs) 