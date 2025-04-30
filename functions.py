import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable

def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def convert_to_numpy(data: pd.DataFrame) -> np.ndarray:
    return data.values

def split_features_labels(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = data[:, 0]  # Labels - digits
    X = data[:, 1:]  # Features - pixel values
    return X, y

def normalize_data(X: np.ndarray) -> np.ndarray:
    return X / 255.0

def digit_to_one_hot(digit: int) -> np.ndarray:
    one_hot = np.zeros(10)
    one_hot[int(digit)] = 1
    return one_hot

def one_hot_to_digit(one_hot: np.ndarray) -> int:
    return np.argmax(one_hot)

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct_predictions = np.sum(y_true == y_pred)
    total_samples = len(y_true)
    return correct_predictions / total_samples

def convert_to_one_hot(y: np.ndarray) -> np.ndarray:
    n_samples = len(y)
    one_hot_encoded = np.zeros((n_samples, 10))
    for i in range(n_samples):
        one_hot_encoded[i] = digit_to_one_hot(y[i])
    return one_hot_encoded

def softmax(z: np.ndarray) -> np.ndarray:
    # subtract max value 
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1, 0)

# Actual neural network 
class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        # scale weights by 0.1 to prevent vanishing or exploding gradients
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        #storing intermediate values during forward pass
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
        #for storing history during training
        self.train_accuracy_history = []
        self.test_accuracy_history = []
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        # hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        
        # output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = softmax(self.z2)
        
        return self.a2
    
    def backward(self, X: np.ndarray, y_one_hot: np.ndarray, learning_rate: float) -> None:
        
        m = X.shape[0]  # number of samples
        
        dz2 = self.a2 - y_one_hot
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.forward(X)
        return np.array([one_hot_to_digit(p) for p in probabilities])
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions = self.predict(X)
        return calculate_accuracy(y, predictions)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray,
              learning_rate: float = 0.1, 
              epochs: int = 1000) -> Tuple[List[float], List[float]]:
        y_train_one_hot = convert_to_one_hot(y_train)
        
        for epoch in range(epochs):
            self.forward(X_train)
            self.backward(X_train, y_train_one_hot, learning_rate)
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                train_accuracy = self.evaluate(X_train, y_train)
                test_accuracy = self.evaluate(X_test, y_test)
                
                self.train_accuracy_history.append(train_accuracy)
                self.test_accuracy_history.append(test_accuracy)
                
                print(f"Epoch {epoch+1}/{epochs}: Training accuracy: {train_accuracy:.4f}, Testing accuracy: {test_accuracy:.4f}")
        
        train_accuracy = self.evaluate(X_train, y_train)
        test_accuracy = self.evaluate(X_test, y_test)
        
        print(f"\nFinal Results:")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Testing accuracy: {test_accuracy:.4f}")
        
        return self.train_accuracy_history, self.test_accuracy_history
