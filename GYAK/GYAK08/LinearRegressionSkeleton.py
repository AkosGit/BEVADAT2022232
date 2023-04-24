import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs=epochs
        self.lr=lr
        self.c=0
        self.m=0

    def fit(self, x: np.array, y: np.array):
        n = float(len(x)) # Number of elements in X
        self.losses = []
        for i in range(self.epochs): 
            y_pred = self.m*x + self.c  # The current predicted value of Y
            residuals = y - y_pred
            loss = np.sum(residuals ** 2)
            self.losses.append(loss)
            D_m = (-2/n) * sum(x * residuals)  # Derivative wrt m
            D_c = (-2/n) * sum(residuals)  # Derivative wrt c
            self.m = self.m - self.lr * D_m  # Update m
            self.c = self.c - self.lr * D_c  # Update c
            return self.losses
    def predict(self, X):
        pred = []
        for x in X:
            y_pred = self.m*x + self.c
            pred.append(y_pred)
        return pred
    def evaluate(self, x, y):
        err = np.mean((self.predict(x) - y) ** 2)
        return f"Mean squared error: {err}"


