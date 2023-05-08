# imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_digits as load
from sklearn.cluster import k_means
from scipy import stats

class KMeansOnDigits():
    def __init__(self,n_clusters,random_state) -> None:
        self.random_state=random_state
        self.n_clusters=n_clusters
    def load_dataset(self):
        self.digits=load()
    def predict(self):
        self.model=KMeans(n_clusters=self.n_clusters,random_state=self.random_state)
        pred=self.model.fit_predict(X=self.digits.data,y=self.digits.target)
        #return (np.array(pred),k)
        self.clusters=pred
    def get_labels(self):
        result = np.zeros_like(self.clusters)
        for cluster in range(10):
            mask = (self.clusters == cluster)
            target = self.digits.target[mask]
            mode = np.bincount(target).argmax()
            result[mask] = mode
        self.labels=result 
    def calc_accuracy(self):
        self.mat=accuracy_score(self.digits.target,self.labels)
