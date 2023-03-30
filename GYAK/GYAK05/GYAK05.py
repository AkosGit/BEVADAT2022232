import numpy as np
from scipy.stats import mode
from typing import Tuple
from sklearn.metrics import confusion_matrix
class KNNClassifier:   
    @property
    def k_neighbors(self):
        return self.k
    def __init__(self, k :int,test_split_ratio :float) -> None:
            self.test_split_ratio=test_split_ratio
            self.k=k
    @staticmethod
    def load_csv(lul):
        np.random.seed(42)
        dataset = np.genfromtxt(lul,delimiter=',')
        print(dataset.shape)
        np.random.shuffle(dataset)
        x,y = dataset[:,:-1],dataset[:,-1] # ???
        return x,y

    def train_test_split(self,features:np.ndarray,lables:np.ndarray):
        test_size = (len(features * self.test_split_ratio))
        train_size = len(features)- test_size
        assert len(features) == test_size + train_size, "Size mismatch"
        self.x_train,self.y_train=features[:train_size,:],lables[:train_size]
        self.x_test,self.y_test=features[train_size:,:],lables[train_size:]

    def euclidean(self,element_of_x):
        return np.sqrt(np.sum((self.x_train-element_of_x)**2,axis=0))
    
    def predict(self,x_test):
        labels = []
        for x_test_element in x_test:
            #táv
            distances = self.euclidean(self.x_train,x_test_element)
            distances = np.array(sorted(zip(distances,self.y_train)))
            #leggyakoribb label
            labels_pred = mode(distances[:self.k,1],keepdims = False).mode
            labels_pred.append(labels_pred)
        self.y_preds= np.array(labels_pred,dtype=np.int64)

    def accuracy(self):
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) *100
    def confusion_matrix(self):
         return confusion_matrix(self.y_test,self.y_preds)