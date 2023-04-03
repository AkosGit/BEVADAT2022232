import pandas as pd
from typing import Tuple
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import seaborn as sns

class KNNClassifier:
    @property
    def k_neighbors(self) -> int:
        return self.k

    def __init__(self, k:int, test_split_ratio: float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    @staticmethod
    def load_csv(path: str) -> Tuple[pd.DataFrame, pd.Series]:
        dataFrame = pd.read_csv(path)
        dataFrame = dataFrame.sample(frac=1, random_state=42).reset_index(drop=True)
        return dataFrame.iloc[:,:-1], dataFrame.iloc[:,-1]
    
    def train_test_split(self,
                         features:pd.DataFrame,
                         labels:pd.Series):
    
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        self.x_train,self.y_train = features.iloc[:train_size,:],labels.iloc[:train_size]
        self.x_test,self.y_test = features.iloc[train_size:train_size+test_size,:], labels.iloc[train_size:train_size + test_size]

    def euclidean(self, element_of_x:pd.DataFrame) -> pd.DataFrame:
        return ((self.x_train - element_of_x)**2).sum(axis=1) ** .5
    
    def predict(self, x_test: pd.DataFrame):
        labels_pred = []
        for index, x_test_element in x_test.iterrows():
            distances = self.euclidean(x_test_element)
            distances = pd.DataFrame(sorted(zip(distances, self.y_train)))
            label_pred = distances.iloc[:self.k,1].mode()
            labels_pred.append(label_pred)

        self.y_preds = pd.DataFrame(labels_pred).iloc[:,0]

    def accuracy(self) -> float:
        true_positive = (self.y_test.reset_index(drop=True) == self.y_preds.reset_index(drop=True)).sum()
        return true_positive/len(self.y_test) * 100
    
    def confusion_matrix(self):
        conf_matrix = confusion_matrix(self.y_test, self.y_preds)
        return conf_matrix
    

    def best_k(self) -> Tuple[int, float]:
        accuracies = []
        for i in range(1, 21):
            self.k = i
            self.predict(self.x_test)
            acc = self.accuracy()
            accuracies.append((i, acc))
        best_k, best_acc = max(accuracies, key=lambda x: x[1])
        return (best_k, round(best_acc, 2))