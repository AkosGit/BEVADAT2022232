# %%
import numpy as np
from scipy.stats import mode
from typing import Tuple
from sklearn.metrics import confusion_matrix

# %%
def load_csv(lul)-> tuple[np.ndarray,np.ndarray]:
    np.random.seed(42)
    dataset = np.genfromtxt(lul,delimiter=',')
    print(dataset.shape)
    np.random.shuffle(dataset)
    x,y = dataset[:,:-1],dataset[:,-1] # ???
    return x,y
x,y = load_csv("iris.csv")

# %%
#átlag: nem tudja kezelni a hiányzó értékeket
np.mean(x,axis=0),np.var(x,axis=0)
# igy jobb
np.nanmean(x,axis=0),np.nanvar(x,axis=0)
# nan helyére átlag rakás
x[np.isnan(x)] = 3.5

# %%
# értékek törléseamik nagyobbak vagy kissebbek
# np.where()[0] 0 a sor index
# 
y = np.delete(y,np.where(x < 0.0)[0],axis=0)
y = np.delete(y,np.where(x > 10.0)[0],axis=0)
x = np.delete(x,np.where(x < 0.0)[0],axis=0)
x = np.delete(x,np.where(x > 10.0)[0],axis=0)
x.shape,y.shape

# %% [markdown]
# ## HERE COMES THE KNN

# %%
def train_test_split(features:np.ndarray,lables:np.ndarray,test_split_ratio:float):
    test_size = (len(features * test_split_ratio))
    train_size = len(features)- test_size
    assert len(features) == test_size + train_size, "Size mismatch"

    x_train,y_train=features[:train_size,:],lables[:train_size]
    x_test,y_test=features[train_size:,:],lables[train_size:]
    return(x_train,y_train,x_test,y_test)
x_train,y_train,x_test,y_test = train_test_split(x,y,0.2)

# %%
def euclidean(points,element_of_x):
    return np.sqrt(np.sum((points-element_of_x)**2,axis=0))

# %%

def predict(x_train,y_train,x_test,y_test,k):
    labels = []
    for x_test_element in x_test:
        #táv
        distances = euclidean(x_train,x_test_element)
        distances = np.array(sorted(zip(distances,y_train)))
        #leggyakoribb label
        labels_pred = mode(distances[:k,1],keepdims = False).mode
        labels_pred.append(labels_pred)
    return np.array(labels_pred,dtype=np.int64)
predict(x_train,y_train,x_test,y_test,3)


# %%
def accuracy(y_test,y_preds):
    true_positive = (y_test == y_preds).sum()
    return true_positive / len(y_test) *100
accuracy(y_test,y_preds)


