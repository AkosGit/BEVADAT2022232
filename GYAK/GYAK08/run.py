import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

from GYAK.GYAK08.LinearRegressionSkeleton import LinearRegression

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
X = df['petal length (cm)'].values
y = df['petal width (cm)'].values
reg=LinearRegression()
reg.fit(X,y)
pr=reg.predict(reg.X_test)
plt.scatter(reg.X_test, reg.y_test)
plt.plot([min(reg.X_test), max(reg.X_test)], [min(pr), max(pr)], color='red') # predicted
plt.show()
reg.evaluate(reg.X_train,reg.y_train)