import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier as dtc
dataset = pd.read_csv("ml2.csv")
dataset = dataset.apply(LabelEncoder().fit_transform)
x = dataset.iloc[:,:-1]
y = dataset['Buys']
c = dtc().fit(x,y)
print(c.predict([[2,0,1,0],[1,0,0,0],[1,0,0,0]]))
from sklearn import tree
tree.plot_tree(c)
