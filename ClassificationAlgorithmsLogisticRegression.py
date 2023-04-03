import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
dataset = pd.read_csv('user+data.csv')

x = dataset.iloc[:,[2,4]].values # wycięcie niezależnych wartości
y = dataset.iloc[:,4].values # pozostałe wartości

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0) # używamy 25% danych jako testowe, pozostałe jakie data set

