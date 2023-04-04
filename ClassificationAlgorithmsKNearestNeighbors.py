import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv('user+data.csv')
X = dataset.iloc[:, 2:4].values # wartości niezależne
Y = dataset.iloc[:, 4].values # wartości zależne ( zmienne )

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0) # 25% używamy jako testing data, reszta pozostaje jako train
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# n_neighbors określa ilość pobliskich elementów do zbadania i zakwalifikowania, zazwyczaj ustawia się 5
# cd. metric domyślnie ustalan się 'minkowski', model sprawdza odległość pomiędzy punktem a pozostałymi elementami
# p jest to równoważne ze standardową przestrzenią metryczną
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_pred, y_test)
print(cm) # wynik oznacza że 64 wartości zostały przewidziane poprawnie, 3 źle. Dla drugiej linijki ( są dwie linki ponieważ jedna przewiduje prawde, druga fałsz bądź na odwrót) 4 linijki źle przewidziały, natomiast 29 dobrze przewidziało. Sumarycznie 93 dobrze przewidzianych, 7 źle