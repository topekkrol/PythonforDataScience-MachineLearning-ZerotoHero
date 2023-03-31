# import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# reading the dataset
data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:-1] # dealete the first row
Y = data.iloc[:,-1].values

# zmiana stanów opisanych za pomocą Str na wartości numeryczne START
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[3])], remainder='passthrough') # [3] wskazanie kolumny która chcemy zmienić na liczby, OneHotEncoder() - typ zmiany (?), remainder - mówi co zrobić z pozostałymi kolumnami w DataSet
X = np.array(ct.fit_transform(X))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1) # test_size określa jaka ilość danych podlega testowi (?)
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

# changing the data to DataFrame
df = pd.DataFrame({'Real Vales':Y_test, 'Predicted Values':Y_pred })
# print(df)
## DO TEGO MOMENTU NA PODSTAWIE MODELU UTWORZYLIŚMY PRZEWIDYWANE DANE I ZESTAWILIŚMY JE Z PRAWDZIWYMI ALE SPRAWDZIĆ JAK SIĘ ZGADZAJĄ

#rmse Pomiar błędu modelu, za każdym razem wychodzi inna wartość (?)
print(np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))