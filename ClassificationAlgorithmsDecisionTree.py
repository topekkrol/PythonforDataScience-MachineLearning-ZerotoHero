import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from  sklearn.model_selection import  train_test_split
from sklearn import metrics
from sklearn import preprocessing

# col_names=['company','job','defree','salary_more_than_100k']
data = pd.read_csv('salaries.csv')

feature_cols = ['company','job','degree']
x =data[feature_cols] # wskazanie na osi x
y = data['salary_more_then_100k'] # wskazanie na osi y

#zmiana na 'Label' zrozumiałe dla maching learning
label_encoder = preprocessing.LabelEncoder()
data['company'] = label_encoder.fit_transform(data['company']) # zmiana danych tekstowych na liczbowe
data['job'] = label_encoder.fit_transform(data['job']) # zmiana danych tekstowych na liczbowe
data['degree'] = label_encoder.fit_transform(data['degree']) # zmiana danych tekstowych na liczbowe

#podział bazy danych na wartości i zmienną docelową / split the dataset in features and target variable
feature_cols = ['company','job','degree']

x = data[feature_cols] # wszystkie wartości poza zmienną
y = data['salary_more_then_100k'] # tylko zmienan wartość

# x = data.values[1:,:3]
# y = data.values[1:,3] #1:,3 one means we are not using the beader - oznaczenie oznacza że nie używamy nagłówka


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=100)

#create decision three classifier object using entropy

clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)

# Train decision three classifier

clf_entropy = clf_entropy.fit(x_train, y_train)

#Predict the response for test dataset

y_pred =clf_entropy.predict(x_test)

print('dokładność', metrics.accuracy_score(y_test,y_pred))