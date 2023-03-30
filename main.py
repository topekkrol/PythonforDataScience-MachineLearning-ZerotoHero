import pandas as pd



x = pd.Series([1,2,3,5,7,8,150])
mean = x.mean()
std = x.std()
z_scores = abs((x-mean)/std)
outliers = x[z_scores <= 1.5]
print(outliers)