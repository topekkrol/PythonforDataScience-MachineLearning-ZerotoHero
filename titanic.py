import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plyk_2 = pd.read_csv('titanic.csv')
df_2= pd.DataFrame(plyk_2)

# korelacja = df_2[['Fare','Age']].corr()

# print(korelacja)

survived_ratio=df_2[['Pclass','Survived']].groupby('Pclass').sum()
no_survived_ratio=df_2['Pclass'].value_counts().sort_index(ascending=True)
survived_ratio['wszyscy'] = no_survived_ratio
survived_ratio['stosunek'] =round(((survived_ratio['Survived'])/(survived_ratio['wszyscy'])),2).map('{:,.0%}'.format)

# print(survived_ratio)

df = df_2[0:5][['Name','Sex']]
print(df.to_dict())