import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plyk = pd.read_csv(r'iris.csv')

df = pd.DataFrame(plyk)

x_axis = df.index
y_axis = df['petal.length']

sns.scatterplot( x=x_axis, y=y_axis, hue=df['variety'])
plt.show()
sns.stripplot(x=df['variety'], y=df['petal.length'])
plt.show()
