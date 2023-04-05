import pandas as pd
import numpy as np

# for visualization
import seaborn as sns
import  matplotlib.pyplot as plt


sns.set_style('white')
# %matplotlib inline

column_names =['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)


movie_titles = pd.read_csv('Movie_Id_Titles')

# dodanie baz danych do siebie, a właściwie dodanie bazy movie_title do column_names
df = pd.merge(df, movie_titles, on='item_id')


#uzyskanie ilości ocen filmów, wskazujemy title, raiting w kwadratowych nawiasach ponieważ na tej kolumnie wykonywane są dodatkowe operacje

# print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())


#uzyskanie najwyższej średniej filmów, wskazujemy title, raiting w kwadratowych nawiasach ponieważ na tej kolumnie wykonywane są dodatkowe operacje

# print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head())

raiting = pd.DataFrame(df.groupby('title')['rating'].mean())
raiting['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
# print(raiting.head())

# plt.figure(figsize=(10,4))
# raiting['num of ratings'].hist(bins=70)
# plt.show()

# plt.figure(figsize=(10,4))
# raiting['rating'].hist(bins=70)
# plt.show()

# tworzenie fajnego wykresu
sns.jointplot(x='rating', y='num of ratings', data=raiting, alpha= 0.5)
# plt.show()


# tworzenie tabeli przestawnej
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# print(raiting.sort_values('num of ratings', ascending=False))

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
# print(starwars_user_ratings.head())


# tworzenie listy filmów dla użytkownika na podstawie korelacji.
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)

similar_to_lialiar = moviemat.corrwith(liarliar_user_ratings)


corr_strwars = pd.DataFrame(similar_to_starwars, columns=['Correlation']) # stworzenie duplikatu similiar_to_starwars z dodatkową kolumną Correlation
corr_strwars.dropna(inplace=True) # usunięcie pustych wartosci
# print(corr_strwars.sort_values('Correlation', ascending=False).head(10))

corr_strwars = corr_strwars.join(raiting['num of ratings'])

print(corr_strwars[corr_strwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())


corr_liarliar = pd.DataFrame(similar_to_lialiar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(raiting['num of ratings'])
print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head())