import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import joblib

# Importar arquivo de filmes e visualizar as primeiras linhas
movies = pd.read_csv('movies.csv', low_memory=False)
ratings = pd.read_csv('ratings.csv', low_memory=False)

# Selecionar somente colunas que serão usadas
movies = movies[['movieId', 'title']]
ratings = ratings[['userId', 'movieId', 'rating']]

# Agrupar as avaliações por ID do filme e contar o número de avaliações
total_ratings = ratings.groupby('movieId')['rating'].count().reset_index()
total_ratings.columns = ['movieId', 'total_ratings']

# Adicionar total ratings para movies conforme o movieId
movies = movies.merge(total_ratings, on='movieId', how='left')

# Remover filmes nulos do banco de dados
movies.dropna(inplace=True)
ratings.dropna(inplace=True)

# Remover filmes com menos de 1000 avaliações
movies = movies[movies['total_ratings'] >= 1000]

# Remover avaliações de filmes com menos de 1000 avaliações
ratings = ratings[ratings['movieId'].isin(movies['movieId'])]

# Verificar quantidade de avaliações por usuário
ratings_count = ratings.groupby('userId')['rating'].count()

# Filtrar os usuários com mais de 100 avaliações
y = ratings_count[ratings_count > 100].index

# Filtrar as avaliações dos usuários com mais de 100 avaliações
ratings = ratings[ratings['userId'].isin(y)]

# Concatenar os datasets de filmes e avaliações
ratings_and_movies = ratings.merge(movies, on='movieId')

# Descartar valores duplicados verificando userId e movieId
ratings_and_movies.drop_duplicates(subset=['userId', 'movieId'], keep='first', inplace=True)

# Remover movieId
del ratings_and_movies['movieId']

# Agrupar por título e userId e calcular a média das avaliações
ratings_and_movies = ratings_and_movies.groupby(['title', 'userId']).rating.mean().reset_index()

# Fazer pivot da tabela
movies_pivot = ratings_and_movies.pivot(index='title', columns='userId', values='rating')

# Substituir ratings nulas por zero
movies_pivot.fillna(0, inplace=True)

# Criar uma matriz esparsa
movies_sparse = csr_matrix(movies_pivot)

# Criar e treinar modelo KNN
model_knn = NearestNeighbors(algorithm='brute')
model_knn.fit(movies_sparse)

# Salvar o modelo e os dados processados
joblib.dump(model_knn, 'model_knn.pkl')
joblib.dump(movies_pivot, 'movies_pivot.pkl')
joblib.dump(movies_sparse, 'movies_sparse.pkl')