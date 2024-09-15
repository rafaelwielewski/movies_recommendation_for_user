import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import joblib

# Carregar o modelo e os dados processados
model_knn = joblib.load('model_knn.pkl')
movies_pivot = joblib.load('movies_pivot.pkl')
movies_sparse = joblib.load('movies_sparse.pkl')

# Carregar as avaliações do usuário
user_ratings = pd.read_csv('ratings_user.csv', low_memory=False)

# Selecionar somente colunas que serão usadas
user_ratings = user_ratings[['movieId', 'rating']]

# Obter títulos dos filmes avaliados pelo usuário
movies = pd.read_csv('movies.csv', low_memory=False)
user_movies = movies[movies['movieId'].isin(user_ratings['movieId'])]['title'].tolist()

# Obter notas dos filmes avaliados pelo usuário
user_ratings_dict = dict(zip(movies[movies['movieId'].isin(user_ratings['movieId'])]['title'], user_ratings['rating']))

# Função para recomendar filmes
def recommend_movies(user_movies, user_ratings_dict, n_recommendations=10):
    # Obter índices dos filmes avaliados pelo usuário
    movie_indices = [movies_pivot.index.get_loc(movie) for movie in user_movies]
    
    # Calcular distâncias e índices para todos os filmes avaliados pelo usuário
    distances, indices = model_knn.kneighbors(movies_sparse[movie_indices], n_neighbors=n_recommendations + 1)
    
    # Usar um array NumPy para armazenar as recomendações
    recommendations = np.zeros(movies_pivot.shape[0])
    
    # Adicionar recomendações ao array
    for i in range(len(user_movies)):
        for j in range(1, len(distances[i])):
            movie_index = indices[i][j]
            recommendations[movie_index] += user_ratings_dict[user_movies[i]] / distances[i][j]
    
    # Obter os índices dos filmes recomendados ordenados pelos scores
    recommended_indices = np.argsort(recommendations)[::-1][:n_recommendations]
    
    # Converter os índices para títulos de filmes
    recommended_movies = movies_pivot.index[recommended_indices].tolist()
    
    return recommended_movies

# Fazer recomendações
recommended_movies = recommend_movies(user_movies, user_ratings_dict)
print("Filmes recomendados:")
for movie in recommended_movies:
    print(movie)