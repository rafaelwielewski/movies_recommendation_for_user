# Projeto de Recomendação de Filmes

Este projeto implementa um sistema de recomendação de filmes usando a lista de filmes e avaliações do MovieLens 32k. O projeto consiste em dois principais componentes: um script de treinamento (`train.py`) e um script de recomendação (`recommend.py`).

## Estrutura do Projeto

- `train.py`: Script para treinar o modelo de recomendação.
- `recommend.py`: Script para fazer recomendações de filmes usando o modelo treinado.
- `movies.csv`: Arquivo CSV contendo a lista de filmes.
- `ratings.csv`: Arquivo CSV contendo as avaliações dos filmes.
- `ratings_user.csv`: Arquivo CSV contendo as avaliações dos filmes feitas pelo usuário.

## Dados

Os dados utilizados neste projeto foram obtidos do MovieLens 32M, disponível em: [https://grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens)

## Dependências

- pandas
- numpy
- scikit-learn
- scipy
- joblib

Você pode instalar as dependências usando o seguinte comando:

```sh
pip install pandas numpy scikit-learn scipy joblib
```

## Treinamento do Modelo

O script `train.py` é responsável por treinar o modelo de recomendação. Ele realiza as seguintes etapas:

1. Carrega os dados de filmes e avaliações.
2. Filtra e processa os dados.
3. Cria uma matriz esparsa de avaliações.
4. Treina um modelo KNN (K-Nearest Neighbors) usando a matriz esparsa.
5. Salva o modelo treinado e os dados processados.

### Exemplo de Uso

Para treinar o modelo, execute o script `train.py`:

```sh
python train.py
```

## Recomendação de Filmes

O script `recommend.py` é responsável por fazer recomendações de filmes usando o modelo treinado. Ele realiza as seguintes etapas:

1. Carrega o modelo treinado e os dados processados.
2. Carrega as avaliações dos filmes feitas pelo usuário.
3. Calcula as distâncias e índices dos filmes mais similares.
4. Gera uma lista de recomendações de filmes.

### Exemplo de Uso

Para fazer recomendações de filmes, execute o script `recommend.py`:

```sh
python recommend.py
```

## Conclusão

Este projeto demonstra como criar um sistema de recomendação de filmes usando a biblioteca scikit-learn e dados do MovieLens. O modelo KNN é treinado para encontrar filmes similares com base nas avaliações dos usuários, e o script de recomendação utiliza esse modelo para sugerir filmes aos usuários com base em suas avaliações anteriores.

Sinta-se à vontade para explorar e modificar o código para atender às suas necessidades específicas. Este projeto pode ser expandido de várias maneiras, como:

- Adicionar mais dados de entrada para melhorar a precisão das recomendações.
- Experimentar com diferentes algoritmos de recomendação.
- Implementar uma interface de usuário para facilitar a interação com o sistema de recomendação.
- Integrar o sistema de recomendação com uma API para permitir acesso remoto.

Esperamos que este projeto sirva como um bom ponto de partida para suas próprias implementações de sistemas de recomendação.
