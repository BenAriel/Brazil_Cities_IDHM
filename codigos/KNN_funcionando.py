import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)

# Carregando o dataset
df = pd.read_csv('BRAZIL_CITIES_REV2022.CSV')

# Verificando valores ausentes
print(df.isna().sum())

# Removendo colunas desnecessárias
df = df.drop(columns=['IDHM Ranking 2010', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao'])
df = df.drop(columns=['CITY', 'STATE', 'MAC', 'WAL-MART', 'POST_OFFICES', 'FIXED_PHONES', 'UBER'])

print(df['CATEGORIA_TUR'].unique())
print(df['RURAL_URBAN'].unique())
print(df['GVA_MAIN'].unique())
print(df['REGIAO_TUR'].unique())

#tratando a coluna categoria de turismo
mapeamento_categoria = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

df['CATEGORIA_TUR'] = df['CATEGORIA_TUR'].map(mapeamento_categoria)

print(df['CATEGORIA_TUR'].unique())

#tratando as atividades principais
mapeamento_rural_urban = {
    'Urbano': 1,
    'Rural Adjacente': 2,
    'Rural Remoto': 3,
    'Intermediário Adjacente': 4,
    'Intermediário Remoto': 5,
    'Sem classificação': None,
    '0': None
}
df['RURAL_URBAN'] = df['RURAL_URBAN'].map(mapeamento_rural_urban)
df = df.dropna(subset=['RURAL_URBAN'])
print(df['RURAL_URBAN'].unique())


#tratando a coluna rural_ourbano
mapeamento_servicos = {
    'Demais serviços': 1,
    'Administração, defesa, educação e saúde públicas e seguridade social': 2,
    'Agricultura, inclusive apoio à agricultura e a pós colheita': 3,
    'Indústrias de transformação': 4,
    'Comércio e reparação de veículos automotores e motocicletas': 5,
    'Pecuária, inclusive apoio à pecuária': 6,
    'Eletricidade e gás, água, esgoto, atividades de gestão de resíduos e descontaminação': 7,
    'Indústrias extrativas': 8,
    'Construção': 9,
    'Produção florestal, pesca e aquicultura': 10
}

# Aplicando o mapeamento à coluna de serviços
df['GVA_MAIN'] = df['GVA_MAIN'].map(mapeamento_servicos)

# Exibindo os tipos de serviços únicos
print(df['GVA_MAIN'].unique())
#tratando a coluna região de turismo
df['REGIAO_TUR'] = df['REGIAO_TUR'].apply(lambda x: 1 if x != '0' else 0)
print(df['REGIAO_TUR'].unique())
# Convertendo variáveis categóricas para numéricas com one-hot encoding
df = pd.get_dummies(df, drop_first=True)

tipos_de_dados = df.dtypes
print(tipos_de_dados)

print(df.columns)

# Definindo as variáveis independentes e dependentes
X = df.drop(columns=['IDHM'])
y = df['IDHM']

# Normalizando os dados
scaler = StandardScaler()

# Reduzindo dimensionalidade com PCA (mantendo 95% da variância explicada)
pca = PCA(n_components=0.95)

# Definindo o KNN
knn = KNeighborsRegressor()

# Criando pipeline
pipeline = Pipeline(steps=[
    ('scaler', scaler),
    ('pca', pca),
    ('knn', knn)
])

# Definindo o grid de parâmetros
param_grid_knn = {
    'knn__n_neighbors': range(1, 100, 2),
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}

# Definindo o scoring
scorer = make_scorer(r2_score)

# Criando o GridSearch
grid_search_knn = GridSearchCV(estimator=pipeline, param_grid=param_grid_knn, scoring=scorer, cv=5)

# Ajustando o GridSearch
grid_search_knn.fit(X, y)

# Melhor parâmetro encontrado
print(f"Melhor parâmetro para KNN: {grid_search_knn.best_params_}")
# Avaliando o modelo final

print("KNN - Best R²:", grid_search_knn.best_score_)
