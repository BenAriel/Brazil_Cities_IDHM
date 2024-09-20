import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pd.set_option('display.max_columns', None)
df = pd.read_csv('BRAZIL_CITIES_REV2022.CSV')
print(df.head())

# Verificando tipos de dados
pd.set_option('display.max_rows', None)
tipos_de_dados = df.dtypes
print(tipos_de_dados)

# Verificando valores ausentes
print(df.isna().sum())

# Removendo colunas com correlação alta
df = df.drop(columns=['IDHM Ranking 2010', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao'])

# Removendo colunas consideradas inúteis
df = df.drop(columns=['CITY', 'STATE', 'MAC', 'WAL-MART', 'POST_OFFICES', 'FIXED_PHONES', 'UBER'])

# Tratando a coluna 'CATEGORIA_TUR'
print(df['CATEGORIA_TUR'].unique())
mapeamento_categoria = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
df['CATEGORIA_TUR'] = df['CATEGORIA_TUR'].map(mapeamento_categoria)
print(df['CATEGORIA_TUR'].unique())

# Tratando a coluna 'RURAL_URBAN'
print(df['RURAL_URBAN'].unique())
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

# Tratando a coluna 'GVA_MAIN'
print(df['GVA_MAIN'].unique())
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
df['GVA_MAIN'] = df['GVA_MAIN'].map(mapeamento_servicos)

# Tratando a coluna 'REGIAO_TUR'
print(df['REGIAO_TUR'].unique())
df['REGIAO_TUR'] = df['REGIAO_TUR'].apply(lambda x: 1 if x != '0' else 0)

tipos_de_dados = df.dtypes
print(tipos_de_dados)

print(len(df.index))
print(df.shape[1])
X = df.drop(columns=['IDHM'])
y = df['IDHM']

# Normalização e PCA para reduzir a dimensionalidade (mantendo 95% da variância explicada)
scaler = StandardScaler()
pca = PCA(n_components='mle')

# Definindo os modelos
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
et = ExtraTreesRegressor(random_state=42)

# Criando pipelines
pipeline_dt = Pipeline(steps=[
    ('scaler', scaler),
    ('pca', pca),
    ('dt', dt)
])

pipeline_rf = Pipeline(steps=[
    ('scaler', scaler),
    ('pca', pca),
    ('rf', rf)
])

pipeline_et = Pipeline(steps=[
    ('scaler', scaler),
    ('pca', pca),
    ('et', et)
])

# Definindo o grid de parâmetros para cada modelo
param_grid_dt = {
    'dt__max_depth': [None, 10, 20, 30],
    'dt__min_samples_split': [2, 10, 20],
    'dt__min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'rf__n_estimators': [20, 30, 50],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 10, 20],
    'rf__min_samples_leaf': [1, 5, 10]
}

param_grid_et = {
    'et__n_estimators': [20, 30, 50],
    'et__max_depth': [None, 10, 20, 30],
    'et__min_samples_split': [2, 10, 20],
    'et__min_samples_leaf': [1, 5, 10]
}

# Definindo o scoring
scorer = make_scorer(r2_score)

# Aplicando GridSearchCV para cada modelo
grid_search_dt = GridSearchCV(estimator=pipeline_dt, param_grid=param_grid_dt, scoring=scorer, cv=5)
grid_search_rf = GridSearchCV(estimator=pipeline_rf, param_grid=param_grid_rf, scoring=scorer, cv=5)
grid_search_et = GridSearchCV(estimator=pipeline_et, param_grid=param_grid_et, scoring=scorer, cv=5)

# Ajustando os modelos
grid_search_dt.fit(X, y)
grid_search_rf.fit(X, y)
grid_search_et.fit(X, y)

# Exibir os melhores parâmetros e a melhor pontuação de R² para cada algoritmo
print("Decision Tree - Best Params:", grid_search_dt.best_params_)
print("Decision Tree - Best R²:", grid_search_dt.best_score_)

print("Random Forest - Best Params:", grid_search_rf.best_params_)
print("Random Forest - Best R²:", grid_search_rf.best_score_)

print("Extra Trees - Best Params:", grid_search_et.best_params_)
print("Extra Trees - Best R²:", grid_search_et.best_score_)
