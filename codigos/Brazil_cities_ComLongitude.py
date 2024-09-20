import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None) 
df = pd.read_csv('BRAZIL_CITIES_REV2022.CSV')
print(df.head())


#verificando tipos de dados
pd.set_option('display.max_rows', None)
tipos_de_dados = df.dtypes
print(tipos_de_dados)


print(df.isna().sum())

#removendo colunas com correlação alta
df=df.drop(columns=['IDHM Ranking 2010','IDHM_Renda','IDHM_Longevidade','IDHM_Educacao'])
print(df)
#removendo colunas que eu considero inútil
df=df.drop(columns=['CITY','STATE'])
print(df)

print(df['CATEGORIA_TUR'].unique())
#tratando a coluna categoria de turismo
mapeamento_categoria = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}

df['CATEGORIA_TUR'] = df['CATEGORIA_TUR'].map(mapeamento_categoria)

print(df['CATEGORIA_TUR'].unique())

print(df['RURAL_URBAN'].unique())
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

print(df['GVA_MAIN'].unique())
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


print(df['REGIAO_TUR'].unique())

#tratando a coluna região de turismo
df['REGIAO_TUR'] = df['REGIAO_TUR'].apply(lambda x: 1 if x != '0' else 0)

#realizando teste sem normalização e redução de dimensão
# Importando bibliotecas necessárias
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer, r2_score

X = df.drop(columns=['IDHM'])
y = df['IDHM']

# Definir os parâmetros para o Grid Search
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'n_estimators': [20, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

param_grid_et = {
    'n_estimators': [20, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
et = ExtraTreesRegressor(random_state=42)
scorer = make_scorer(r2_score)


grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, scoring=scorer, cv=5)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, scoring=scorer, cv=5)
grid_search_et = GridSearchCV(estimator=et, param_grid=param_grid_et, scoring=scorer, cv=5)

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

