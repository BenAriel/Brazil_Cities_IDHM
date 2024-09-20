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
df=df.drop(columns=['CITY','STATE','MAC','WAL-MART','POST_OFFICES','FIXED_PHONES','UBER'])
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

df = df.apply(pd.to_numeric, errors='coerce')

#realizando teste sem normalização e redução de dimensão
# Importando bibliotecas necessárias
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
df_normalizar = df.drop(columns=['IDHM'])
df_idhm = df['IDHM']
colunas_numericas = df_normalizar.select_dtypes(include=['float64', 'int64']).columns
df_normalizar[colunas_numericas] = StandardScaler().fit_transform(df_normalizar[colunas_numericas])

df_normalizar = df_normalizar.drop(columns=['COMP_T'])

corr = df_normalizar.corr()

from sklearn.decomposition import PCA
print(corr)
pca = PCA(n_components='mle')

# Aplicando o PCA e transformando os dados
pca_components = pca.fit_transform(df_normalizar)

# Convertendo os componentes principais em um DataFrame
pca_df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(pca_components.shape[1])])

df_final = pd.concat([pca_df, df_idhm], axis=1)

print(df_final.isna().sum())
df_final = df_final.dropna()

X = df_final.drop(columns=['IDHM'])
y = df_final['IDHM']



param_grid_knn = {
    'n_neighbors': range(1, 40,2), 
    'weights': ['uniform', 'distance'],
    'p' : [1,2,3] 
    }

knn = KNeighborsRegressor()
scorer = make_scorer(r2_score)



grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, scoring=scorer, cv=5)

grid_search_knn.fit(X, y)

# Exibir os melhores parâmetros e a melhor pontuação de R² para cada algoritmo

print("KNN - Best Params:", grid_search_knn.best_params_)
print("KNN - Best R²:", grid_search_knn.best_score_)

