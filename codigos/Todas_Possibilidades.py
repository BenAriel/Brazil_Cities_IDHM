import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.neighbors import LocalOutlierFactor

# Carregar os dados
df = pd.read_csv('BRAZIL_CITIES_REV2022.CSV')

# Preprocessamento básico dos dados
df = df.drop(columns=['IDHM Ranking 2010', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao'])
#colunas que eu achei desnecessárias para os modelos
df = df.drop(columns=['CITY', 'STATE', 'MAC', 'WAL-MART', 'POST_OFFICES', 'FIXED_PHONES', 'UBER'])

#passando colunas categoricas para numericas
df['CATEGORIA_TUR'] = df['CATEGORIA_TUR'].map({'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5})
df['RURAL_URBAN'] = df['RURAL_URBAN'].map({
    'Urbano': 1, 'Rural Adjacente': 2, 'Rural Remoto': 3, 
    'Intermediário Adjacente': 4, 'Intermediário Remoto': 5, 
    'Sem classificação': None, '0': None})
df = df.dropna(subset=['RURAL_URBAN'])
df['GVA_MAIN'] = df['GVA_MAIN'].map({
    'Demais serviços': 1, 'Administração, defesa, educação e saúde públicas e seguridade social': 2,
    'Agricultura, inclusive apoio à agricultura e a pós colheita': 3, 'Indústrias de transformação': 4,
    'Comércio e reparação de veículos automotores e motocicletas': 5, 'Pecuária, inclusive apoio à pecuária': 6,
    'Eletricidade e gás, água, esgoto, atividades de gestão de resíduos e descontaminação': 7, 'Indústrias extrativas': 8,
    'Construção': 9, 'Produção florestal, pesca e aquicultura': 10
})
df['REGIAO_TUR'] = df['REGIAO_TUR'].apply(lambda x: 1 if x != '0' else 0)

X = df.drop(columns=['IDHM'])
y = df['IDHM']

# Função para remoção de outliers
def remove_outliers(X, contamination=0.05):
    lof = LocalOutlierFactor(contamination=contamination)
    outliers = lof.fit_predict(X) == 1
    return X[outliers], y[outliers]

# Definindo os modelos
models = {
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Extra Trees': ExtraTreesRegressor(random_state=42)
}

# Definindo os pipelines
def create_pipeline(model, use_pca=False):
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components='mle')))
    steps.append(('model', model))
    return Pipeline(steps=steps)

# Definir os grids de parâmetros
param_grids = {
    'Decision Tree': {
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 10, 20],
        'model__min_samples_leaf': [1, 5, 10]
    },
    'Random Forest': {
        'model__n_estimators': [20, 30, 50],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 10, 20],
        'model__min_samples_leaf': [1, 5, 10]
    },
    'Extra Trees': {
        'model__n_estimators': [20, 30, 50],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 10, 20],
        'model__min_samples_leaf': [1, 5, 10]
    }
}

# Função para rodar grid search com combinações
def run_grid_search(X, y, model_name, model, param_grid, remove_outliers_flag=False, use_pca=False):
    if remove_outliers_flag:
        X, y = remove_outliers(X)

    pipeline = create_pipeline(model, use_pca=use_pca)
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(r2_score))
    grid_search.fit(X, y)
    
    return grid_search.best_params_, grid_search.best_score_

# Armazenar os melhores resultados
best_results = []

# Executar as combinações
for model_name, model in models.items():
    for remove_outliers_flag in [True, False]:
        for use_pca in [True, False]:
            params, score = run_grid_search(X, y, model_name, clone(model), param_grids[model_name], remove_outliers_flag, use_pca)
            best_results.append({
                'Model': model_name,
                'Remove Outliers': remove_outliers_flag,
                'Use PCA': use_pca,
                'Best Params': params,
                'Best Score (R²)': score
            })

# Salvar os resultados em um arquivo txt
with open('best_model_combinations.txt', 'w') as f:
    for result in best_results:
        f.write(f"Model: {result['Model']}\n")
        f.write(f"Remove Outliers: {result['Remove Outliers']}\n")
        f.write(f"Use PCA: {result['Use PCA']}\n")
        f.write(f"Best Params: {result['Best Params']}\n")
        f.write(f"Best Score (R²): {result['Best Score (R²)']}\n")
        f.write('-' * 40 + '\n')

print("Melhores combinações salvas em 'best_model_combinations.txt'")
