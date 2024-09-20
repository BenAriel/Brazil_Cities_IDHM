import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, r2_score
import matplotlib.pyplot as plt

# Configurações do pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Função para carregar e limpar os dados
def carregar_dados(filepath):
    df = pd.read_csv(filepath)

    # Remover colunas desnecessárias
    df = df.drop(columns=['IDHM Ranking 2010', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao',
                          'CITY', 'STATE', 'MAC', 'WAL-MART', 'POST_OFFICES', 'FIXED_PHONES', 'UBER'])

    # Mapeamentos para as variáveis categóricas
    mapeamento_categoria = {'0': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
    df['CATEGORIA_TUR'] = df['CATEGORIA_TUR'].map(mapeamento_categoria)

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

    df['REGIAO_TUR'] = df['REGIAO_TUR'].apply(lambda x: 1 if x != '0' else 0)


    return df

# Carregar e preparar os dados
df = carregar_dados('BRAZIL_CITIES_REV2022.CSV')

# Definir variáveis independentes e dependentes
X = df.drop(columns=['IDHM'])
y = df['IDHM']

# Normalizar e aplicar PCA para redução de dimensionalidade
scaler = StandardScaler()
pca = PCA(n_components='mle')

# Definir modelos
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)
et = ExtraTreesRegressor(random_state=42)
knn = KNeighborsRegressor()
lr = LinearRegression()

# Função para criar pipeline e realizar GridSearch
resultados_r2 = {}

def realizar_grid_search(modelo, param_grid):
    pipeline = Pipeline(steps=[
        ('scaler', scaler),
        ('pca', pca),
        (modelo.__class__.__name__.lower(), modelo)
    ])
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=make_scorer(r2_score), cv=5)
    grid_search.fit(X, y)
    
    # Armazenar os melhores resultados
    resultados_r2[modelo.__class__.__name__] = grid_search.best_score_
    
    print(f"{modelo.__class__.__name__} - Best Params:", grid_search.best_params_)
    print(f"{modelo.__class__.__name__} - Best R²:", grid_search.best_score_)

# Parâmetros para os modelos
param_grid_dt = {
    'decisiontreeregressor__max_depth': [None, 10, 20, 30],
    'decisiontreeregressor__min_samples_split': [2, 10, 20],
    'decisiontreeregressor__min_samples_leaf': [1, 5, 10]
}

param_grid_rf = {
    'randomforestregressor__n_estimators': [20, 30, 50],
    'randomforestregressor__max_depth': [None, 10, 20, 30],
    'randomforestregressor__min_samples_split': [2, 10, 20],
    'randomforestregressor__min_samples_leaf': [1, 5, 10]
}

param_grid_et = {
    'extratreesregressor__n_estimators': [20, 30, 50],
    'extratreesregressor__max_depth': [None, 10, 20, 30],
    'extratreesregressor__min_samples_split': [2, 10, 20],
    'extratreesregressor__min_samples_leaf': [1, 5, 10]
}

param_grid_knn = {
    'kneighborsregressor__n_neighbors': range(1, 100, 2),
    'kneighborsregressor__weights': ['uniform', 'distance'],
    'kneighborsregressor__p': [1, 2]
}

param_grid_lr = {
    'linearregression__fit_intercept': [True, False]
}

# Aplicando GridSearch para cada modelo
realizar_grid_search(dt, param_grid_dt)
realizar_grid_search(rf, param_grid_rf)
realizar_grid_search(et, param_grid_et)
realizar_grid_search(knn, param_grid_knn)
realizar_grid_search(lr, param_grid_lr)

# Gerar gráfico comparativo dos resultados R²
plt.figure(figsize=(10, 6))
modelos = list(resultados_r2.keys())
scores = list(resultados_r2.values())

plt.barh(modelos, scores, color='skyblue')
plt.xlabel('Melhor R²')
plt.title('Comparação de Modelos - Melhor R² após GridSearchCV')
plt.xlim(0, 1)  # Definir limite do eixo x para melhor visualização
plt.show()
