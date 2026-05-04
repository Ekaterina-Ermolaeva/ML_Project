import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor

X_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\House_prices\data_train_final_hp.csv')
y_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\House_prices\target_train_final_hp.csv')

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
       test_size=0.2,
       random_state=42
)

#расчет всех метрик для каждой модели
def get_model_scores(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    rmsle_train = round(np.sqrt(mean_squared_log_error(y_train, y_pred_train)), 4)
    rmsle_test = round(np.sqrt(mean_squared_log_error(y_train, y_pred_train)), 4)
    mae = round(mean_absolute_error(y_test, y_pred_test), 4)
    mse = round(mean_squared_error(y_test, y_pred_test), 4)
    r2 = round(r2_score(y_test, y_pred_test), 4)
    rmsle_cv = round(np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_log_error", cv=5)).mean(), 4)
    return rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv

#датафрейм для результатов моделей
def add_result(results_df, model_name, RMSLE_test, RMSLE_train, MAE, MSE, R2, cross_val):
    new_row = pd.DataFrame([[model_name, RMSLE_test, RMSLE_train, MAE, MSE, R2, cross_val]], 
              columns=["Model","RMSLE_test", "RMSLE_train", "MAE","MSE","R2", "Cross-Val"])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

results = pd.DataFrame(columns=["Model","RMSLE_test", "RMSLE_train", "MAE","MSE","R2", "Cross-Val"])

with open(r'C:\Users\kate1\Desktop\ML\Project\House_prices\config_house_prices.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 1. LinearRegression - базовая линейная регрессия для предсказания цен на жилье
lin = LinearRegression()
lin.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(lin, X_train, y_train, X_test, y_test)
results = add_result(results, lin, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 2. Ridge - линейная регрессия с L2-регуляризацией (alpha=1000)
ridge = Ridge(alpha=1000, random_state=42)
ridge.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(ridge, X_train, y_train, X_test, y_test)
results = add_result(results, ridge, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 3. Lasso - линейная регрессия с L1-регуляризацией (alpha=0.001)
lasso = Lasso(alpha=0.001, random_state=42)
lasso.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(lasso, X_train, y_train, X_test, y_test)
results = add_result(results, lasso, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 4. ElasticNet - линейная регрессия с комбинацией L1 и L2 регуляризации (alpha=1000, l1_ratio=0.1)
elasticnet = ElasticNet(alpha=1000, l1_ratio=0.1, random_state=42)
elasticnet.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(elasticnet, X_train, y_train, X_test, y_test)
results = add_result(results, elasticnet, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 5. KNN Regressor - предсказание на основе 9 ближайших соседей с манхэттенской метрикой
knn = KNeighborsRegressor(n_neighbors=9, weights='uniform', metric='minkowski', p=1, leaf_size=20, algorithm='auto')
knn.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(knn, X_train, y_train, X_test, y_test)
results = add_result(results, knn, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 6. Decision Tree Regressor - дерево решений без ограничения глубины, min_samples_split=10, min_samples_leaf=4
dt = DecisionTreeRegressor(max_depth=None, min_samples_split=10, min_samples_leaf=4, max_features=None, criterion='squared_error')
dt.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(dt, X_train, y_train, X_test, y_test)
results = add_result(results, dt, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 7. Random Forest Regressor - ансамбль из 100 деревьев глубиной 15, с выборкой 80% данных
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_samples=0.8, ccp_alpha=0, random_state=42)
rf.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(rf, X_train, y_train, X_test, y_test)
results = add_result(results, rf, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 8. XGBoost Regressor - градиентный бустинг с глубокими деревьями (depth=9) и высокой регуляризацией
xgboost = XGBRegressor(n_estimators=400, max_depth=9, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0.1, random_state=42)
xgboost.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(xgboost, X_train, y_train, X_test, y_test)
results = add_result(results, xgboost, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 9. CatBoost Regressor - градиентный бустинг с автоматической работой с категориальными признаками
catboost = CatBoostRegressor(iterations=400, depth=3, learning_rate=0.1, subsample=0.9, colsample_bylevel=0.9, l2_leaf_reg=5, random_strength=0.2, verbose=False, random_state=42)
catboost.fit(X_train, y_train)
rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(catboost, X_train, y_train, X_test, y_test)
results = add_result(results, catboost, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 10. PyTorch Neural Network - полносвязная нейронная сеть с Dropout для регуляризации (32->16->1)
bool_cols = X_train.select_dtypes(include=['bool']).columns
X_train[bool_cols] = X_train[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

X_train_nn = torch.Tensor(X_train.values)
X_test_nn = torch.Tensor(X_test.values)
y_train_nn = torch.Tensor(y_train.values).reshape(-1, 1)
y_test_nn = torch.Tensor(y_test.values).reshape(-1, 1)

class NN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

input_size = X_train.shape[1]

best_rmsle = float('inf')
best_lr = None
best_model = None
best_results = {}

model = NN(input_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    
epochs = 100
    
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_nn)
    loss = criterion(outputs, y_train_nn)
    loss.backward()
    optimizer.step()
    
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_nn).numpy().flatten()
    y_train_pred = model(X_train_nn).numpy().flatten()
    
rmsle_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
rmsle_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2 = r2_score(y_test, y_test_pred)
    
if rmsle_test < best_rmsle:
    best_rmsle = rmsle_test
    best_lr = 0.001
    best_model = model
    best_results = {
        'rmsle_test': rmsle_test,
        'rmsle_train': rmsle_train,
        'r2': r2
    }
    
for epoch in range(100):
    best_model.train()
    optimizer.zero_grad()
    outputs = best_model(X_train_nn)
    loss = criterion(outputs, y_train_nn)
    loss.backward()
    optimizer.step()

best_model.eval()
with torch.no_grad():
    y_test_pred = best_model(X_test_nn).numpy().flatten()
    y_train_pred = best_model(X_train_nn).numpy().flatten()

rmsle_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
rmsle_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
rmsle_cv = rmsle_test

model_name = "PyTorch_NN_Final"
results = add_result(results, model_name, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 11. Voting Ensemble - ансамбль из 5 XGBoost моделей с разными гиперпараметрами (голосование усреднением)
models = []
params_list = [
    {'n_estimators': 1200, 'max_depth': 6, 'learning_rate': 0.007, 'subsample': 0.6, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'min_child_weight': 10, 'gamma': 0.7},
    {'n_estimators': 1500, 'max_depth': 5, 'learning_rate': 0.005, 'subsample': 0.55, 'colsample_bytree': 0.55, 'colsample_bylevel': 0.55, 'reg_alpha': 1.5, 'reg_lambda': 3.0, 'min_child_weight': 15, 'gamma': 1.0},
    {'n_estimators': 1800, 'max_depth': 5, 'learning_rate': 0.003, 'subsample': 0.5, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 20, 'gamma': 1.5},
    {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.65, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 2.5, 'min_child_weight': 12, 'gamma': 0.8},
    {'n_estimators': 2000, 'max_depth': 7, 'learning_rate': 0.002, 'subsample': 0.6, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 4.0, 'min_child_weight': 18, 'gamma': 1.2}
]

for i, params in enumerate(params_list, 1):
    model = XGBRegressor(**params, random_state=42+i, n_jobs=-1, verbosity=0, objective='reg:squarederror', eval_metric='rmse')
    model.fit(X_train, y_train)
    models.append((f'xgb_{i}', model))

voting_model = VotingRegressor(estimators=models)
voting_model.fit(X_train, y_train)

rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(voting_model, X_train, y_train, X_test, y_test)
results = add_result(results, "VotingEnsemble_5XGB", rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

# 12. Stacking Ensemble - стекинг 5 XGBoost моделей с Ridge регрессией в качестве мета-модели
X_train_stack, X_val_stack, y_train_stack, y_val_stack = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

base_models = []
for i, params in enumerate(params_list, 1):
    model = XGBRegressor(**params, random_state=42+i, n_jobs=-1, verbosity=0, objective='reg:squarederror', eval_metric='rmse')
    model.fit(X_train_stack, y_train_stack)
    base_models.append((f'xgb_{i}', model))

meta_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0, random_state=42),
    cv=5
)
meta_model.fit(X_train, y_train)

rmsle_test_meta, rmsle_train_meta, mae_meta, mse_meta, r2_meta, rmsle_cv_meta = get_model_scores(meta_model, X_train, y_train, X_test, y_test)
results = add_result(results, "StackingEnsemble_5XGB_Ridge", rmsle_test_meta, rmsle_train_meta, mae_meta, mse_meta, r2_meta, rmsle_cv_meta)

if __name__ == "__main__":
    print(results)


def run_house_prices_main():
    X_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\House_prices\data_train_final_hp.csv')
    y_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\House_prices\target_train_final_hp.csv')

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42
    )

    #расчет всех метрик для каждой модели
    def get_model_scores(model, X_train, y_train, X_test, y_test):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        rmsle_train = round(np.sqrt(mean_squared_log_error(y_train, y_pred_train)), 4)
        rmsle_test = round(np.sqrt(mean_squared_log_error(y_train, y_pred_train)), 4)
        mae = round(mean_absolute_error(y_test, y_pred_test), 4)
        mse = round(mean_squared_error(y_test, y_pred_test), 4)
        r2 = round(r2_score(y_test, y_pred_test), 4)
        rmsle_cv = round(np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_log_error", cv=5)).mean(), 4)
        return rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv

    #датафрейм для результатов моделей
    def add_result(results_df, model_name, RMSLE_test, RMSLE_train, MAE, MSE, R2, cross_val):
        new_row = pd.DataFrame([[model_name, RMSLE_test, RMSLE_train, MAE, MSE, R2, cross_val]], 
                columns=["Model","RMSLE_test", "RMSLE_train", "MAE","MSE","R2", "Cross-Val"])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        return results_df

    results = pd.DataFrame(columns=["Model","RMSLE_test", "RMSLE_train", "MAE","MSE","R2", "Cross-Val"])

    with open(r'C:\Users\kate1\Desktop\ML\Project\House_prices\config_house_prices.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 1. LinearRegression - базовая линейная регрессия для предсказания цен на жилье
    lin = LinearRegression()
    lin.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(lin, X_train, y_train, X_test, y_test)
    results = add_result(results, lin, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 2. Ridge - линейная регрессия с L2-регуляризацией (alpha=1000)
    ridge = Ridge(alpha=1000, random_state=42)
    ridge.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(ridge, X_train, y_train, X_test, y_test)
    results = add_result(results, ridge, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 3. Lasso - линейная регрессия с L1-регуляризацией (alpha=0.001)
    lasso = Lasso(alpha=0.001, random_state=42)
    lasso.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(lasso, X_train, y_train, X_test, y_test)
    results = add_result(results, lasso, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 4. ElasticNet - линейная регрессия с комбинацией L1 и L2 регуляризации (alpha=1000, l1_ratio=0.1)
    elasticnet = ElasticNet(alpha=1000, l1_ratio=0.1, random_state=42)
    elasticnet.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(elasticnet, X_train, y_train, X_test, y_test)
    results = add_result(results, elasticnet, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 5. KNN Regressor - предсказание на основе 9 ближайших соседей с манхэттенской метрикой
    knn = KNeighborsRegressor(n_neighbors=9, weights='uniform', metric='minkowski', p=1, leaf_size=20, algorithm='auto')
    knn.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(knn, X_train, y_train, X_test, y_test)
    results = add_result(results, knn, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 6. Decision Tree Regressor - дерево решений без ограничения глубины, min_samples_split=10, min_samples_leaf=4
    dt = DecisionTreeRegressor(max_depth=None, min_samples_split=10, min_samples_leaf=4, max_features=None, criterion='squared_error')
    dt.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(dt, X_train, y_train, X_test, y_test)
    results = add_result(results, dt, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 7. Random Forest Regressor - ансамбль из 100 деревьев глубиной 15, с выборкой 80% данных
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_samples=0.8, ccp_alpha=0, random_state=42)
    rf.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(rf, X_train, y_train, X_test, y_test)
    results = add_result(results, rf, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 8. XGBoost Regressor - градиентный бустинг с глубокими деревьями (depth=9) и высокой регуляризацией
    xgboost = XGBRegressor(n_estimators=400, max_depth=9, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, gamma=0.1, random_state=42)
    xgboost.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(xgboost, X_train, y_train, X_test, y_test)
    results = add_result(results, xgboost, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 9. CatBoost Regressor - градиентный бустинг с автоматической работой с категориальными признаками
    catboost = CatBoostRegressor(iterations=400, depth=3, learning_rate=0.1, subsample=0.9, colsample_bylevel=0.9, l2_leaf_reg=5, random_strength=0.2, verbose=False, random_state=42)
    catboost.fit(X_train, y_train)
    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(catboost, X_train, y_train, X_test, y_test)
    results = add_result(results, catboost, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 10. PyTorch Neural Network - полносвязная нейронная сеть с Dropout для регуляризации (32->16->1)
    bool_cols = X_train.select_dtypes(include=['bool']).columns
    X_train[bool_cols] = X_train[bool_cols].astype(int)
    X_test[bool_cols] = X_test[bool_cols].astype(int)

    X_train_nn = torch.Tensor(X_train.values)
    X_test_nn = torch.Tensor(X_test.values)
    y_train_nn = torch.Tensor(y_train.values).reshape(-1, 1)
    y_test_nn = torch.Tensor(y_test.values).reshape(-1, 1)

    class NN(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        def forward(self, x):
            return self.net(x)

    input_size = X_train.shape[1]

    best_rmsle = float('inf')
    best_lr = None
    best_model = None
    best_results = {}

    model = NN(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    epochs = 100
        
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_nn)
        loss = criterion(outputs, y_train_nn)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test_nn).numpy().flatten()
        y_train_pred = model(X_train_nn).numpy().flatten()
        
    rmsle_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmsle_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2 = r2_score(y_test, y_test_pred)
        
    if rmsle_test < best_rmsle:
        best_rmsle = rmsle_test
        best_lr = 0.001
        best_model = model
        best_results = {
            'rmsle_test': rmsle_test,
            'rmsle_train': rmsle_train,
            'r2': r2
        }
        
    for epoch in range(100):
        best_model.train()
        optimizer.zero_grad()
        outputs = best_model(X_train_nn)
        loss = criterion(outputs, y_train_nn)
        loss.backward()
        optimizer.step()

    best_model.eval()
    with torch.no_grad():
        y_test_pred = best_model(X_test_nn).numpy().flatten()
        y_train_pred = best_model(X_train_nn).numpy().flatten()

    rmsle_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    rmsle_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)
    rmsle_cv = rmsle_test

    model_name = "PyTorch_NN_Final"
    results = add_result(results, model_name, rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 11. Voting Ensemble - ансамбль из 5 XGBoost моделей с разными гиперпараметрами (голосование усреднением)
    models = []
    params_list = [
        {'n_estimators': 1200, 'max_depth': 6, 'learning_rate': 0.007, 'subsample': 0.6, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6, 'reg_alpha': 1.0, 'reg_lambda': 2.0, 'min_child_weight': 10, 'gamma': 0.7},
        {'n_estimators': 1500, 'max_depth': 5, 'learning_rate': 0.005, 'subsample': 0.55, 'colsample_bytree': 0.55, 'colsample_bylevel': 0.55, 'reg_alpha': 1.5, 'reg_lambda': 3.0, 'min_child_weight': 15, 'gamma': 1.0},
        {'n_estimators': 1800, 'max_depth': 5, 'learning_rate': 0.003, 'subsample': 0.5, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5, 'reg_alpha': 2.0, 'reg_lambda': 5.0, 'min_child_weight': 20, 'gamma': 1.5},
        {'n_estimators': 1000, 'max_depth': 6, 'learning_rate': 0.01, 'subsample': 0.65, 'colsample_bytree': 0.5, 'colsample_bylevel': 0.5, 'reg_alpha': 0.5, 'reg_lambda': 2.5, 'min_child_weight': 12, 'gamma': 0.8},
        {'n_estimators': 2000, 'max_depth': 7, 'learning_rate': 0.002, 'subsample': 0.6, 'colsample_bytree': 0.6, 'colsample_bylevel': 0.6, 'reg_alpha': 3.0, 'reg_lambda': 4.0, 'min_child_weight': 18, 'gamma': 1.2}
    ]

    for i, params in enumerate(params_list, 1):
        model = XGBRegressor(**params, random_state=42+i, n_jobs=-1, verbosity=0, objective='reg:squarederror', eval_metric='rmse')
        model.fit(X_train, y_train)
        models.append((f'xgb_{i}', model))

    voting_model = VotingRegressor(estimators=models)
    voting_model.fit(X_train, y_train)

    rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv = get_model_scores(voting_model, X_train, y_train, X_test, y_test)
    results = add_result(results, "VotingEnsemble_5XGB", rmsle_test, rmsle_train, mae, mse, r2, rmsle_cv)

    # 12. Stacking Ensemble - стекинг 5 XGBoost моделей с Ridge регрессией в качестве мета-модели
    X_train_stack, X_val_stack, y_train_stack, y_val_stack = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    base_models = []
    for i, params in enumerate(params_list, 1):
        model = XGBRegressor(**params, random_state=42+i, n_jobs=-1, verbosity=0, objective='reg:squarederror', eval_metric='rmse')
        model.fit(X_train_stack, y_train_stack)
        base_models.append((f'xgb_{i}', model))

    meta_model = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0, random_state=42),
        cv=5
    )
    meta_model.fit(X_train, y_train)

    rmsle_test_meta, rmsle_train_meta, mae_meta, mse_meta, r2_meta, rmsle_cv_meta = get_model_scores(meta_model, X_train, y_train, X_test, y_test)
    results = add_result(results, "StackingEnsemble_5XGB_Ridge", rmsle_test_meta, rmsle_train_meta, mae_meta, mse_meta, r2_meta, rmsle_cv_meta)

    return results