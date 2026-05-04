import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier

X_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\Titanic\data_train_final_T.csv')
y_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\Titanic\target_train_final_T.csv')  

X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train,
       test_size=0.2,
       random_state=42
)

#функция для датафрейма, где будут храниться все лучшие результаты моделей
#на основе этого выберем лучшую модель в финале и построим финальный прогноз
def add_result(results_df, model_name, test_accuracy, train_accuracy, precision, recall, f1, cross_val):
    new_row = pd.DataFrame([[model_name, test_accuracy, train_accuracy, precision, recall, f1, cross_val]], 
              columns=['Model', 'Accuracy_test', 'Accuracy_train', 'Precision', 'Recall', 'F1', 'Cross-val'])
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

results = pd.DataFrame(columns=['Model', 'Accuracy_test', 'Accuracy_train', 'Precision', 'Recall', 'F1', 'Cross-val'])

#функция для расчета всех метрик для каждой модели
def get_model_scores(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    accuracy_train = round(accuracy_score(y_train, y_pred_train), 4)
    accuracy_test = round(accuracy_score(y_test, y_pred_test), 4)
    precision = round(precision_score(y_test, y_pred_test, zero_division=0), 4)
    recall = round(recall_score(y_test, y_pred_test, zero_division=0), 4)
    f1 = round(f1_score(y_test, y_pred_test, zero_division=0), 4)
    
    return accuracy_test, accuracy_train, precision, recall, f1

with open(r'C:\Users\kate1\Desktop\ML\Project\Titanic\config_titanic.yaml', 'r') as f:
    config_titanic = yaml.safe_load(f)

# 1. Логистическая регрессия - базовая линейная модель для бинарной классификации
model = LogisticRegression(**config_titanic['logistic_regression']['params']) 
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 2. KNN (K-Nearest Neighbors) - классификация на основе k ближайших соседей
model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 3. Decision Tree - дерево решений с ограничением глубины 3
model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1)
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 4. Random Forest - ансамбль деревьев решений с ограничениями на глубину и выборку
model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, max_features=0.5, max_samples=0.7, ccp_alpha=0.005, random_state=42)
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 5. CatBoost - градиентный бустинг от Яндекса с автоматической работой с категориальными признаками
model = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=100, l2_leaf_reg=3, bagging_temperature=0, verbose=False, random_state=42)
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 6. LightGBM - градиентный бустинг с высокой скоростью обучения и поддержкой больших данных
model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31, min_child_samples=20, random_state=42)
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 7. XGBoost - градиентный бустинг с регуляризацией и высокой точностью
model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, gamma=0.01, random_state=42)
cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
model.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 8. PyTorch Neural Network - полносвязная нейронная сеть с 3 слоями и Sigmoid активацией
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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.Tensor(x.values)
        return self.net(x)
    
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if not torch.is_tensor(x):
                x = torch.Tensor(x.values)
            return (self.forward(x).numpy().flatten() > 0.5).astype(int)
    
input_size = X_train.shape[1]

best_accuracy = 0
best_lr = None
best_model = None
best_results = {}

model = NN(input_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
    
epochs = 100
    
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_nn)
    loss = criterion(outputs, y_train_nn)
    loss.backward()
    optimizer.step()
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
if accuracy_test > best_accuracy:
    best_accuracy = accuracy_test
    best_lr = 0.001
    best_model = model
    best_results = {
        'accuracy_train': accuracy_train,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
model_name = "PyTorch_NN_Final"
results = add_result(results, model_name, best_accuracy, best_results['accuracy_train'], best_results['precision'], best_results['recall'], best_results['f1'], '-')

# 9. Voting Ensemble - ансамбль из 3 XGBoost моделей с голосованием и весами [1,2,1]
model1 = XGBClassifier(max_depth=3, random_state=42)
model2 = XGBClassifier(max_depth=3, random_state=42)
model3 = XGBClassifier(max_depth=4, random_state=42)

voting_clf = VotingClassifier(
    estimators=[('xgb1', model1), ('xgb2', model2), ('xgb3', model3)],
    voting='soft',
    weights=[1, 2, 1]
)

cross_val = round(np.mean(cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')), 4)
voting_clf.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(voting_clf, X_train, y_train, X_test, y_test)
results = add_result(results, voting_clf, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

# 10. Stacking Ensemble - стекинг 3 XGBoost моделей с логистической регрессией в качестве мета-модели
stacking_clf = StackingClassifier(
    estimators=[
        ('xgb1', XGBClassifier(max_depth=3, random_state=42)),
        ('xgb2', XGBClassifier(max_depth=3, random_state=42)),
        ('xgb3', XGBClassifier(max_depth=4, random_state=42))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

cross_val = round(np.mean(cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy')), 4)
stacking_clf.fit(X_train, y_train)
accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(stacking_clf, X_train, y_train, X_test, y_test)
results = add_result(results, stacking_clf, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

if __name__ == "__main__":
    print(results)


def run_titanic_main():
    X_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\Titanic\data_train_final_T.csv')
    y_train = pd.read_csv(r'C:\Users\kate1\Desktop\ML\Project\Titanic\target_train_final_T.csv')  

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=0.2,
        random_state=42
    )
    #функция для датафрейма, где будут храниться все лучшие результаты моделей
    #на основе этого выберем лучшую модель в финале и построим финальный прогноз
    def add_result(results_df, model_name, test_accuracy, train_accuracy, precision, recall, f1, cross_val):
        new_row = pd.DataFrame([[model_name, test_accuracy, train_accuracy, precision, recall, f1, cross_val]], 
                columns=['Model', 'Accuracy_test', 'Accuracy_train', 'Precision', 'Recall', 'F1', 'Cross-val'])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        return results_df

    results = pd.DataFrame(columns=['Model', 'Accuracy_test', 'Accuracy_train', 'Precision', 'Recall', 'F1', 'Cross-val'])

    #функция для расчета всех метрик для каждой модели
    def get_model_scores(model, X_train, y_train, X_test, y_test):
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        accuracy_train = round(accuracy_score(y_train, y_pred_train), 4)
        accuracy_test = round(accuracy_score(y_test, y_pred_test), 4)
        precision = round(precision_score(y_test, y_pred_test, zero_division=0), 4)
        recall = round(recall_score(y_test, y_pred_test, zero_division=0), 4)
        f1 = round(f1_score(y_test, y_pred_test, zero_division=0), 4)
        
        return accuracy_test, accuracy_train, precision, recall, f1

    with open(r'C:\Users\kate1\Desktop\ML\Project\Titanic\config_titanic.yaml', 'r') as f:
        config_titanic = yaml.safe_load(f)

    # 1. Логистическая регрессия - базовая линейная модель для бинарной классификации
    model = LogisticRegression(**config_titanic['logistic_regression']['params']) 
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 2. KNN (K-Nearest Neighbors) - классификация на основе k ближайших соседей
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 3. Decision Tree - дерево решений с ограничением глубины 3
    model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, min_samples_leaf=1)
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 4. Random Forest - ансамбль деревьев решений с ограничениями на глубину и выборку
    model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, min_samples_leaf=4, max_features=0.5, max_samples=0.7, ccp_alpha=0.005, random_state=42)
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 5. CatBoost - градиентный бустинг от Яндекса с автоматической работой с категориальными признаками
    model = CatBoostClassifier(depth=4, learning_rate=0.1, iterations=100, l2_leaf_reg=3, bagging_temperature=0, verbose=False, random_state=42)
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 6. LightGBM - градиентный бустинг с высокой скоростью обучения и поддержкой больших данных
    model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, num_leaves=31, min_child_samples=20, random_state=42)
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 7. XGBoost - градиентный бустинг с регуляризацией и высокой точностью
    model = XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, gamma=0.01, random_state=42)
    cross_val = round(np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')), 4)
    model.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    results = add_result(results, model, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 8. PyTorch Neural Network - полносвязная нейронная сеть с 3 слоями и Sigmoid активацией
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
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            if not torch.is_tensor(x):
                x = torch.Tensor(x.values)
            return self.net(x)
        
        def predict(self, x):
            self.eval()
            with torch.no_grad():
                if not torch.is_tensor(x):
                    x = torch.Tensor(x.values)
                return (self.forward(x).numpy().flatten() > 0.5).astype(int)
        
    input_size = X_train.shape[1]

    best_accuracy = 0
    best_lr = None
    best_model = None
    best_results = {}

    model = NN(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    epochs = 100
        
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_nn)
        loss = criterion(outputs, y_train_nn)
        loss.backward()
        optimizer.step()
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(model, X_train, y_train, X_test, y_test)
    if accuracy_test > best_accuracy:
        best_accuracy = accuracy_test
        best_lr = 0.001
        best_model = model
        best_results = {
            'accuracy_train': accuracy_train,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    model_name = "PyTorch_NN_Final"
    results = add_result(results, model_name, best_accuracy, best_results['accuracy_train'], best_results['precision'], best_results['recall'], best_results['f1'], '-')

    # 9. Voting Ensemble - ансамбль из 3 XGBoost моделей с голосованием и весами [1,2,1]
    model1 = XGBClassifier(max_depth=3, random_state=42)
    model2 = XGBClassifier(max_depth=3, random_state=42)
    model3 = XGBClassifier(max_depth=4, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[('xgb1', model1), ('xgb2', model2), ('xgb3', model3)],
        voting='soft',
        weights=[1, 2, 1]
    )

    cross_val = round(np.mean(cross_val_score(voting_clf, X_train, y_train, cv=5, scoring='accuracy')), 4)
    voting_clf.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(voting_clf, X_train, y_train, X_test, y_test)
    results = add_result(results, voting_clf, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    # 10. Stacking Ensemble - стекинг 3 XGBoost моделей с логистической регрессией в качестве мета-модели
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgb1', XGBClassifier(max_depth=3, random_state=42)),
            ('xgb2', XGBClassifier(max_depth=3, random_state=42)),
            ('xgb3', XGBClassifier(max_depth=4, random_state=42))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

    cross_val = round(np.mean(cross_val_score(stacking_clf, X_train, y_train, cv=5, scoring='accuracy')), 4)
    stacking_clf.fit(X_train, y_train)
    accuracy_test, accuracy_train, precision, recall, f1 = get_model_scores(stacking_clf, X_train, y_train, X_test, y_test)
    results = add_result(results, stacking_clf, accuracy_test, accuracy_train, precision, recall, f1, cross_val)

    return results
    