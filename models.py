import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import torch.nn as nn
import torch
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from config import DEFAULT_PARAMS


class Ols:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = sm.add_constant(X)
        self.model = sm.OLS(y, X).fit()

    def predict(self, X):
        X = sm.add_constant(X, has_constant='add')
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.params
    
    def get_summary(self):
        return self.model.summary()
    
class Linear:
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.coef_
    
    def get_summary(self):
        return self.model.intercept_
    
class RidgeModel:
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_params(self):
        return self.model.coef_
    
    def get_summary(self):
        return self.model.intercept_
    
class RandomForest:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def get_importances(self, X):
        self.importances = self.model.feature_importances_
        print(len(self.importances))
        print(len(X))
        print(len(set(X)))
        self.importance = {}
        for i, each in enumerate(X):
            self.importance[each] = self.importances[i]
        self.importance = sorted(self.importance.items(), key=lambda x: x[1], reverse=True)
        print(len(self.importance))
        return self.importance
    

class Xgb:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def get_importances(self, X):
        self.importances = self.model.feature_importances_
        self.importance = {}
        for i, each in enumerate(X):
            self.importance[each] = self.importances[i]
        # sort
        self.importance = sorted(self.importance.items(), key=lambda x: x[1], reverse=True)
        return self.importance
    

class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyModel, self).__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, hidden_size*2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size*2, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_classes)
        # )
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.fc(x)
        return out
    
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, num_classes, epochs=100):
        self.model = MyModel(input_size, hidden_size, num_classes)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = epochs
        
    def fit(self, X, y):
        for epoch in range(self.epochs):
            inputs = torch.from_numpy(X).float()
            labels = torch.from_numpy(y).float().reshape(-1, 1)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

    def predict(self, X):
        inputs = torch.from_numpy(X).float()
        return self.model(inputs).detach().numpy().reshape(-1)
    
    def get_params(self):
        return self.model.parameters()
    
    def get_summary(self):
        return self.model
    

def get_model_class(model_name):
    if model_name == 'ols':
        return Ols
    elif model_name == 'linear':
        return Linear
    elif model_name == 'ridge':
        return RidgeModel
    elif model_name == 'rf':
        return RandomForest
    elif model_name == 'xgb':
        return Xgb
    elif model_name == 'net':
        return NeuralNetwork
    elif model_name == 'svr':
        return SVR
    elif model_name == 'krr':
        return KernelRidge
    else:
        return None
    
def get_model(model_name):
    model_class = get_model_class(model_name)
    if model_class is None:
        return None
    return model_class(**DEFAULT_PARAMS[model_name])