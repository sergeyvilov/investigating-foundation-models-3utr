import numpy as np

from torch import nn
from torch.optim import AdamW
import torch

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from misc import pearson_r

class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(1024,128,32,), 
                 p_dropout=0.5, weight_decay=0, lr=1e-3,
                batch_size = 1024, N_epochs = 300):
                
        self.hidden_layer_sizes = hidden_layer_sizes
        self.p_dropout = p_dropout
        self.weight_decay = weight_decay
        self.lr = lr
        
        self.batch_size = batch_size
        self.N_epochs = N_epochs
        
        self.loss_fn = nn.BCELoss()

    def init_model(self):
        
        #define model architecture
        model_layers = []
        for layer_size in self.hidden_layer_sizes:
            model_layers.extend((nn.LazyLinear(layer_size), nn.Dropout(self.p_dropout), nn.ReLU(),))
        model_layers.append(nn.LazyLinear(1))
        model_layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*model_layers)
        
        #initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        
    def scorer(self, y_true, y_pred):
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
        return roc_auc_score(y_true, y_pred)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):

        X_train, y_train = check_X_y(X_train, y_train)
        
        self.classes_ = unique_labels(y_train)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        if X_val is not None:
            X_val, y_val = check_X_y(X_val, y_val)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

        self.init_model()

        batches_per_epoch = int(np.ceil(len(X_train)//self.batch_size))

        self.history_ = [] #history for train and validation metrics

        for epoch in range(self.N_epochs):
            self.model.train()
            train_score, val_score = 0, None
            for batch_idx in range(batches_per_epoch):
                # take a batch
                X_batch = X_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                y_batch = y_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                # forward pass
                y_pred = self.model(X_batch).reshape(-1)
                loss = self.loss_fn(y_pred, y_batch)
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()
                train_score += self.scorer(y_batch, y_pred)/batches_per_epoch
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(X_val).reshape(-1)
                val_score = self.scorer(y_val, y_pred)
            self.history_.append((train_score,val_score))
            
        return self
        
    def predict_proba(self, X):

        check_is_fitted(self)

        X = check_array(X)

        X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1)
            
        y_pred = y_pred.detach().numpy()
        y_pred = np.vstack((1-y_pred,y_pred)).T

        return y_pred
        
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def score(self, X, y_true):

        X, y_true = check_X_y(X, y_true)

        X = torch.tensor(X, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1)
        
        return self.scorer(y_true,y_pred)
        
class MLPRegressor(BaseEstimator,RegressorMixin):
    def __init__(self, hidden_layer_sizes=(1024,128,32,), 
                 p_dropout=0.5, weight_decay=0, lr=0.0005,
                batch_size = 1024, N_epochs = 500):
                
        self.hidden_layer_sizes = hidden_layer_sizes
        self.p_dropout = p_dropout
        self.weight_decay = weight_decay
        self.lr = lr
        
        self.batch_size = batch_size
        self.N_epochs = N_epochs
        
        self.loss_fn = nn.MSELoss()

    def init_model(self):
        
        #define model architecture
        model_layers = []
        for layer_size in self.hidden_layer_sizes:
            model_layers.extend((nn.LazyLinear(layer_size), nn.Dropout(self.p_dropout), nn.ReLU(),))
        model_layers.append(nn.LazyLinear(1))
        self.model = nn.Sequential(*model_layers)
        
        #initialize optimizer
        self.optimizer = AdamW(self.model.parameters(), weight_decay=self.weight_decay, lr=self.lr)

    def scorer(self, y_true, y_pred):
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
        #print(pearson_r(y_true, y_pred),r2_score(list(y_true), list(y_pred)))

        return pearson_r(y_true, y_pred) 
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):

        X_train, y_train = check_X_y(X_train, y_train)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        #print(self.__dict__)

        if X_val is not None:
            X_val, y_val = check_X_y(X_val, y_val)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

        self.init_model()

        batches_per_epoch = int(np.ceil(len(X_train)//self.batch_size))

        self.history_ = [] #history for train and validation metrics

        for epoch in range(self.N_epochs):
            self.model.train()
            train_score, val_score = 0, None
            for batch_idx in range(batches_per_epoch):
                # take a batch
                X_batch = X_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                y_batch = y_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                # forward pass
                y_pred = self.model(X_batch).reshape(-1)
                loss = self.loss_fn(y_pred, y_batch)
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()
                train_score +=  self.scorer(y_batch, y_pred)/batches_per_epoch
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    y_pred = self.model(X_val).reshape(-1)
                val_score = self.scorer(y_val, y_pred)
            self.history_.append((train_score,val_score))
            
        return self
        
    def predict(self, X):

        check_is_fitted(self)

        X = check_array(X)
        
        X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1)
            
        y_pred = y_pred.detach().numpy()
        
        return y_pred
        
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def score(self, X, y_true):
        
        X, y_true = check_X_y(X, y_true)

        X = torch.tensor(X, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1)
        
        return self.scorer(y_true,y_pred)
        