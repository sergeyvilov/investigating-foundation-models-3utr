import numpy as np
from torch import nn
from torch.optim import AdamW
import torch

def pearson_r(x,y):
    '''
    Compute Pearson r coefficient between samples x and y
    '''
    x = np.array(x)
    y = np.array(y)
    cov_xy = np.mean((x - x.mean()) * (y - y.mean()))
    r = cov_xy / (x.std() * y.std())
    return r

class MLPRegressor():
    def __init__(self, hidden_layer_sizes=(1024,128,32,), 
                 p_dropout=0.5, weight_decay=0, lr=0.0005,
                batch_size = 1024, N_epochs = 500, **kwargs):
                
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

        self.history = [] #history for train and validation metrics

    def scorer(self, y_true, y_pred):
        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy()
        return pearson_r(y_true, y_pred) ** 2 
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)

        self.init_model()

        batches_per_epoch = int(np.ceil(len(X_train)//self.batch_size))
                
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
            self.history.append((train_score,val_score))
        
    def predict(self, X):

        X = torch.tensor(X, dtype=torch.float32)

        self.model.eval()
        
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1)
            
        y_pred = y_pred.detach().numpy()
        return y_pred
        
    def set_params(self, **kwargs):
        
        self.__dict__.update(kwargs)
        
    def score(self, X, y_true):

        X = torch.tensor(X, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X).reshape(-1)
        
        return self.scorer(y_true,y_pred)