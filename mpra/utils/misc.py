import sklearn.metrics

class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class GroupBaggedRegressor():
    
    def __init__(self, clf, n_estimators=10):
        
        clf_type = type(clf)
        clf_params = clf.get_params()
        
        self.estimators = [clf_type(**clf_params) for clf_idx in range(n_estimators)]
        self.n_estimators = n_estimators
        
    def fit(self,X,y,groups):
    
        for round_idx in range(self.n_estimators):
            
            np.random.seed(round_idx)

            sampled_groups = np.random.choice(groups, size=len(groups), replace=True)
            groups_filter = [True if group in sampled_groups else False for group in groups]

            X_round = X[groups_filter]
            y_round = y[groups_filter]
            
            self.estimators[round_idx].fit(X_round,y_round)
            
    def predict(self, X):
        
        self.preds = np.zeros((X.shape[0],self.n_estimators))
        
        for estimator_idx in range(self.n_estimators):
            
            self.preds[:,estimator_idx] = self.estimators[estimator_idx].predict(X)
            
        av_preds = self.preds.mean(axis=1)
            
        return av_preds
    
    def score(self, X, y_true):

        y_pred = self.predict(X)
        return sklearn.metrics.r2_score(y_true,y_pred)
    
    
from torch import nn
from torch.optim import AdamW
import torch
import numpy as np

class MLPRegressor():
    def __init__(self, hidden_layer_sizes=(64,32,16,), 
                 p_dropout=0, weight_decay=0, lr=0.0005,
                batch_size = 1000, N_epochs = 500, **kwargs):
                
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
        y_true = y_true.detach().numpy()[:,0]
        y_pred = y_pred.detach().numpy()[:,0]
        return sklearn.metrics.r2_score(y_true, y_pred) ** 2 
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        if not X_val is None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        
        self.init_model()
        
        self.history = [] #history for train and validation metrics
        
        batches_per_epoch = int(np.ceil(len(X_train)//self.batch_size))
        
        self.model.train()
        
        for epoch in range(self.N_epochs):
            train_score, val_score = 0, 0
            for batch_idx in range(batches_per_epoch):
                # take a batch
                X_batch = X_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                y_batch = y_train[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
                # forward pass
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                # update weights
                self.optimizer.step()
                train_score += self.scorer(y_batch, y_pred)/batches_per_epoch
            if not X_val is None:
                self.model.eval()
                y_pred = self.model(X_val)
                val_score = self.scorer(y_val, y_pred)
            self.history.append((train_score,val_score))
        
    def predict(self, X):
        
        self.model.eval()
        
        X = torch.tensor(X, dtype=torch.float32)
        
        y_pred = self.model(X)
        y_pred = y_pred.detach().numpy()[:,0]
        return y_pred
        
    def set_params(self, **kwargs):
        
        self.__dict__.update(kwargs)
        
    def score(self, X, y_true):
        
        X = torch.tensor(X, dtype=torch.float32)
        y_true = torch.tensor(y_true, dtype=torch.float32).reshape(-1, 1)
        
        y_pred = self.predict(X)
        return self.scorer(y_true,y_pred)
    
#M = MLPRegressor(p_dropout=0.1,weight_decay=1e-3,batch_size=1000)
#M.set_params(N_epochs=500)
#M.fit(X_train,y_train,X_test,y_test)

class NelderMeadCV():
    
    def __init__(self, clf, start_point, cv_splits=3):
        
        self.cv_splits = cv_splits
        self.clf = clf
        self.x0 = start_point
        
    def optimize(self, X, y, groups):
        
        def objective(args):
            
            C, gamma, epsilon = args
            
            C = 2.**round(np.log2(C))
            gamma = 2.**round(np.log2(gamma))
            epsilon = 10.**round(np.log10(epsilon))

            
            #self.clf.set_params({'C':C, 'gamma':gamma, 'epsilon':epsilon})
            self.clf.set_params(C=C, gamma=gamma, epsilon=epsilon)
            
            pipe = sklearn.pipeline.make_pipeline(sklearn.preprocessing.StandardScaler(),self.clf)
            
            cv_score = sklearn.model_selection.cross_val_score(pipe, X, y, groups=groups, 
                 cv = sklearn.model_selection.GroupKFold(n_splits= self.cv_splits), scoring = 'neg_mean_absolute_error', n_jobs = -1)
        
            return -cv_score.mean()
        
        res = scipy.optimize.minimize(objective, x0=self.x0, method='Nelder-Mead', 
                                options={'disp':True,'maxiter':300,'return_all':True})
        
        return res