import sklearn.metrics
import numpy as np
import scipy.stats

class dotdict(dict):
    '''
    dot.notation access to dictionary attributes
    '''
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def pearson_r(x, y, compute_CI=False):
    '''
    Compute Pearson r coefficient between samples x and y and 95% confidence interval
    '''
    x = np.array(x)
    y = np.array(y)
    pearson_r = scipy.stats.pearsonr(x,y)
    if not compute_CI:
        return pearson_r[0]
    ci_95 = pearson_r.confidence_interval()
    ci_95 = np.diff(ci_95)[0]/2
    pearson_r = pearson_r[0]
    return (pearson_r,ci_95)

def get_best_models(series):
    '''
    Return indices of the best models with overlapping confidence interval
    Input: a series with  values (score,score_CI) and model names in indices
    '''

    def is_overlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))>0

    best_models = []

    best_auc, best_auc_err =  series.sort_values().iloc[-1]

    for model, (auc, auc_err) in series.items():
            if is_overlap((best_auc-best_auc_err,best_auc+best_auc_err),(auc-auc_err,auc+auc_err)):
                best_models.append(model)

    return best_models

def highlight_ns(x, best_models):
    #mark the best model and models with insignificant difference with the best model bold
    column_name = x.name
    return ['font-weight: bold' if model in list(best_models[column_name]) else ''
                for model in x.index]
    
#def pearson_r(x,y):
#    '''
#    Compute Pearson r coefficient between samples x and y
#    '''
#    x = np.array(x)
#    y = np.array(y)
#    cov_xy = np.mean((x - x.mean()) * (y - y.mean()))
#    r = cov_xy / (x.std() * y.std())
#    return r
    
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