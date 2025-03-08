import sklearn.metrics
import numpy as np
import scipy.stats

model_alias = {'DNABERT': 'dnabert', 
          'DNBT-3UTR-RNA': 'dnabert-3utr-2e', 
          'DNABERT2': 'dnabert2', 
          'DNABERT2-ZOO': 'dnabert2-zoo', 
          'DNBT2-3UTR-RNA': 'dnabert2-3utr-2e',
          'NT-MS-v2-100M': 'ntrans-v2-100m',
          'NT-3UTR-RNA': 'ntrans-v2-100m-3utr-2e',
          'STSP-3UTR-RNA': 'stspace-3utr-2e', 
          'STSP-3UTR-RNA-SA': 'stspace-spaw-3utr-2e',
          'STSP-3UTR-DNA': 'stspace-3utr-DNA',
          'STSP-3UTR-DNA-SA': 'stspace-spaw-3utr-DNA',
          'STSP-3UTR-RNA-HS': 'stspace-3utr-hs',
          '5-mers Siegel et al., 2022':'5mers',
          'Griesemer et al., 2021': 'griesemer',
          'k-mer': '3K',
          'PhyloP-100way': 'PhyloP-100way',
          'PhyloP-241way': 'PhyloP-241way',
          'CADD-1.7': 'CADD-1.7',
          'Zoo-AL':'zoo-al',
          'StateSpace-HS':'stspace-3utr-hs',

        }

model_bar_colors = {'DNABERT': 'seagreen', 
                    'DNBT-3UTR-RNA': 'palegreen', 
                    'DNABERT2': 'seagreen', 
                    'DNABERT2-ZOO': 'seagreen', 
                    'DNBT2-3UTR-RNA': 'palegreen',
                    'NT-MS-v2-100M': 'seagreen',
                    'NT-3UTR-RNA': 'palegreen',
                    'STSP-3UTR-RNA': 'palegreen', 
                    'STSP-3UTR-RNA-SA': 'palegreen',
                    'STSP-3UTR-DNA': 'palegreen',
                    'STSP-3UTR-DNA-SA': 'palegreen',
                    'STSP-3UTR-RNA-HS': 'palegreen', 
                    '5-mers Siegel et al., 2022':'palegreen',
                    'Griesemer et al., 2021':'palegreen',
                    'Zoo-AL': 'seagreen',
                    'CADD-1.7':'seagreen',
                    'PhyloP-100way':'seagreen',
                    'PhyloP-241way':'seagreen',
                    'k-mer':'palegreen',
                    'BC3MS*':"#0072B2",
                    'Saluki human':'tomato',
                    'BC3MS':'mediumturquoise',}

model_colors = {
    'DNABERT': '#E6194B',  
    'DNBT-3UTR-RNA': '#3CB44B',  
    'DNABERT2': '#FFE119',  
    'DNABERT2-ZOO': '#4363D8',  
    'DNBT2-3UTR-RNA': '#F58231',  
    'NT-MS-v2-100M': '#911EB4',  
    'NT-3UTR-RNA': '#46F0F0',  
    'STSP-3UTR-RNA': '#F032E6',  
    'STSP-3UTR-RNA-SA': '#BCF60C',  
    'STSP-3UTR-DNA': '#FABEBE',  
    'STSP-3UTR-DNA-SA': '#008080',  
    'STSP-3UTR-RNA-HS': '#E6BEFF',  
    'CADD': '#9A6324',  
    '13-mer': '#FFFAC8',  
    'PhyloP-241way': '#800000',  
    'PhyloP-100way': '#AAFFC3',  
    'Zoo-AL': '#808000',  
    'CADD-1.7': '#FFD8B1'
}

#model_colors = {'DNABERT': '#D55E00', 
#                'DNBT-3UTR-RNA': '#FF9F00', 
#                'DNABERT2': '#3F3B6C', 
#                'DNABERT2-ZOO': '#0072B2', 
#                'DNBT2-3UTR-RNA': '#958CFF',
#                'NT-MS-v2-100M': '#008173',
#                'NT-3UTR-RNA': '#00DEA2',
#                'STSP-3UTR-RNA': '#0900FF', 
#                'STSP-3UTR-RNA-SA': '#59C3FF',
#                'STSP-3UTR-DNA': '#59C3FF',
#                'STSP-3UTR-DNA-SA': '#5970ff',
#                'STSP-3UTR-RNA-HS': 'k', 
#                'CADD':'#CC79A7',
#                '13-mer':"#CC79A7",
#                'PhyloP-241way':"#ffd373",
#                'PhyloP-100way':"#e69f00",
#                'Zoo-AL': '#7f4124',
#                'CADD-1.7':"#C5B500",}

dna_models = ['DNABERT','DNABERT2','DNABERT2-ZOO','NT-MS-v2-100M', 'STSP-3UTR-DNA', 'STSP-3UTR-DNA-SA','CADD-1.7','PhyloP-100way','PhyloP-241way']

rna_models = ['Zoo-AL','DNBT-3UTR-RNA', 'NT-3UTR-RNA', 'DNBT2-3UTR-RNA','STSP-3UTR-RNA', 'STSP-3UTR-RNA-SA', 'STSP-3UTR-RNA-HS']

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
        return min(a[1], b[1]) - max(a[0], b[0])>=0

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