import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import ttest_ind, pearsonr
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict, \
    RepeatedStratifiedKFold, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import balanced_accuracy_score, get_scorer
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import BaseCrossValidator
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from itertools import combinations
from copy import deepcopy
import multiprocessing as mp



class ConfoundRegressorCategoricalX(BaseEstimator, TransformerMixin):
    """This is class to regress out categorical confounds. It is inspired by
    the work of Snoek et al., 2019, but is different in a few ways.
    """    
    
    def __init__(self):
        self.weights_ = None


    def fit(self, X, y=None):
        """[summary]
        Args:
            X (np.array): A numpy array of shape (n_samples, n_features+1). 
                The last column holds the categorical confound.
            y : Does nothing. Defaults to None.
        Returns:
            class: Returns the class.
        """        

        # Omit those features with nonzero variance (so if all are 0 they stay that way)
        self.confound_fit = X[:,-1]
        X = X[:,0:-1]
        
        # Calculate means of each feature from the total 
        self.group_means = {}
        total_mean = X.mean(0)
        self.groups = np.unique(self.confound_fit.ravel())
        for g in self.groups:
            self.group_means[g] = X[self.confound_fit == g].mean(axis=0) - total_mean
        return self

    def transform(self, X):
        """[summary]
        Args:
            X (np.array): An array of shape (n_samples, n_features+1).
                The last column hold the categorical confound.
        Returns:
            X_transformed (np.array): An array of shape (n_samples, n_features).
        """        

        self.confound_transform = X[:,-1]
        X = X[:,0:-1]

        # Time to subtract the category mean from X
        X_new = np.zeros_like(X)
        
        for g in self.groups:
            X_new[self.confound_transform == g] = X[self.confound_transform == g] - self.group_means[g]
        return X_new


class CounterBalance(BaseEstimator, TransformerMixin):

    def __init__(self, groups, random_state):
        self.groups = groups
        self.random_state = random_state

    def fit(self, X, y):

        df = pd.DataFrame({"idx":range(len(y)), "y":y, "groups":self.groups})

        keep_idx = pd.DataFrame()

        for g in df.groups.unique().tolist():
            dfx = df.query("groups == {}".format(g))
            min_list = dfx.y.value_counts().values
            if len(min_list) == 2:
                min_subs = min(min_list)
            else: 
                min_subs = 0
            for label in [0,1]:
                dfxx = dfx.query("y == {}".format(str(label)))
                dfxx = dfxx.reset_index()
                if len(dfxx) == min_subs:
                    keep_idx = pd.concat([keep_idx, dfxx])
                else:
                    remove_n = len(dfxx) - min_subs
                    np.random.seed(self.random_state)
                    drop_indices = np.random.choice(dfxx.index, remove_n, replace=False)
                    dfxx = dfxx.drop(drop_indices, axis=0)
                    keep_idx = pd.concat([keep_idx, dfxx])

        self.keep_indices = np.sort(keep_idx.idx.values)

        groups_with_label = self.groups[self.keep_indices] + 100*y[self.keep_indices]

        # Remove indices from groups that have less than 2 members
        gr, co = np.unique(groups_with_label, return_counts=True)
        delete_groups = gr[co<2]
        print("GROUPS {} ONLY HAVE ONE MEMBER, DELETING THEM, TOO.".format(delete_groups))

        value = []

        for i in delete_groups:
            # Get indices of to-be-deleted units 
            value.append(self.keep_indices[np.argwhere(groups_with_label == i).flatten()[0]])

        for v in value:
            self.keep_indices = np.delete(self.keep_indices, np.argwhere(self.keep_indices == v))

        
        return(self)

    def transform(self, X):
        return(X[self.keep_indices])


class SiteTransferSplit(BaseCrossValidator):
    """NB: This class ONLY works with IMAGEN test data with center 1-8 and sex 0,1"""

    def __init__(self, groups, n_splits=None):
        self.groups = groups

    def split(self, X, y, groups=None):
        X, y = indexable(X, y)

        indices = np.arange(_num_samples(X))
        q = 0
        for train_gr in combinations(range(1, 9), 6):
            train_gr = np.array(train_gr)
            test_gr = np.setdiff1d(np.arange(1,9), train_gr)
            for sex in [0,1]:
                if sex == 0:
                    train_gr_ = list(train_gr + 10)
                    test_gr_ = list(test_gr)
                elif sex == 1:
                    test_gr_ = list(test_gr + 10)
                    train_gr_ = list(train_gr)
                
                q+=1
                train_index = [i for i in range(len(self.groups)) if self.groups[i] in train_gr_]
                test_index = [i for i in range(len(self.groups)) if self.groups[i] in test_gr_]
                train_index = indices[train_index]
                test_index = indices[test_index]
                yield ((train_index, test_index))

        self.n_splits = q

    def get_n_splits(self, X, y=None, groups=None):
        return(self.n_splits)
            
                












        