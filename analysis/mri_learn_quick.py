""" MRILearn Quick """

from os.path import join
import h5py
import numpy as np
import pandas as pd
from datetime import datetime 
from copy import deepcopy
import statsmodels.api as sm
from joblib import Parallel, delayed

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, get_scorer, make_scorer, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from confounds import *

def sensitivity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp/(tp+fn)

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn/(tn+fp) 

def shuffle_y(y, groups=None):
    """ Permute a vector or array y along axis 0.

    Args:
        y (numpy array): The to-be-permuted array. The array will be permuted
            along axis 0.  
        groups (str): If set, the permutations of y will only happen within members
            of the same group. Is a string indicating a confound in the conf_dict.

    Returns:
        y_permuted: The permuted array y_permuted.
    """    
    if groups is None:
        indices = np.random.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = (groups == group)
            indices[this_mask] = np.random.permutation(indices[this_mask])
    return y[indices]

def _flatten(X):
    img_size = X[0].shape
    X = X.reshape([X.shape[0], np.prod(img_size)])
    return(X)


class MRILearn:
    def __init__(self, p):
        """ Define parameters for the MRILearn object.

        Args:
            p (dict): Dictionary containing parameters with key-value pairs:
                "data_dir" (str): Full location of HDF5 file.
                "verbose" (int): Higher value means more output in the console.
                "n_jobs" (int): Number of cores to use in parallel. 
                "conf_list" (list): List containing strings indicating the confound
                    vectors saved in the HDF5 file. 
        """        
        self.data_dir = p["data_dir"]
        self.verbose = p["verbose"]
        self.n_jobs = p["n_jobs"]
        self.conf_list = p["conf_list"]

        # These are defined to calculate both BA and AUC later.
        self.score_func = make_scorer(balanced_accuracy_score)
        self.sc_auc = get_scorer("roc_auc")


    def load_data(self):
        """ Load the data from the HDF5 file, where the neuroimaging data is saved
            under "X", the label under "y" and the confounds under the strings 
            specified in the self.conf_list defined in __init__().
        """        
        d = h5py.File(self.data_dir, "r")
        self.X = np.array(d["X"])
        # self.X needs to be flattened as sklearn expects 2D input.
        if self.X.ndim > 2:
            self.X = _flatten(self.X)
        if self.X.ndim == 1:
            self.X = self.X.reshape(-1,1)   # sklearn expects 2D input
        self.y = np.array(d["y"])

        # Load the confounding variables into a dictionary
        if self.conf_list:
            self.conf_dict = {}
            for conf in self.conf_list:
                self.conf_dict[conf] = np.array(d[conf])


    def train_test_split(self, random_state, stratify_group=None):
        """ Split the data into a trainval set (_tv) and a test set (_test).

        Args:
            random_state (int): A random state for reproducible splits.
            stratify_group (str, optional): A string that indicates a confound.
                If a confound is selected, subjects will be stratified according 
                to confound and outcome label. Defaults to None, in which case 
                subjects are only stratified according to the outcome label.
        """        
        if stratify_group:
            stratify_group = self.conf_dict[stratify_group] + 100*self.y
        else:
            stratify_group = self.y

        trainval_idx, test_idx = train_test_split(range(len(self.X)), stratify=stratify_group, \
            random_state=random_state)
        self.X_tv = self.X[trainval_idx]
        self.y_tv = self.y[trainval_idx]
        self.X_test = self.X[test_idx]
        self.y_test = self.y[test_idx]
        self.conf_dict_tv = {}
        self.conf_dict_test = {}
        for conf in self.conf_dict.keys():
            self.conf_dict_tv[conf] = self.conf_dict[conf][trainval_idx]
            self.conf_dict_test[conf] = self.conf_dict[conf][test_idx]


    def change_input_to(self, confound, onehot=True):
        """ Change the inputs of the model to a vector containing a confound.

        Args:
            confound (str): The name of the confound which loaded in conf_dict, 
                see self.load_data().
            onehot (bool, optional): Whether to one-hot encode the new input. When 
                this is not done, linear models will underperform. Defaults to True.
        """        
        if onehot:
            self.X_tv = OneHotEncoder(sparse=False).fit_transform(self.conf_dict_tv[confound].reshape(-1, 1))
            self.X_test = OneHotEncoder(sparse=False).fit_transform(self.conf_dict_test[confound].reshape(-1, 1))
        else:
            self.X_tv = self.conf_dict_tv[confound].reshape(-1, 1)
            self.X_test = self.conf_dict_test[confound].reshape(-1, 1)
            

    def change_output_to(self, confound):
        """ Change the targets (outputs) of the classifier to a confound vector.

        Args:
            confound (str): The name of the confound which loaded in conf_dict, 
                see self.load_data().
        """        
        self.y_tv = self.conf_dict_tv[confound]
        self.y_test = self.conf_dict_test[confound]


    def wd_transform(self, pipeline):
        """ Transform all input data self.X with a predefined pipeline. Note this
            only transforms self.X and not self.X_tv and self.X_test: so run this 
            before self.train_test_split().

        Args:
            pipeline (sklearn object): a sklearn pipeline that has .fit() and 
                .transform() that does not need self.y. 
        """        
        self.X = pipeline.fit_transform(self.X)


    def run(self, pipe, grid, cv_splitter=None):
        """ The main function to run the classification. 

        Args:
            pipe (sklearn Pipeline): a pipeline containing preprocessing steps 
                and the classification model. 
            grid (dict): [description]
            cv_splitter (sklearn splitter object): Data splitter that defines
                the folds in the GridSearchCV. Defaults to None.

        Returns:
            results (dict): A dictionary containing classification metrics for the 
                best parameters.
            best_params (dict): The best parameters found by grid search.
        """        

        gs = GridSearchCV(pipe, grid, cv=cv_splitter, scoring=self.score_func,\
            return_train_score=True, verbose=self.verbose, n_jobs=self.n_jobs, pre_dispatch="2*n_jobs")
        gs = gs.fit(self.X_tv, self.y_tv)
        train_score = np.mean(gs.cv_results_["mean_train_score"])
        valid_score = gs.best_score_
        test_score = gs.score(self.X_test, self.y_test)

        # Add AUC, SE and SP if label is binary
        if np.array_equal(self.y_test, self.y_test.astype(bool)):
            y_pred = gs.predict(self.X_test)
            sensitivity = sensitivity_score(y_true=self.y_test, y_pred=y_pred)
            specificity = specificity_score(y_true=self.y_test, y_pred=y_pred)
            roc_auc = self.sc_auc(gs.best_estimator_, self.X_test, self.y_test)
        else:
            sensitivity, specificity, roc_auc = np.nan, np.nan, np.nan

        results = {
            "train_score" : train_score, 
            "valid_score" : valid_score, 
            "test_score" : test_score,
            "roc_auc" : roc_auc, 
            "sensitivity" : sensitivity,
            "specificity" : specificity
        }
        self.estimator = gs.best_estimator_
        return results, gs.best_params_


    def one_permutation(self, i):
        """ Run one permutation without PBCC results.

        Args:
            i (int): Index of permutation.

        Returns:
            pt_score (float): Balanced accuracy score from permuted samples.
            pt_score_auc (float): ROC AUC score from permuted samples.
        """
        X_tv_permuted = shuffle_y(self.X_tv, groups=self.groups_tv)
        X_test_permuted = shuffle_y(self.X_test, groups=self.groups_test)

        # Perform a grid search on the permuted data 
        gs = GridSearchCV(self.pipe, self.grid, cv=self.cv_splitter, scoring=self.score_func,\
            return_train_score=True, verbose=self.verbose, n_jobs=1)
        gs = gs.fit(X_tv_permuted, self.y_tv)
        pt_score = gs.score(X_test_permuted, self.y_test)

        # Only return AUC when y is binary
        if np.array_equal(self.y_test, self.y_test.astype(bool)):
            pt_score_auc = self.sc_auc(gs.best_estimator_, X_test_permuted, self.y_test)
        else:
            pt_score_auc = np.nan

        return pt_score, pt_score_auc


    def permutation_test(self, pipe, grid, n_permutations=100, groups_tv=None, groups_test=None, cv_splitter=None):
        """ Function to perform a permutation test without calculating prediction-based 
            confound correction statistics.

        Args:
            pipe (sklearn Pipeline): a pipeline containing preprocessing steps 
                and the classification model. 
            grid (dict): [description]
            cv_splitter (sklearn splitter object): Data splitter that defines
                the folds in the GridSearchCV. Defaults to None.
            n_permutations (int, optional): Number of permutations to perform. Defaults to 100.
            groups_tv (str, optional): String indicating confound. If specified, samples 
                in the trainval set are only permuted within the specified confound category. 
                Defaults to None.
            groups_test (str, optional): String indicating confound. If specified, samples 
                in the test set are only permuted within the specified confound category. 
                Defaults to None.
        Returns:
            results (dict): A dictionary containing permutation scores.
        """        

        # These need to be defined within the class for self.one_permutation().
        self.groups_tv, self.groups_test = groups_tv, groups_test
        self.pipe, self.grid, self.cv_splitter = pipe, grid, cv_splitter
        
        model_name = type(pipe["model"]).__name__
        print("This is model {}".format(model_name))

        # Run the permtest parallel on self.n_jobs cores.
        res = Parallel(n_jobs=self.n_jobs)(delayed(self.one_permutation)(i) for i in range(n_permutations))
        res = np.array(res)

        results = {
            "permutation_scores" : res[:,0],
            "permutation_scores_auc" : res[:,1]
        }
        return(results)


    def one_permutation_pbc(self, i):
        """ Run one permutation with PBCC results.

        Args:
            i (int): Index of permutation.

        Returns:
            pt_score (float): Balanced accuracy score from permuted samples.
            pt_score_auc (float): ROC AUC score from permuted samples.
            d2_* (float): D2 scores for fitting the PBCC models with different
                independent variables *. See function pbcc().
        """        

        # The data labels are shuffled rather than the labels as the relationship 
        # between confound and label should be kept, see Dinga et al., 2020.
        X_tv_permuted = shuffle_y(self.X_tv, groups=self.groups_tv)
        X_test_permuted = shuffle_y(self.X_test, groups=self.groups_test)

        # Perform a grid search on the permuted data 
        gs = GridSearchCV(self.pipe, self.grid, cv=self.cv_splitter, scoring=self.score_func,\
            return_train_score=True, verbose=self.verbose, n_jobs=1)
        gs = gs.fit(X_tv_permuted, self.y_tv)
        pt_score = gs.score(X_test_permuted, self.y_test)

        # If the label is binary, add AUC data
        if np.array_equal(self.y_test, self.y_test.astype(bool)):
            pt_score_auc = self.sc_auc(gs.best_estimator_, X_test_permuted, self.y_test)
        else:
            pt_score_auc = np.nan

        # Extract D2 scores
        d2_pred, d2_conf, d2_conf_pred = pbcc(gs.best_estimator_, X_test_permuted, self.y_test, self.conf_dict_test["s"], self.conf_dict_test["c"])

        return pt_score, pt_score_auc, d2_pred, d2_conf, d2_conf_pred


    def permutation_test_pbc(self, pipe, grid, n_permutations=1000, groups_tv=None, groups_test=None, cv_splitter=None):
        """ Function to perform a permutation test WITH calculating prediction-based 
            confound correction statistics. Currently only functional for confounds sex
            and site.

        Args:
            pipe (sklearn Pipeline): a pipeline containing preprocessing steps 
                and the classification model. 
            grid (dict): [description]
            cv_splitter (sklearn splitter object): Data splitter that defines
                the folds in the GridSearchCV. Defaults to None.
            n_permutations (int, optional): Number of permutations to perform. Defaults to 100.
            groups_tv (str, optional): String indicating confound. If specified, samples 
                in the trainval set are only permuted within the specified confound category. 
                Defaults to None.
            groups_test (str, optional): String indicating confound. If specified, samples 
                in the test set are only permuted within the specified confound category. 
                Defaults to None.
        Returns:
            results (dict): A dictionary containing permutation scores and the PBCC scores for
            the permuted samples.
        """        
        
        # These need to be defined for the self.one_permutation_pbc() function.
        self.groups_tv, self.groups_test = groups_tv, groups_test
        self.pipe, self.grid, self.cv_splitter = pipe, grid, cv_splitter
        
        model_name = type(pipe["model"]).__name__
        print("This is model {}".format(model_name))

        # Run the permtest parallel on self.n_jobs cores.
        res = Parallel(n_jobs=self.n_jobs)(delayed(self.one_permutation_pbc)(i) for i in range(n_permutations))
        res = np.array(res)

        results = {
            "permutation_scores" : res[:,0], 
            "permutation_scores_auc" : res[:,1],
            "d2_pred" : res[:,2],
            "d2_conf": res[:,3],  
            "d2_conf_pred": res[:,4]
        }
        return(results)


def pbcc(estimator, X, y_true, confound1, confound2):
        """ Calculate D2 scores for the Prediction-Based Confound Control by Dinga
            et al., 2020. 
        Args:
            estimator (sklearn object): A trained model or pipeline.
            X (numpy array): The input data of the test set.
            y_true (numpy array or list): The targets of the test set.
            confound1 (numpy array or list): The vector containing values for confound1.
            confound2 (numpy array or list): The vector containing values for confound2.
        Returns:
            d2_pred: explained var of logit model with only predictions as independent var.
            d2_conf: explained var of logit model with only confounds as independent var.
            d2_conf_pred: explained var of logit model with predictions+confounds as independent vars.
        """    

        # TODO hardcoded: this only works with sex as confound1 and site as confound2.

        p = estimator.predict(X)
        # Multiclass categorical confounds (such as site) should be OneHotEncoded to avoid
        # any falsely assumed ordinal relationship between categories
        confound2 = OneHotEncoder(sparse=False).fit_transform(confound2.reshape(-1,1))
        cs = np.column_stack([confound1, confound2])
        pcs = np.column_stack([p, cs])
        
        # Add intercepts
        p = sm.add_constant(p, prepend=False)
        cs = sm.add_constant(cs, prepend=False)
        pcs = sm.add_constant(pcs, prepend=False)

        # Fit models with different independent variables and extract D^2 = prsquared
        d2_pred = sm.Logit(y_true, p).fit(method='bfgs').prsquared 
        d2_conf = sm.Logit(y_true, cs).fit(method='bfgs').prsquared 
        d2_conf_pred = sm.Logit(y_true, pcs).fit(method='bfgs').prsquared 
        
        return d2_pred, d2_conf, d2_conf_pred 
