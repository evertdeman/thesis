""" This file is meant to perform classification and should be run in the command
line. 
"""


import pandas as pd
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from datetime import datetime 
from copy import deepcopy
import time

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.naive_bayes import GaussianNB

# Local imports
sys.path.insert(1, "..")
from mri_learn_quick import MRILearn, pbcc
from confounds import *

# Define settings for the experiment 
DATA_DIR = "/ritter/share/data/IMAGEN"
N_TRIALS = 10
N_SPLITS = 5
# Number of permutati
# ons for standard PT. Set to None if no PT should be done.
N_PERMUTATIONS = 100
# Number of permutations for within-block PT. Set to None if no WBPT should be done.
N_WBPERMUTATIONS = None
save_coefs = False

# The random states are loaded from a pre-made file to keep computations reproducible.
random_states = np.load("random_states.npy")[:N_TRIALS]

# Parameters for the MRILearn object
params = {
    "verbose" : 3, 
    "n_jobs" : 5, 
    "conf_list" : ["c", "s", "group"]
}

pipes = [
    Pipeline([
        ("varth", VarianceThreshold()),
        ("feature_selection", SelectKBest()),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2500))
    ]),
    Pipeline([
        ("varth", VarianceThreshold()),
        ("feature_selection", SelectKBest()),
        ("scale", MinMaxScaler(feature_range=(-1,1))),
        ("model", LinearSVC(max_iter=2500))

    ]),
    Pipeline([
        ("varth", VarianceThreshold()),
        ("feature_selection", SelectKBest()),
        ("scale", MinMaxScaler(feature_range=(-1,1))),
        ("model", SVC(kernel="rbf"))
    ]),
    Pipeline([
        ("feature_selection", SelectKBest()), # Only added for compatibility, k is always "all"!
        ("model", GradientBoostingClassifier(max_depth=5, n_estimators=100, max_features="sqrt"))
    ]), 
    Pipeline([
        ("varth", VarianceThreshold()),
        ("feature_selection", SelectKBest()),
        ("scale", StandardScaler()),
        ("model", GaussianNB())
    ])
]
grids = [
    {
        "model__C" : [1e-8, 1e-5, 1e-3, 1, 1e3, 1e5, 1e8]
    },
    {
        "model__C" : [1e-8, 1e-5, 1e-3, 1, 1e3, 1e5, 1e8]
    },
    {
        "model__C" : [1e-8, 1e-5, 1e-3, 1, 1e3, 1e5, 1e8],
        "model__gamma" : [1e-5, 1e-3, 0.01, 0.1, 1]
    },
    {}, 
    {}
]

# Here you could select which pipelines you want to run

pipes = [pipes[i] for i in [1,3]]
grids = [grids[i] for i in [1,3]]

#pipes = [pipes[0,1,3]]
#grids = [grids[0,1,3]]

# Here you can select which HDF5 files you want to include in analysis. 
# Each entry should be (file_name, k_features).
files = [
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/T1w/T1w_FU2-FU2_n789_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/T1w/T1w_FU2-FU2_n403_sex1_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/T1w/T1w_FU2-FU2_n386_sex0_z0.525_d0.h5"), 10000)
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/T1w/T1w_BL-FU2_n507_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/T1w/T1w_BL-FU2_n269_sex1_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/T1w/T1w_BL-FU2_n238_sex0_z0.525_d0.h5"), 10000)

    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/dti-FA/dti-FA_FU2-FU2_n789_z0.525_d0.h5"),  10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/dti-FA/dti-FA_FU2-FU2_n403_sex1_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/dti-FA/dti-FA_FU2-FU2_n386_sex0_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/dti-FA/dti-FA_BL-FU2_n269_sex1_z0.525_d0.h5"), 10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/dti-FA/dti-FA_BL-FU2_n238_sex0_z0.525_d0.h5"),10000),
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/dti-FA/dti-FA_BL-FU2_n507_z0.525_d0.h5"), 10000)

    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/fs-stats/fs-stats_BL-FU2_n507.h5"), "all")
    (join(DATA_DIR, "h5files/ESPAD19a_01_56/fs-stats/fs-stats_FU2-FU2_n789.h5"), "all")
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/fs-stats/fs-stats_FU2-FU2_n403_sex1.h5"), "all")
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/fs-stats/fs-stats_FU2-FU2_n386_sex0.h5"), "all")
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/fs-stats/fs-stats_BL-FU2_n269_sex1.h5"), "all")
    #(join(DATA_DIR, "h5files/ESPAD19a_01_56/fs-stats/fs-stats_BL-FU2_n238_sex0.h5"), "all")
]


# The experiments dict is a grid where you can select which technique and io you want to 
experiments = {
    #"technique" : ["baseline"],
    "technique" : ["baseline", "cb", "wdcr", "cvcr", "cbcvcr"],
    "io" : [
        ("X", "y")
        #("X", "s"), 
        #("X", "c"),
        #("s", "y"),
        #("c", "y")
    ], 
    "trial" : range(N_TRIALS), 
    "pipesgrids" : zip(pipes, grids)
}


for f, k_features in files:

    params["data_dir"] = f

    # Create the folder in which to save the results
    SAVE_DIR = "results/{}/{}".format(os.path.basename(params["data_dir"]).replace(".h5",""),\
        datetime.now().strftime("%Y%m%d-%H%M"))
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # DataFrames to store results in 
    df = pd.DataFrame(columns=["model", "technique", "trial", "io", "train_score", "valid_score", "test_score", "roc_auc", "sensitivity","specificity"])
    df_pt = pd.DataFrame(columns=["model", "technique", "trial", "io", "permutation_scores", "permutation_scores_auc"])
    df_ptwb = pd.DataFrame(columns=["model", "technique", "trial", "io", "permutation_scores", "permutation_scores_auc"])


    if save_coefs:
        coefs_matrix = np.empty([N_TRIALS, 1050624])
    else:
        pass


    # Go through all combinations of experiments
    for p in ParameterGrid(experiments):
        
        technique = p["technique"]
        i, o = p["io"]
        io = "{}-{}".format(i,o)
        pipe = p["pipesgrids"][0]
        grid = p["pipesgrids"][1]
        trial = p["trial"]
        model_name = type(pipe["model"]).__name__
        random_state = random_states[trial]


        if (i == "X" and model_name != "GradientBoostingClassifier"):
            grid["feature_selection__k"] = [k_features]
        else:
            grid["feature_selection__k"] = ["all"]

        # Some parameter combinations do not work
        # CVCR or CBCVCR does not work with models other than LR
        if ((technique == "cvcr" or technique == "cbcvcr") and model_name != "LogisticRegression"):
            print("Skipping CVCR because model != LR.")
            continue
        # When taking only male or female data, sex is not a confound
        if ("sex" in f and (io == "X-s" or io == "s-y")):
            print("Skipping X-s and s-y because sex is the same among subjects.")
            continue
        # CBCVCR can only be done on combined data
        if ("sex" in f and technique == "cbcvcr"):
            print("Skipping CBCVCR because sex is the same among subjects.")
            continue
        

        # What happens during the confound correction techniques: 
        if technique == "baseline":
            m = MRILearn(params)
            m.load_data()
            m.train_test_split(random_state=random_state)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        # Whole Dataset Confound Regression both confounds
        elif (technique == "wdcr") and (i == "X"):
            m = MRILearn(params)
            m.load_data()
            m.X = np.concatenate([m.X, m.conf_dict["group"].reshape(-1,1)], axis=1)
            crc = ConfoundRegressorCategoricalX()
            m.wd_transform(crc)
            m.train_test_split(random_state=random_state)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        elif (technique == "wdcr") and (i != "X"):
            m = MRILearn(params)
            m.load_data()
            m.train_test_split(random_state=random_state)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        # CrossValidated Confound Regression both confounds
        elif (technique == "cvcr") and (i == "X"):
            m = MRILearn(params)
            m.load_data()
            m.X = np.concatenate([m.X, m.conf_dict["group"].reshape(-1,1)], axis=1)
            crc = ("crc", ConfoundRegressorCategoricalX())
            pipe = deepcopy(pipe)
            pipe.steps.insert(0, crc)
            m.train_test_split(random_state=random_state)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        elif (technique == "cvcr") and (i != "X"):
            m = MRILearn(params)
            m.load_data()
            m.train_test_split(random_state=random_state)
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
        # CounterBalancing both confounds
        elif technique == "cb":
            m = MRILearn(params)
            m.load_data()

            # Counterbalance for both sex and site, which is "group"
            cb = CounterBalance(m.conf_dict["group"], random_state)
            m.X = cb.fit_transform(m.X, m.y)
            m.y = cb.transform(m.y)
            m.conf_dict["c"] = cb.transform(m.conf_dict["c"])
            m.conf_dict["s"] = cb.transform(m.conf_dict["s"])
            m.conf_dict["group"] = cb.transform(m.conf_dict["group"])

            # Ensure groups are stratified within splits
            m.train_test_split(random_state=random_state, stratify_group="group")
            stratify_groups = m.conf_dict_tv["group"] + 100 * m.y_tv
            skf = tuple(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state).split(m.X_tv, stratify_groups))
        # CB-CVCR: counterbalancing sex and CVCR for site
        elif (technique == "cbcvcr") and (i == "X"):
            m = MRILearn(params)
            m.load_data()
            # Counterbalance for sex ONLY 
            cb = CounterBalance(m.conf_dict["s"], random_state)
            m.X = cb.fit_transform(m.X, m.y)
            m.y = cb.transform(m.y)
            m.conf_dict["c"] = cb.transform(m.conf_dict["c"])
            m.conf_dict["s"] = cb.transform(m.conf_dict["s"])
            m.conf_dict["group"] = cb.transform(m.conf_dict["group"])

            # The CVCR part for site ONLY
            m.X = np.concatenate([m.X, m.conf_dict["c"].reshape(-1,1)], axis=1)
            crc = ("crc", ConfoundRegressorCategoricalX())
            pipe = deepcopy(pipe)
            pipe.steps.insert(0, crc)

            # Ensure groups are stratified within splits
            m.train_test_split(random_state=random_state, stratify_group="s")
            stratify_groups = m.conf_dict_tv["s"] + 100 * m.y_tv
            skf = tuple(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state).split(m.X_tv, stratify_groups))
        elif (technique == "cbcvcr") and (i != "X"):
            m = MRILearn(params)
            m.load_data()

            cb = CounterBalance(m.conf_dict["s"], random_state)
            m.X = cb.fit_transform(m.X, m.y)
            m.y = cb.transform(m.y)
            m.conf_dict["c"] = cb.transform(m.conf_dict["c"])
            m.conf_dict["s"] = cb.transform(m.conf_dict["s"])
            m.conf_dict["group"] = cb.transform(m.conf_dict["group"])

            # Ensure groups are stratified within splits
            m.train_test_split(random_state=random_state, stratify_group="s")
            stratify_groups = m.conf_dict_tv["s"] + 100 * m.y_tv
            skf = tuple(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state).split(m.X_tv, stratify_groups))
        elif technique == "cb_sex":
            m = MRILearn(params)
            m.load_data()

            # Counterbalance for both sex and site, which is "group"
            cb = CounterBalance(m.conf_dict["s"], random_state)
            m.X = cb.fit_transform(m.X, m.y)
            m.y = cb.transform(m.y)
            m.conf_dict["c"] = cb.transform(m.conf_dict["c"])
            m.conf_dict["s"] = cb.transform(m.conf_dict["s"])
            m.conf_dict["group"] = cb.transform(m.conf_dict["group"])

            # Ensure groups are stratified within splits
            m.train_test_split(random_state=random_state, stratify_group="s")
            stratify_groups = m.conf_dict_tv["s"] + 100 * m.y_tv
            skf = tuple(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state).split(m.X_tv, stratify_groups))

        
        print("D")

        # Change input and output if so defined 
        if i != "X":
            m.change_input_to(i, onehot=True)
        if o != "y":
            m.change_output_to(o)

        start = time.time()
        # Run the actual classification
        run, best_params = m.run(pipe, grid, cv_splitter=skf)

        if save_coefs:
            coefs = m.estimator.named_steps["model"].coef_
            coefs = m.estimator.named_steps["feature_selection"].inverse_transform(coefs)
            coefs = m.estimator.named_steps["varth"].inverse_transform(coefs)
            coefs = coefs.flatten()
            coefs_matrix[trial, :] = coefs
            np.save(join(SAVE_DIR, "coefs"), coefs_matrix)


        # Calculate PBCC D2 values
        if (technique=="baseline") and (i == "X" and o =="y"):
            d2_pred, d2_conf, d2_conf_pred = pbcc(m.estimator, m.X_test, m.y_test, m.conf_dict_test["s"], m.conf_dict_test["c"])
        else:
            d2_pred, d2_conf, d2_conf_pred = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Append results
        result = {
            "model" : model_name,
            "technique" : technique,
            "trial" : trial,
            "io" : "{}-{}".format(i,o), 
            **run,
            "runtime" : time.time() - start, 
            **best_params, 
            "d2_pred" : d2_pred, 
            "d2_conf" : d2_conf, 
            "d2_conf_pred" : d2_conf_pred
        }

        print(result)
        df = df.append(result, ignore_index=True)
        # Save classification results
        df.to_csv(join(SAVE_DIR, "run.csv"))

        
        # RUN PERMUTATION TEST 
        if N_PERMUTATIONS:
            # In this case a permtest with PBCC values is done
            if (technique=="baseline") and (i == "X" and o =="y"):
                start = time.time()
                pt = m.permutation_test_pbc(pipe, grid, n_permutations=N_PERMUTATIONS, cv_splitter=skf)
                pt_result = {
                    "model" : [model_name]*N_PERMUTATIONS,
                    "technique" : [technique]*N_PERMUTATIONS,
                    "trial" : [trial]*N_PERMUTATIONS,
                    **pt,
                    "io" : ["{}-{}".format(i,o)]*N_PERMUTATIONS, 
                    "runtime" : time.time() - start
                }
                print(result)
                df_pt = pd.concat([df_pt, pd.DataFrame(pt_result)])
                df_pt.to_csv(join(SAVE_DIR, "pt.csv"))
            # In other cases, PBCC values are not computed
            else:
                start = time.time()
                pt = m.permutation_test(pipe, grid, n_permutations=N_PERMUTATIONS, cv_splitter=skf)
                pt_result = {
                    "model" : [model_name]*N_PERMUTATIONS,
                    "technique" : [technique]*N_PERMUTATIONS,
                    "trial" : [trial]*N_PERMUTATIONS,
                    **pt,
                    "io" : ["{}-{}".format(i,o)]*N_PERMUTATIONS, 
                    "runtime" : time.time() - start
                }
                print(result)
                df_pt = pd.concat([df_pt, pd.DataFrame(pt_result)])
                df_pt.to_csv(join(SAVE_DIR, "pt.csv"))

        # Add Within-Block PT.  
        if N_WBPERMUTATIONS:
            if (technique=="baseline") and (i == "X" and o =="y"):
                # WBPT only works if there is only one confound site, so sex is all one.
                if len(np.unique(m.conf_dict["s"])) == 1:
                    start = time.time()
                    ptwb = m.permutation_test_pbc(pipe, grid, n_permutations=N_WBPERMUTATIONS, groups_tv=m.conf_dict_tv["c"], \
                        groups_test=m.conf_dict_test["c"], cv_splitter=skf)
                    ptwb_result = {
                        "model" : [model_name]*N_WBPERMUTATIONS,
                        "technique" : [technique]*N_WBPERMUTATIONS,
                        "trial" : [trial]*N_WBPERMUTATIONS,
                        **ptwb,
                        "io" : ["{}-{}".format(i,o)]*N_WBPERMUTATIONS, 
                        "runtime" : time.time() - start
                    }
                    print(result)
                    df_ptwb = pd.concat([df_ptwb, pd.DataFrame(ptwb_result)])
                    df_ptwb.to_csv(join(SAVE_DIR, "ptwb.csv"))


