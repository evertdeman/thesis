import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scikits.bootstrap import ci
from os.path import join

def return_asterisks(pval):
    """Return significance stars

    Args:
        pval (float): p-value [0-1]

    Returns:
        ast (str): significance stars or "n.s."
    """    
    if pval < 0.001:
        ast = '***'
    elif pval < 0.01:
        ast = '**'
    elif pval < 0.05:
        ast = '*'
    else:
        ast = 'n.s.'
    return ast


def pbcc_results(df_run, df_pt=None):

    dfr = pd.DataFrame(columns=["model", "technique", "io"])

    df_run = df_run.query("technique == 'baseline' and io == 'X-y'")
    if isinstance(df_pt, pd.DataFrame):
        df_pt = df_pt.query("technique == 'baseline' and io == 'X-y'")

    technique = "baseline"
    io = "X-y"

    for setting in df_run.setting.unique():
        for inmod in df_run.inmod.unique():
            for model in df_run.model.unique():
                dfx = df_run.query("((setting == '{}' and inmod == '{}') and model == '{}')".format(setting, inmod, model))
                if isinstance(df_pt, pd.DataFrame):
                    dfx_pt = df_pt.query("((setting == '{}' and inmod == '{}') and model == '{}')".format(setting, inmod, model))

            
                d2_pred = dfx.d2_pred.values
                d2_conf = dfx.d2_conf.values
                d2_conf_pred = dfx.d2_conf_pred.values

                Delta_p = d2_conf_pred - d2_conf 
                Delta_c = d2_conf_pred - d2_pred 
                shared = d2_conf_pred - d2_conf - d2_pred

                ci_res = ci(Delta_p)

                Delta_p = Delta_p.mean()
                d2_pred = d2_pred.mean()
                d2_conf = d2_conf.mean()
                Delta_c = Delta_c.mean()
                shared = shared.mean()

                # Permtests
                if isinstance(df_pt, pd.DataFrame):
                    dfx_pt.loc[:,"Delta_p"] = dfx_pt.loc[:,"d2_conf_pred"] - dfx_pt.loc[:,"d2_conf"]
                    C = sum(Delta_p <= dfx_pt["Delta_p"].values)
                    pvalue_Dp = (C+1)/(len(dfx_pt["Delta_p"].values) + 1)
                else:
                    pvalue_Dp = np.nan 

                dfr = dfr.append({
                    "model" : model, 
                    "technique" : technique, 
                    "io" : io, 
                    "inmod" : inmod,
                    "setting": setting,
                    "Delta_p" : Delta_p, 
                    "ci_low" : ci_res[0], 
                    "ci_high" : ci_res[1], 
                    "pvalue_Dp" : pvalue_Dp,
                    "d2_pred" : d2_pred, 
                    "d2_conf" : d2_conf, 
                    "Delta_c" : Delta_c, 
                    "shared" : shared
                }, ignore_index=True)
    return dfr  

def make_dfs(dir_id, dir_pr, pt=True):
    
    inmod = dir_id.split("/")[2].split("_")[0]
    
    df_run = pd.DataFrame()
    df_run = df_run.append(pd.read_csv(join("..", dir_pr, "run.csv")).assign(setting="pr").assign(inmod=inmod))
    df_run = df_run.append(pd.read_csv(join("..", dir_id, "run.csv")).assign(setting="id").assign(inmod=inmod))

    if pt:
        df_pt = pd.DataFrame()
        df_pt = df_pt.append(pd.read_csv(join("..", dir_pr, "pt.csv")).assign(setting="pr").assign(inmod=inmod))
        df_pt = df_pt.append(pd.read_csv(join("..", dir_id, "pt.csv")).assign(setting="id").assign(inmod=inmod))
        return df_run, df_pt
    else:
        return df_run


def parse_table(df_run, df_pt=None):

    dfr = pd.DataFrame(columns=["model", "technique", "io", "ba_mean", "ba_std", "ba_ci_low", "ba_ci_high", "ba_pvalue", "sens", "spec"])

    for setting in df_run.setting.unique():
        for inmod in df_run.inmod.unique():
            for model in df_run.model.unique():
                for technique in df_run.technique.unique():
                    for io in df_run.io.unique():
                        dfx = df_run.query("((setting == '{}' and inmod == '{}') and (model == '{}' and technique == '{}')) and io == '{}'".format(setting, inmod, model, technique, io))
                        if isinstance(df_pt, pd.DataFrame):
                            dfx_pt = df_pt.query("((setting == '{}' and inmod == '{}') and (model == '{}' and technique == '{}')) and io == '{}'".format(setting, inmod, model, technique, io))


                        test_scores = dfx.test_score.values
                        mean_test_score = np.mean(test_scores)

                        ci_res = ci(test_scores)
                        
                        if isinstance(df_pt, pd.DataFrame):
                            permutation_scores = dfx_pt.permutation_scores.values
                            if len(permutation_scores) == 0:
                                pvalue = np.nan
                            else:
                                C = sum(mean_test_score <= permutation_scores)
                                pvalue = (C+1)/(len(permutation_scores) + 1)
                        else:
                            pvalue = np.nan

                        if io != "X-c":
                            test_sores_auc = dfx.roc_auc
                            mean_test_score_auc = np.mean(test_sores_auc)
                            ci_res_auc = ci(test_sores_auc)
                            if isinstance(df_pt, pd.DataFrame):
                                permutation_scores_auc = dfx_pt.permutation_scores_auc.values
                                if len(permutation_scores_auc) == 0:
                                    pvalue_auc = np.nan
                                else:
                                    C = sum(mean_test_score_auc <= permutation_scores_auc)
                                    pvalue_auc = (C+1)/(len(permutation_scores_auc) + 1)
                            else:
                                pvalue_auc = np.nan
                        else:
                            mean_test_score_auc, pvalue_auc = np.nan, np.nan

                        dfr = dfr.append({
                            "model" : model, 
                            "technique" : technique, 
                            "io" : io, 
                            "inmod" : inmod,
                            "setting": setting,
                            "ba_mean" : mean_test_score, 
                            "ba_std" : np.std(test_scores), 
                            "ba_ci_low" : ci_res[0], 
                            "ba_ci_high" : ci_res[1], 
                            "ba_pvalue" : pvalue, 
                            "sens" : np.mean(dfx.sensitivity),
                            "spec" : np.mean(dfx.specificity),
                            "auc_mean" : mean_test_score_auc, 
                            "auc_pvalue" : pvalue_auc, 
                            "auc_ci_low" : ci_res_auc[0],
                            "auc_ci_high" : ci_res_auc[1]
                        }, ignore_index=True)
    return dfr



class Visualization:

    def __init__(self, cp, dpi):
        self.c = sns.color_palette()
        self.dpi = dpi


    def plot(self, dfr, query, x, y, figsize=(6, 4), low="ba_ci_low", high="ba_ci_high", \
        dodge=0.4, pval="ba_pvalue", chance=None, xlab="balanced accuracy", yticks=None, 
        sharex=False, sharey=True, ticks=True):

        df = dfr.query(query)
        c = sns.color_palette()

        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=self.dpi, sharex=sharex, sharey=sharey)


        for k, sett in enumerate(["id", "pr"]):

            dfx = df.query("setting == '{}'".format(sett))

            a = sns.pointplot(y=y, x=x, hue="model", join=False, data=dfx, \
                                dodge=dodge, scale=0.35, ax = ax[k], ci=None, palette=c)
            
            a.legend_.remove()
            if yticks:
                ax[k].set_yticklabels(yticks)

            n_io = len(df.io.unique())
            if chance:
                for z, ch in enumerate(chance):
                    ax[k].axvline(x=ch, ymin=z/len(chance),ymax=(z+1)/(len(chance)), label="chance", c='k', ls='--', lw=0.5)

            ax[k].set_xlabel("")
            ax[k].set_ylabel("")
            if not ticks:
                ax[k].tick_params(left=False)

            n_models = len(dfx["model"].unique())
            dodge_ = np.linspace(0, dodge, n_models) - np.linspace(0, dodge, n_models).mean() 

            for j, h in enumerate(dfx[y].unique()):
                for i, m in enumerate(dfx["model"].unique()):
                    dfxx = dfx.query("model == '{}' and {} == '{}'".format(m, y, h))
                    test_score = dfxx[x].values[0]
                    ci_low = dfxx[low].values[0]
                    ci_high = dfxx[high].values[0]
                    pvalue = dfxx[pval].values[0]
                    ast = return_asterisks(pvalue)
                    err = [[abs(test_score - ci_low)], [abs(test_score-ci_high)]]
                    ax[k].errorbar(test_score, j+dodge_[i], xerr=err, c=c[i], elinewidth=0.5)
                    ax[k].text(ci_high, j+dodge_[i], r"{}".format(ast), c=c[i], fontsize=9)

        ax[0].set_title("ID")
        ax[1].set_title("PR")

        handles, _ = ax[k].get_legend_handles_labels()
        handles = [handles[i] for i in len(chance)-1 + np.arange(len(dfx["model"].unique())+1)]
        fig.legend(handles, ["chance", "LR", "LSVM", "KSVM", "GB", "NB"], fontsize=8, frameon=False, loc="upper right")
        sns.despine()
        fig.tight_layout()

        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel(xlab)

            # Adjust the scaling factor to fit your legend text completely outside the plot
            # (smaller value results in more space being made for the legend)
        plt.subplots_adjust(right=0.85)

        return fig,ax