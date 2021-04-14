"""This class will have everything needed to create HDF5 input files to use for
any analysis in this IMAGEN project.
"""

# Global imports
import pandas as pd
import numpy as np 
from os.path import join 
import os
import glob
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import nibabel as nib
import h5py
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold



def _create_dataset(df, output_shape, z_factor, colname):
    """Create a numpy array from multiple MRI nifti images.

    Args:
        df (pd.DataFrame): DataFrame containing the paths to all nifti files.
        output_shape (array, list or tuple): Shape of output 
        z_factor (float): zoom factor for the nifti files.
        colname (str): column name of the list containing the paths to the nifti
            files.

    Returns:
        data_matrix (numpy array): array of len(df)*output_shape containing all
            opened nifti files. 
    """

    # Load mask 
    mask = nib.load("masks/MNI152_T1_1mm_bet_mask.nii.gz").get_fdata().astype(int)
    zoomed_mask = zoom(mask, z_factor).astype(int)

    # Load MRI voxels
    data_matrix = np.empty(shape=((len(df),) + output_shape))
    for idx, row in df.iterrows():
        path = row[colname]
        scan = nib.load(path)
        struct_arr = scan.get_fdata().astype(np.float32)
        struct_arr = zoom(struct_arr, z_factor)
        # The following line ensures that all values outside of the brain are 
        # exactly zero
        struct_arr = np.where(zoomed_mask != 0, struct_arr, 0)
        data_matrix[idx] = struct_arr
    return data_matrix


def _h5save(destination, X, y, s, c, groups, subj):
    """ Save HDF5 file. 

    Args:
        destination (str): Destination of HDF5 file
        X (np array): input data, FS-STATS or whole-brain
        y (np array): target variable
        s (np array): sex 
        c (np array): scanner site
        groups (np array): sexXsite information
        subj (np array): subject ID
    """    
    h5 = h5py.File(destination, "w")
    h5.create_dataset("X", data=X)
    h5.create_dataset('y', data=y)
    h5.create_dataset("s", data=s)
    h5.create_dataset("c", data=c)
    h5.create_dataset("group", data=groups)
    h5.create_dataset("subj", data=subj)
    h5.close()


class InputCreator:

    def __init__(self, DATA_DIR, tp_scan, tp_label, qtable, label, qtable_pr=None):
        self.DATA_DIR = DATA_DIR
        self.tp_scan = tp_scan
        self.tp_label = tp_label 
        if tp_scan != tp_label:
            assert qtable_pr, "The qtable at the scan time is needed!" 
        self.qtable_pr = qtable_pr
        self.qtable = qtable
        self.qtable_desc = os.path.basename(qtable).split("_")[1]
        self.label = label
        self.suffix = None

    def _give_save_name(self):
        """Generates a name under which to save the HDF5 file."""        
        if (type(self.lab0) == list) and (type(self.lab1) == list):
            left = "".join([str(y) for y in self.lab0])
            right = "".join([str(y) for y in self.lab1])
        elif (type(self.lab0) == int) and (type(self.lab1) == int):
            left = "lt{}".format(self.lab0)
            right = "mt{}".format(self.lab1)

        save_name = "_".join([
            self.tp_scan + "-" + self.tp_label, 
            "n{}".format(len(self.df))
        ])

        if self.suffix:
            save_name += "_{}".format(self.suffix)

        return(save_name)

    def _give_save_folder(self):
        """Creates the name of the folder in which to save the HDF5 file."""        
        if (type(self.lab0) == list) and (type(self.lab1) == list):
            left = "".join([str(y) for y in self.lab0])
            right = "".join([str(y) for y in self.lab1])
        elif (type(self.lab0) == int) and (type(self.lab1) == int):
            left = "lt{}".format(self.lab0)
            right = "mt{}".format(self.lab1)
        save_folder = "_".join([
            self.qtable_desc + self.label, 
            left, 
            right
        ])
        return(save_folder)


    def load_tc(self):
        """Loads the demographics with sex and site information."""        
        tc = pd.read_csv(join(self.DATA_DIR, "qs/FU3/participants/IMAGEN_demographics.csv"), dtype={"PSC2":str})
        tc = tc.rename(columns={"PSC2" : "ID", "recruitment centre":"center"})

        # Introduce a test centre code 
        d = {
            "BERLIN" : 1, 
            "DRESDEN" : 2,
            "DUBLIN" : 3, 
            "HAMBURG" : 4, 
            "LONDON" : 5, 
            "MANNHEIM" : 6, 
            "NOTTINGHAM" : 7, 
            "PARIS" : 8
        }
        d2 = {
            "M" : 0, 
            "F" : 1
        }
        tc["center_code"] = tc["center"].map(d)
        tc["sex"] = tc["sex"].map(d2)
        tc = tc.query("sex in [0, 1]")
        self.df = tc

    def load_psytools(self):
        """Loads the psytools questionnaire self.qtable column self.label."""        
        dfq = pd.read_csv(join(self.DATA_DIR, self.qtable), dtype={"User code":str})
        dfq = dfq.rename(columns={"User code":"ID"})
        dfq["ID"] = dfq["ID"].str.replace("-C", "")
        assert self.label in list(dfq), "{} is not a column in the chosen questionnaire.".format(self.label)
        dfq = dfq.loc[:, ["ID", self.label]].reset_index(drop=True)
        dfq = dfq.rename(columns={self.label:"q"})
        dfq.ID = dfq.ID.str.replace("-I", "")
        dfq.ID = dfq.ID.str.replace("-C", "")
        self.df = pd.merge(self.df, dfq)

    def make_label(self, lab0, lab1):
        """Creates a binary label from the label given in load_psytools.

        Args:
            lab0 (list or str): Turns subjects in or lower than lab0 into the 
                control group 0
            lab1 (list or str): Turns subjects in or higher than lab1 into the
                experimental group 1
        """        
        self.lab0, self.lab1 = lab0, lab1
        if (type(lab0) == list) and (type(lab1) == list):
            self.df = self.df.query("q in {} or q in {}".format(lab0, lab1))
            self.df = self.df.reset_index(drop=True)
            self.df["label"] = self.df.q > max(lab0)
        elif (type(lab0) == int) and (type(lab1) == int):
            self.df = self.df.query("q < {} or q > {}".format(lab0, lab1))
            self.df = self.df.reset_index(drop=True)
            self.df["label"] = self.df.q > lab0
        self.df.label = self.df.label.astype(int)

    def select_sex(self, sex):
        """Only include members of one sex for analysis.

        Args:
            sex (int, 0 or 1): sex to keep for analysis: 1 is female
                and 0 is male
        """        
        self.df = self.df.query("sex == {}".format(sex))
        self.suffix = "sex{}".format(sex)


    def select_site(self, sites):
        """Only include subjects of one or more scanner sites.

        Args:
            sites (list): List of sites to keep: see self.load_tc for codes.
        """        
        self.df = self.df.query("center_code in @sites")
        self.suffix = "site{}".format("".join([str(c) for c in sites]))


    def pr_exclude(self):
        """Exclude those subjects that are not in the control group at age 14."""        
        dfq = pd.read_csv(join(self.DATA_DIR, self.qtable_pr), dtype={"User code":str})
        dfq = dfq.rename(columns={"User code":"ID"})
        dfq["ID"] = dfq["ID"].str.replace("-C", "")
        assert self.label in list(dfq), "{} is not a column in the chosen questionnaire.".format(self.label)
        dfq = dfq.loc[:, ["ID", self.label]].reset_index(drop=True)
        dfq = dfq.rename(columns={self.label:"qpr"})
        self.df = pd.merge(self.df, dfq)
        if (type(self.lab0) == list):
            self.df = self.df.query("qpr in {}".format(self.lab0))
        elif (type(self.lab0) == int):
            self.df = self.df.query("qpr < {}".format(self.lab0))

    def add_wb_path(self, bids_csv, colname, tp_scan=None):
        """Add the paths to whole-brain nifti files.

        Args:
            bids_csv (str): Path to the participants.csv in a BIDS folder. 
            colname (str): Column name in the participants.csv corresponding to 
                the nifti paths
            tp_scan (str, optional): Time point for the scan, use when different 
                than specified before. Defaults to None.
        """        
        if not tp_scan:
            tp_scan = self.tp_scan
        ppts = pd.read_csv(join(self.DATA_DIR, bids_csv), dtype={"ID":str})
        ppts = ppts.query("time_point == '{}'".format(tp_scan))
        ppts = ppts[["ID", "path"]]
        ppts = ppts.rename(columns={"path":colname})
        self.df = pd.merge(self.df, ppts)


    def save_wb(self, output_shape, z_factor, colname, del_zeros=False, tp_scan=None):
        """Save whole-brain dataset

        Args:
            output_shape (np array, list or tuple): Size of desired WB scans
            z_factor (float): zoom factor, should correspond to output_shape
            colname (str): column name of nifti file paths
            del_zeros (bool, optional): If True, values outside the brain (zero
                by default) are deleted. Defaults to False.
            tp_scan (str, optional): Time point of scan if different from specified
                in __init__. Defaults to None.
        """        
        if tp_scan:
            self.tp_scan = tp_scan

        self.save_name = self._give_save_name()
        self.save_folder = self._give_save_folder()

        mri_type = colname.replace("path_", "")
        new_folder = join(self.DATA_DIR, "h5files", self.save_folder, mri_type)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)

        print("Downloading data from server...")
        X = _create_dataset(self.df, output_shape, z_factor, colname)
        if del_zeros:
            X = X.reshape([X.shape[0], np.prod(X.shape[1:])])
            X = VarianceThreshold().fit_transform(X)
        y = self.df.label.values 
        sex = self.df.sex.values 
        center = self.df.center_code.values 
        group = sex * 10 + center
        subject = self.df["ID"].values.astype(int)

        suffix = "_d0" if del_zeros else ""
        
        dest = join(new_folder, "{}_{}_z{}{}.h5".format(mri_type, self.save_name, z_factor, suffix))
        print("Saving at {}".format(dest))
        _h5save(dest, X, y, sex, center, group, subject)
        

    def save_fs_stats(self, tp_scan=None):
        """Save the Freesurfer Statistics (FSS).

        Args:
            tp_scan (str, optional): Time point of scan if different from specified
                in __init__. Defaults to None.
        """        
        if tp_scan:
            self.tp_scan = tp_scan
        # Save ppts csv
        self.save_name = self._give_save_name()
        self.save_folder = self._give_save_folder()
        new_folder = join(self.DATA_DIR, "h5files", self.save_folder, "fs-stats")
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        self.df.to_csv(join(new_folder, "fs-stats_{}.csv".format(self.save_name)))

        # Selects all the tsv files in the freesurfer folder 
        files = glob.glob(join(self.DATA_DIR, "IMAGEN_RAW/2.7/{}/imaging/freesurfer/*.tsv".format(self.tp_scan)))
        # Excluded euler.tsv that has a very different format
        files.remove(join(self.DATA_DIR, "IMAGEN_RAW/2.7/{}/imaging/freesurfer/euler.tsv".format(self.tp_scan)))
        fs_stats = pd.DataFrame(columns=["ID"])

        for f in files:
            dfx = pd.read_csv(f, sep="\t")
            id_col = list(dfx)[0]
            dfx = pd.read_csv(f, sep="\t", dtype={id_col:str})
            dfx = dfx.rename(columns={id_col:"ID"})
            dfx["ID"] = dfx["ID"].astype(str)
            fs_stats = fs_stats.merge(dfx, "outer", on="ID")
        
        self.df = pd.merge(self.df, fs_stats)

        # Then save this bitch 
        roi_names = list(self.df.loc[:,"path_T1w":])
        roi_names.remove("path_T1w")
        X = self.df[roi_names].values 
        y = self.df.label.values 
        sex = self.df.sex.values 
        center = self.df.center_code.values 
        group = sex * 10 + center
        subject = self.df["ID"].values.astype(int)

        dest = join(new_folder, "fs-stats_{}.h5".format(self.save_name))
        print("Saving at {}".format(dest))

        _h5save(dest, X, y, sex, center, group, subject)


    def save_volumes(self, tp_scan=None):
        """Save only the white and gray matter volumes from the FSS.

        Args:
            tp_scan (str, optional): Time point of scan if different from specified
                in __init__. Defaults to None.
        """        
        if tp_scan:
            self.tp_scan = tp_scan
        # Save ppts csv
        self.save_name = self._give_save_name()
        self.save_folder = self._give_save_folder()
        new_folder = join(self.DATA_DIR, "h5files", self.save_folder, "volumes")
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        self.df.to_csv(join(new_folder, "volumes_{}.csv".format(self.save_name)))

        files = glob.glob(join(self.DATA_DIR, "IMAGEN_RAW/2.7/{}/imaging/freesurfer/*volume.tsv".format(self.tp_scan)))
        volumes = pd.DataFrame(columns=["ID"])

        for f in files:
            dfx = pd.read_csv(f, sep="\t")
            id_col = list(dfx)[0]
            dfx = pd.read_csv(f, sep="\t", dtype={id_col:str})
            dfx = dfx.rename(columns={id_col:"ID"})
            dfx["ID"] = dfx["ID"].astype(str)
            volumes = volumes.merge(dfx, "outer", on="ID")

        self.df = pd.merge(self.df, volumes)

        # Then save this bitch 
        roi_names = list(self.df.loc[:,"path_T1w":])
        roi_names.remove("path_T1w")
        X = self.df[roi_names].values 
        y = self.df.label.values 
        sex = self.df.sex.values 
        center = self.df.center_code.values 
        group = sex * 10 + center
        subject = self.df["ID"].values.astype(int)

        dest = join(new_folder, "volumes_{}.h5".format(self.save_name))
        print("Saving at {}".format(dest))

        _h5save(dest, X, y, sex, center, group, subject)