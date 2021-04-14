"""This file prep.py is created to preprocess raw MRI images by registration to MNI 
space with ANTs.
"""

import numpy as np
import os
import time as t
import nibabel as nib
from os.path import join
import re
import matplotlib.pyplot as plt
import ants
import multiprocessing as mp
import pandas as pd


class PREP:
    def __init__(self, mri_scan, destination):
        """Initialize PREP class

        Args:
            mri_scan (str): full path to nifti (or even mgz) file that should be
                registered.
            destination (str): full path to destination of preprocessed output 
                file.
        """        
        self.mri_scan = mri_scan
        self.file_name = mri_scan.split("/")[-1].split(".mgz")[0]
        self.destination = destination  

        # Create folders for the output files
        if not os.path.exists(self.destination):
            os.makedirs(self.destination)

    def normalize_py(self, mask):
        """Registers (normalizes) the mri_scan from __init__ to MNI space with 
        antspyx. Uses nonlinear registration SyN which can be changed. 

        Args:
            mask (str): path to mask to use for registration. Typically an MNI 
                brain in ./masks/
        """        
        print("File: {}: starting normalization.".format(self.file_name))
        fixed = ants.image_read(mask)
        moving = ants.image_read(self.mri_scan)

        start = t.time()
        mytx = ants.registration(fixed=fixed , moving=moving, type_of_transform='SyN', outprefix=self.file_name)
        warped_moving = mytx["warpedmovout"]
        print("File {}: normalization complete. That took {} seconds.".format(self.file_name, t.time()-start))
        self.warped_moving = warped_moving

        save_loc = join(self.destination, self.file_name + "_warped.nii.gz")
        warped_moving.to_file(save_loc)

        # ANTs creates these files that we won't need any longer
        os.system("rm {}1Warp.nii.gz".format(self.file_name))
        os.system("rm {}1InverseWarp.nii.gz".format(self.file_name))
        os.system("rm {}0GenericAffine.mat".format(self.file_name))


if __name__ == "__main__":
    DATA_DIR = "/ritter/share/data/IMAGEN/IMAGEN_freesurfer_BIDS"
    mask = "./masks/MNI152_T1_1mm_brain.nii"

    # Load the participants.csv of the unprocessed images 
    ppts = pd.read_csv(join(DATA_DIR, "participants_brainmask.csv"), dtype={"ID":str}, index_col=0)
    ppts = ppts.drop("old_path", axis=1)

    ppts["new_path"] = ppts.path.str.replace(".mgz", "_warped.nii.gz")
    ppts["new_path"] = ppts.new_path.str.replace("IMAGEN_freesurfer_BIDS", "IMAGEN_prep-brainmask_BIDS")

    i=0
    for fname in ppts.new_path.tolist():
        if os.path.isfile(fname):
            #print("{} exists".format(fname))
            i+=1 
            
    print("{} scans were already preprocessed.".format(i))
    mri_scans = ppts.path.tolist()
    ppts.dir = ppts.dir.str.replace("IMAGEN_freesurfer_BIDS", "IMAGEN_prep-brainmask_BIDS")
    destinations = ppts.dir.tolist()
    new_paths = ppts.new_path.tolist()

    def wrapper(i):
        new_path = new_paths[i]
        if not os.path.isfile(new_path):
            prep = PREP(mri_scans[i], destinations[i])
            prep.normalize_py(mask)
        else:
            pass

    with mp.Pool(20) as p:
        p.map(wrapper,range(len(mri_scans)))