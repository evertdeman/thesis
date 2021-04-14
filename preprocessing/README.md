# Preprocessing 

This directory deals with preprocessing whole-brain images. 
The only preprocessing step I take for my thesis is registering T1-weighted Freesurfer `brainmask.mgz` images to MNI space. 

Steps:
1. `bids.ipynb` is there to move raw nifti/mgz files from the copy of the IMAGEN data on our server into BIDS format; 
2. `prep.py` let you preprocess the images from the newly created BIDS folder into a new BIDS folder with the preprocessed images; 
3. `ppts_csv.ipynb` is there to create a `participants.csv` file for the newly created BIDS folder with preprocessed data.

The folder `masks` contains MNI brains, used for registration in `prep.py`.
