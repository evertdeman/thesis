# Analysis 

This directory is for running the classification analyses. 

Files: 
- `analysis.py` to run the analysis, change parameters, filenames etc, and run `python analysis.py` on the command line. Results will appear in the directory `analysis/results`;
- `confounds.py` contains classes to implement various confound-correction techniques; 
- `mri_learn_quick.py` contains the class `MRILearn` that executes the actual classification pipeline and permutation tests; 
- `fnames.json` is a file containing all feature names in the Freesurfer statistics input data; 
- `random_states.npy` contains random states to keep analysis reproducible. 
